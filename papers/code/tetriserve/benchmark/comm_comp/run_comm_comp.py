import torch
import os
import json
import torch
import random
import numpy as np

from tetriserve.xfuser.config.config import (
    EngineConfig,
    ModelConfig,
    RuntimeConfig,
    ParallelConfig,
    SequenceParallelConfig,
    InputConfig,
    DataParallelConfig,
    PipeFusionParallelConfig,
    TensorParallelConfig,
    FastAttnConfig,
)
from tetriserve.distributed import init_distributed_environment
from tetriserve.engine.runner import FluidServeRunner
from tetriserve.config.request import ScheduledRequests
from tetriserve.distributed.comm import all_to_all_4D
from tetriserve.xfuser.core.long_ctx_attention.hybrid.attn_layer import xFuserLongContextAttention

all_to_all_4D_begin = torch.cuda.Event(enable_timing=True)
all_to_all_4D_end = torch.cuda.Event(enable_timing=True)
all_to_all_4D_times = []


def new_all_to_all_4D(
    self: xFuserLongContextAttention,
    input: torch.tensor,
    scatter_idx: int = 2,
    gather_idx: int = 1,
    group=None,
    use_sync: bool = False,
):
    global all_to_all_4D_begin, all_to_all_4D_end, all_to_all_4D_times
    all_to_all_4D_begin.record()
    output = all_to_all_4D(input, scatter_idx, gather_idx, group=group, use_sync=use_sync)
    all_to_all_4D_end.record()
    torch.cuda.synchronize()
    all_to_all_4D_times.append(all_to_all_4D_begin.elapsed_time(all_to_all_4D_end) / 1000.0)
    return output


MODEL_DICT = {
    "flux": "black-forest-labs/FLUX.1-dev",
    "sd3": "stabilityai/stable-diffusion-3-medium-diffusers",
}

HEIGHT_WIDTH_DICT = {
    "flux": [(256, 256), (512, 512), (1024, 1024), (2048, 2048)],
    "sd3": [(256, 256), (256, 512), (512, 512), (512, 1024), (1024, 1024), (1024, 2048), (2048, 2048)],
}


def init_runner(rank, world_size, master_port, model_name):
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(master_port)

    init_distributed_environment(
        rank=rank,
        local_rank=rank,
        world_size=world_size,
        distributed_init_method=f"tcp://localhost:{master_port}",
        backend="nccl",
    )

    torch.cuda.set_device(rank)

    engine_config = EngineConfig(
        model_config=ModelConfig(
            model=MODEL_DICT[model_name],
        ),
        runtime_config=RuntimeConfig(
            dtype=torch.float16,
            use_torch_compile=False,
            use_parallel_vae=False,
        ),
        parallel_config=ParallelConfig(
            sp_config=SequenceParallelConfig(
                ulysses_degree=world_size,
                ring_degree=1,
            ),
            dp_config=DataParallelConfig(
                dp_degree=1,
            ),
            pp_config=PipeFusionParallelConfig(
                pp_degree=1,
            ),
            tp_config=TensorParallelConfig(
                tp_degree=1,
            ),
            dit_parallel_size=world_size,
            vae_parallel_size=0,
        ),
        fast_attn_config=FastAttnConfig(
            use_fast_attn=False,
        ),
    )

    xFuserLongContextAttention.all_to_all_4d = new_all_to_all_4D

    runner = FluidServeRunner(
        pretrained_model_name_or_path=engine_config.model_config.model, engine_config=engine_config
    )

    return runner


def run_e2e(rank, world_size, master_port, args):
    runner = init_runner(rank, world_size, master_port, args.model)

    json_path = f"{args.output_dir}/{args.model}/detailed_comm_comp_time_{world_size}.json"

    existing_data = []
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            existing_data = json.load(f)

    # Create all combinations and sort by their product
    combinations = []
    for batch_size in [4]:
        for height, width in HEIGHT_WIDTH_DICT[args.model]:
            combinations.append((batch_size, height, width, batch_size * height * width))

    combinations.sort(key=lambda x: x[-1])

    for batch_size, height, width, _ in combinations:

        input_config = InputConfig(
            batch_size=batch_size,
            height=height,
            width=width,
            prompt=["a beautiful image"] * batch_size,
            num_inference_steps=20,
            output_type="latent",
            seed=42,
        )

        for ulysses_degree in [1, 2, 4, 8]:
            for ring_degree in [1]:
                if ulysses_degree * ring_degree != world_size:
                    continue

                if args.logging:
                    skip = False
                    # if this item is already in the log, skip
                    for item in existing_data:
                        if (
                            item["bs"] == batch_size
                            and item["height"] == height
                            and item["width"] == width
                            and item["ulysses_degree"] == ulysses_degree
                            and item["ring_degree"] == ring_degree
                        ):
                            print(
                                f"skip bs {batch_size}, height {height}, width {width}, ulysses_degree {ulysses_degree}, ring_degree {ring_degree}, world_size {world_size}"
                            )
                            skip = True
                            break

                    if skip:
                        continue
                    else:
                        print(
                            f"run bs {batch_size}, height {height}, width {width}, ulysses_degree {ulysses_degree}, ring_degree {ring_degree}, world_size {world_size}"
                        )

                sched_reqs = ScheduledRequests(
                    req_ids=[0],
                    sub_req_ids=list(range(-1, -batch_size - 1, -1)),
                    parallel_ranks=list(range(ulysses_degree * ring_degree)),
                    attn_ulysses_degree=ulysses_degree,
                    attn_ring_degree=ring_degree,
                    sp_degree=ulysses_degree * ring_degree,
                    latent_length=None,
                    input_config=input_config,
                    is_last_steps=[True] * batch_size,
                    arrival_time=0,
                    latency_threshold=None,
                    estimated_time=0,
                    origin_steps=[input_config.num_inference_steps] * batch_size,
                    start_step=0,
                    end_step=input_config.num_inference_steps,
                )

                # Warmup
                if rank == 0:
                    print(f"Warming up for bs {batch_size}, height {height}, width {width}...")
                for _ in range(10):  # Increased warmup runs
                    runner.generate(sched_reqs)
                torch.cuda.synchronize()

                # Collect detailed timing data for each iteration
                individual_e2e_times = []
                individual_comm_times = []
                individual_comp_times = []

                if rank == 0:
                    print(f"Running {args.repeat} iterations for detailed timing...")

                for iteration in range(args.repeat):
                    global all_to_all_4D_times
                    all_to_all_4D_times.clear()

                    # Timing for this iteration
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)

                    start_event.record()
                    runner.generate(sched_reqs)
                    end_event.record()

                    torch.cuda.synchronize()
                    e2e_time = start_event.elapsed_time(end_event) / 1000.0  # Convert to seconds
                    comm_time = sum(all_to_all_4D_times)
                    comp_time = e2e_time - comm_time

                    individual_e2e_times.append(e2e_time)
                    individual_comm_times.append(comm_time)
                    individual_comp_times.append(comp_time)

                    if rank == 0 and (iteration + 1) % 10 == 0:
                        print(f"  Completed {iteration + 1}/{args.repeat} iterations")

                if rank == 0:
                    # Calculate statistics
                    e2e_avg = np.mean(individual_e2e_times)
                    e2e_std = np.std(individual_e2e_times)
                    comm_avg = np.mean(individual_comm_times)
                    comm_std = np.std(individual_comm_times)
                    comp_avg = np.mean(individual_comp_times)
                    comp_std = np.std(individual_comp_times)

                    print(
                        f"batch_size: {batch_size}, height: {height}, width: {width}, "
                        f"ulysses_degree: {ulysses_degree}, ring_degree: {ring_degree}, "
                        f"iterations: {args.repeat}"
                    )
                    print(f"  E2E Time - Avg: {e2e_avg:.4f}s, Std: {e2e_std:.4f}s")
                    print(f"  Comm Time - Avg: {comm_avg:.4f}s, Std: {comm_std:.4f}s")
                    print(f"  Comp Time - Avg: {comp_avg:.4f}s, Std: {comp_std:.4f}s")

                    if args.logging:
                        to_save = {
                            "bs": batch_size,
                            "height": height,
                            "width": width,
                            "ulysses_degree": ulysses_degree,
                            "ring_degree": ring_degree,
                            "mlp_world_size": world_size,
                            "num_iter": args.repeat,
                            "steps": input_config.num_inference_steps,
                            # Individual measurements
                            "individual_e2e_times": individual_e2e_times,
                            "individual_comm_times": individual_comm_times,
                            "individual_comp_times": individual_comp_times,
                            # Statistics
                            "e2e_time_avg": e2e_avg,
                            "e2e_time_std": e2e_std,
                            "comm_time_avg": comm_avg,
                            "comm_time_std": comm_std,
                            "comp_time_avg": comp_avg,
                            "comp_time_std": comp_std,
                            # Legacy fields for compatibility
                            "avg_e2e_time": e2e_avg,
                            "e2e_time": sum(individual_e2e_times),
                            "all_to_all_4D_time": sum(individual_comm_times),
                            "avg_all_to_all_4D_time": comm_avg,
                        }

                        # Create directory if it doesn't exist
                        os.makedirs(os.path.dirname(json_path), exist_ok=True)

                        # Append new data
                        if not isinstance(existing_data, list):
                            existing_data = [existing_data]
                        existing_data.append(to_save)

                        # Save updated data
                        with open(json_path, "w") as f:
                            json.dump(existing_data, f, indent=2)


if __name__ == "__main__":
    from torch.multiprocessing import spawn
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-gpu-num", "-g", type=int, default=2)
    parser.add_argument("--model", "-m", type=str, default="sd3")
    parser.add_argument("--repeat", type=int, default=100)  # Changed default to 100
    parser.add_argument("--logging", action="store_true", default=False)
    parser.add_argument("--output-dir", type=str, default="log/benchmark/comm_comp")
    args = parser.parse_args()

    nprocs = args.gpu_num

    spawn(
        run_e2e,
        args=(
            nprocs,
            random.randint(8000, 65535),
            args,
        ),
        nprocs=nprocs,
    )
