import os
import json
import torch
import random
import numpy as np
from typing import List, Tuple

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

MODEL_DICT = {
    "flux": "black-forest-labs/FLUX.1-dev",
    "sd3": "stabilityai/stable-diffusion-3-medium-diffusers",
}

HEIGHT_WIDTH_DICT = {
    "flux": [(256, 256), (512, 512), (1024, 1024), (2048, 2048), (4096, 4096)],
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

    runner = FluidServeRunner(
        pretrained_model_name_or_path=engine_config.model_config.model, engine_config=engine_config
    )

    return runner


def run_e2e_stats(rank, world_size, master_port, args):
    runner = init_runner(rank, world_size, master_port, args.model)

    json_path = f"log/benchmark/scaling_efficiency/{args.model}/stats/e2e_scaling_efficiency_stats_{world_size}.json"

    existing_data = []
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            existing_data = json.load(f)

    # Create all combinations and sort by their product
    combinations = []
    for batch_size in [1, 2, 4]:
        for height, width in HEIGHT_WIDTH_DICT[args.model]:
            combinations.append((batch_size, height, width, batch_size * height * width))

    combinations.sort(key=lambda x: x[-1])

    for batch_size, height, width, _ in combinations:
        input_config = InputConfig(
            batch_size=batch_size,
            height=height,
            width=width,
            prompt=["a beautiful image"] * batch_size,
            num_inference_steps=5,
            output_type="latent",
            seed=42,
        )

        for ulysses_degree in [1, 2, 4, 8]:
            for ring_degree in [1, 2, 4, 8]:
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
                    origin_steps=[input_config.num_inference_steps] * batch_size,
                    start_step=0,
                    end_step=input_config.num_inference_steps,
                    is_last_steps=[True] * batch_size,
                    arrival_time=0,
                    latency_threshold=None,
                    estimated_time=0,
                )

                # Warmup
                if rank == 0:
                    print(f"Warming up for bs {batch_size}, height {height}, width {width}...")
                for _ in range(5):  # Do 5 warmup runs
                    runner.generate(sched_reqs)
                torch.cuda.synchronize()

                prompt_embeds, pooled_prompt_embeds, latents, timesteps, sigmas = runner.pipeline.inference_prologue(
                    prompt=sched_reqs.input_config.prompt,
                    height=sched_reqs.input_config.height,
                    width=sched_reqs.input_config.width,
                    num_inference_steps=sched_reqs.input_config.num_inference_steps,
                    output_type=sched_reqs.input_config.output_type,
                )

                # Run 100 times for statistics
                durations = []
                for i in range(100):
                    if rank == 0 and i % 10 == 0:
                        print(f"Running iteration {i}/100...")

                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)

                    start_event.record()
                    runner.pipeline.inference_denoising(
                        height=sched_reqs.input_config.height,
                        width=sched_reqs.input_config.width,
                        batch_size=sched_reqs.input_config.batch_size,
                        num_inference_steps=sched_reqs.input_config.num_inference_steps,
                        timesteps=timesteps,
                        latents=latents,
                        prompt_embeds=prompt_embeds,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        sigmas=sigmas,
                        output_type=sched_reqs.input_config.output_type,
                    )
                    end_event.record()

                    torch.cuda.synchronize()
                    duration = start_event.elapsed_time(end_event) / 1000.0  # Convert to seconds
                    durations.append(duration)

                if rank == 0:
                    # Calculate statistics
                    mean_duration = np.mean(durations)
                    std_duration = np.std(durations)
                    var_duration = np.var(durations)
                    min_duration = np.min(durations)
                    max_duration = np.max(durations)
                    p95_duration = np.percentile(durations, 95)
                    p99_duration = np.percentile(durations, 99)

                    print(
                        f"batch_size: {batch_size}, height: {height}, width: {width}, "
                        f"ulysses_degree: {ulysses_degree}, ring_degree: {ring_degree}, "
                        f"mean: {mean_duration:.2f}s, std: {std_duration:.2f}s, "
                        f"min: {min_duration:.2f}s, max: {max_duration:.2f}s, "
                        f"p95: {p95_duration:.2f}s, p99: {p99_duration:.2f}s"
                    )

                    if args.logging:
                        to_save = {
                            "bs": batch_size,
                            "height": height,
                            "width": width,
                            "ulysses_degree": ulysses_degree,
                            "ring_degree": ring_degree,
                            "mlp_world_size": world_size,
                            "num_iter": 100,
                            "steps": input_config.num_inference_steps,
                            "mean_denoising_duration": mean_duration,
                            "std_denoising_duration": std_duration,
                            "var_denoising_duration": var_duration,
                            "min_denoising_duration": min_duration,
                            "max_denoising_duration": max_duration,
                            "p95_denoising_duration": p95_duration,
                            "p99_denoising_duration": p99_duration,
                            "all_denoising_durations": durations,
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
    parser.add_argument("--logging", action="store_true", default=False)
    args = parser.parse_args()

    nprocs = args.gpu_num

    spawn(
        run_e2e_stats,
        args=(
            nprocs,
            random.randint(8000, 65535),
            args,
        ),
        nprocs=nprocs,
    )
