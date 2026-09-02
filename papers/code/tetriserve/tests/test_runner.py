import os
import json
import torch
import random
from copy import deepcopy
import multiprocessing as mp

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


def init_runner(rank, world_size, master_port, model_name, latent_request_queues):
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
        pretrained_model_name_or_path=engine_config.model_config.model,
        engine_config=engine_config,
        request_queues=latent_request_queues,
    )

    return runner


def run_one_req(rank, world_size, master_port, args):
    runner = init_runner(rank, world_size, master_port, args.model, args.latent_request_queues)
    ulysses_degree = world_size
    ring_degree = 1

    input_config = InputConfig(
        batch_size=args.batch_size,
        height=args.height,
        width=args.width,
        prompt=["a beautiful image"] * args.batch_size,
        num_inference_steps=10,
        output_type="latent",
        seed=42,
    )

    input_config_1 = deepcopy(input_config)
    input_config_1.num_inference_steps = 5

    sched_reqs = ScheduledRequests(
        req_ids=[0],
        sub_req_ids=list(range(-1, -args.batch_size - 1, -1)),
        parallel_ranks=list(range(ulysses_degree * ring_degree)),
        attn_ulysses_degree=ulysses_degree,
        attn_ring_degree=ring_degree,
        sp_degree=ulysses_degree * ring_degree,
        latent_length=None,
        input_config=input_config_1,
        origin_steps=[10] * args.batch_size,
        start_step=0,
        end_step=input_config_1.num_inference_steps,
        is_last_steps=[False] * args.batch_size,
        arrival_time=0,
        latency_threshold=None,
        estimated_time=0,
    )

    ret = runner.generate(reqs=sched_reqs)
    print(ret)


"""
python tests/test_runner.py -g 2 -m sd3 --batch-size 2 --height 512 --width 512
python tests/test_runner.py -g 2 -m flux --batch-size 2 --height 512 --width 512
"""

if __name__ == "__main__":
    from torch.multiprocessing import spawn
    import argparse

    # Set multiprocessing start method to spawn for compatibility with torch.multiprocessing
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("-gpu-num", "-g", type=int, default=2)
    parser.add_argument("--model", "-m", type=str, default="sd3")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--logging", action="store_true", default=False)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    args = parser.parse_args()

    nprocs = args.gpu_num

    # Create queues using the proper multiprocessing context
    ctx = mp.get_context("spawn")
    args.latent_request_queues = [ctx.Queue() for _ in range(nprocs)]

    spawn(
        run_one_req,
        args=(
            nprocs,
            random.randint(8000, 65535),
            args,
        ),
        nprocs=nprocs,
    )
