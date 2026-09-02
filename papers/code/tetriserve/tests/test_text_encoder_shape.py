import os
import json
import torch
import random

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


def test_run(rank, world_size, master_port, model_name):
    runner = init_runner(rank, world_size, master_port, model_name)

    for batch_size in [1, 2, 4]:
        for height in [512, 1024, 2048]:
            for width in [512, 1024, 2048]:

                input_config = InputConfig(
                    batch_size=batch_size,
                    height=height,
                    width=width,
                    prompt=["a beautiful image"] * batch_size,
                    num_inference_steps=10,
                    output_type="latent",
                    seed=42,
                )

                prompt_embeds, pooled_prompt_embeds, latents, timesteps = runner.pipeline.inference_prologue(
                    prompt=input_config.prompt,
                    height=input_config.height,
                    width=input_config.width,
                    num_inference_steps=input_config.num_inference_steps,
                    output_type=input_config.output_type,
                    generator=torch.Generator(device="cuda").manual_seed(input_config.seed),
                )

                print(f"batch_size: {batch_size}, height: {height}, width: {width}")
                print(f"===shape of prompt_embeds: {prompt_embeds.shape}, type: {prompt_embeds.dtype}")
                print(
                    f"===shape of pooled_prompt_embeds: {pooled_prompt_embeds.shape}, type: {pooled_prompt_embeds.dtype}"
                )
                print(f"===shape of latents: {latents.shape}, type: {latents.dtype}")
                print(f"===shape of timesteps: {timesteps.shape}, type: {timesteps.dtype}")
                print()


"""
python tests/test_text_encoder_shape.py 2>&1 | tee log/test_text_encoder_shape.log
"""

if __name__ == "__main__":
    from torch.multiprocessing import spawn
    from tetriserve.zmq_utils import find_free_port

    world_size = 1
    master_port = find_free_port()
    model_name = "sd3"

    spawn(test_run, args=(world_size, master_port, model_name), nprocs=world_size)
