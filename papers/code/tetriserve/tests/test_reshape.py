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

from tetriserve import SUPPORTED_MODEL_DICT


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
            model=SUPPORTED_MODEL_DICT[model_name],
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

    input_config = InputConfig(
        batch_size=1,
        height=512,
        width=512,
        prompt=["a beautiful image"] * 1,
        num_inference_steps=10,
        output_type="latent",
        seed=42,
    )

    prompt_embeds, pooled_prompt_embeds, latents, timesteps, sigmas = runner.pipeline.inference_prologue(
        prompt=input_config.prompt,
        height=input_config.height,
        width=input_config.width,
        num_inference_steps=input_config.num_inference_steps,
        output_type=input_config.output_type,
        generator=torch.Generator(device="cuda").manual_seed(input_config.seed),
    )

    print(f"shape of prompt_embeds: {prompt_embeds.shape}")
    print(f"shape of pooled_prompt_embeds: {pooled_prompt_embeds.shape}")
    print(f"shape of latents: {latents.shape}")
    print(f"timesteps: {timesteps}")

    output = runner.pipeline.inference_denoising(
        height=input_config.height,
        width=input_config.width,
        batch_size=input_config.batch_size,
        num_inference_steps=input_config.num_inference_steps,
        timesteps=timesteps,
        latents=latents,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        sigmas=sigmas,
        is_last_steps=[False] * input_config.batch_size,
    )

    ref_output = runner.pipeline(
        prompt=input_config.prompt,
        height=input_config.height,
        width=input_config.width,
        num_inference_steps=input_config.num_inference_steps,
        output_type=input_config.output_type,
        generator=torch.Generator(device="cuda").manual_seed(input_config.seed),
    )

    ref_output_1 = runner.pipeline(
        prompt=input_config.prompt,
        height=input_config.height,
        width=input_config.width,
        num_inference_steps=input_config.num_inference_steps,
        output_type=input_config.output_type,
        generator=torch.Generator(device="cuda").manual_seed(input_config.seed),
    )

    print(f"shape of ref_output latent: {ref_output.images.shape}")

    # Use a more lenient comparison with rtol and atol
    assert torch.allclose(ref_output.images, ref_output_1.images, rtol=1e-3, atol=1e-3)
    assert torch.allclose(output.images, ref_output.images, rtol=1e-3, atol=1e-3)
    # assert torch.allclose(output_1.images, ref_output_1.images, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    from torch.multiprocessing import spawn
    from tetriserve.zmq_utils import find_free_port

    world_size = 1
    master_port = find_free_port()
    # model_name = "sd3"
    model_name = "flux"

    spawn(test_run, args=(world_size, master_port, model_name), nprocs=world_size)
