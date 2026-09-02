"""
Test: 2-step inference on a single GPU (SD3).
"""

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tetriserve.distributed import init_distributed_environment
from tetriserve.engine.runner import FluidServeRunner
from tetriserve.config.request import ScheduledRequests
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
from tests.utils.image_compare import compare_images


MODEL_ID = "stabilityai/stable-diffusion-3-medium-diffusers"
PROMPT = "a photo of an astronaut riding a horse on mars"
SEED = 42
HEIGHT = 512
WIDTH = 512
TOTAL_STEPS = 28
FIRST_STAGE_STEPS = TOTAL_STEPS // 2
SECOND_STAGE_STEPS = TOTAL_STEPS - FIRST_STAGE_STEPS


def _free_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _init_runner(rank: int, world_size: int, master_port: int, request_queues):
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
            model=MODEL_ID,
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

    return FluidServeRunner(
        pretrained_model_name_or_path=engine_config.model_config.model,
        engine_config=engine_config,
        request_queues=request_queues,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_multistep_single_gpu_matches_baseline():
    rank = 0
    world_size = 1
    master_port = _free_port()
    request_queues = [mp.Queue()]

    runner = _init_runner(rank, world_size, master_port, request_queues)

    try:
        base_input_config = InputConfig(
            batch_size=1,
            height=HEIGHT,
            width=WIDTH,
            prompt=[PROMPT],
            num_inference_steps=FIRST_STAGE_STEPS,
            output_type="latent",
            seed=SEED,
        )

        single_step_config = InputConfig(
            batch_size=1,
            height=HEIGHT,
            width=WIDTH,
            prompt=[PROMPT],
            num_inference_steps=TOTAL_STEPS,
            output_type="pil",
            seed=SEED,
        )

        sched_single = ScheduledRequests(
            req_ids=[0],
            sub_req_ids=[-1],
            parallel_ranks=[0],
            attn_ulysses_degree=1,
            attn_ring_degree=1,
            sp_degree=1,
            latent_length=None,
            input_config=single_step_config,
            origin_steps=[TOTAL_STEPS],
            start_step=0,
            end_step=TOTAL_STEPS,
            is_last_steps=[True],
            arrival_time=0,
            latency_threshold=None,
            estimated_time=0,
        )

        single_step_images = runner.generate(reqs=sched_single, output_type="pil")
        single_step_image = single_step_images[0]

        sched_step1 = ScheduledRequests(
            req_ids=[0],
            sub_req_ids=[-1],
            parallel_ranks=[0],
            attn_ulysses_degree=1,
            attn_ring_degree=1,
            sp_degree=1,
            latent_length=None,
            input_config=base_input_config,
            origin_steps=[TOTAL_STEPS],
            start_step=0,
            end_step=FIRST_STAGE_STEPS,
            is_last_steps=[False],
            arrival_time=0,
            latency_threshold=None,
            estimated_time=0,
        )

        step1_outputs = runner.generate(reqs=sched_step1, output_type="latent")
        latent_future = step1_outputs[0]

        step2_input_config = InputConfig(
            batch_size=1,
            height=HEIGHT,
            width=WIDTH,
            prompt=[PROMPT],
            num_inference_steps=SECOND_STAGE_STEPS,
            output_type="pil",
            seed=SEED,
        )

        sched_step2 = ScheduledRequests(
            req_ids=[0],
            sub_req_ids=[-1],
            parallel_ranks=[0],
            attn_ulysses_degree=1,
            attn_ring_degree=1,
            sp_degree=1,
            latent_length=None,
            input_config=step2_input_config,
            origin_steps=[TOTAL_STEPS],
            start_step=FIRST_STAGE_STEPS,
            end_step=TOTAL_STEPS,
            is_last_steps=[True],
            arrival_time=0,
            latency_threshold=None,
            estimated_time=0,
        )

        output_images = runner.generate(reqs=sched_step2, latent_future_dict={-1: latent_future})
        output_image = output_images[0]

        match, metrics = compare_images(output_image, single_step_image)
        assert match, f"Image mismatch. PSNR={metrics['psnr']:.2f}, SSIM={metrics['ssim']:.4f}"
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
