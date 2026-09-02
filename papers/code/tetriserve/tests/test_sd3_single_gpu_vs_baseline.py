"""
Test C.2: Run TetriServe SD3 on single GPU (SP=1) and compare output
against diffusers baseline from C.1.

Verifies the TetriServe pipeline itself produces correct images before
testing multi-GPU configurations.
"""

import os
import socket
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from PIL import Image

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
PROMPT = "A beautiful sunset over mountains"
SEED = 42
HEIGHT = 512
WIDTH = 512
STEPS = 20

BASELINE_PATH = Path(__file__).parent / "fixtures" / "diffusers_baseline" / "sd3_512x512_seed42.png"


def _free_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not BASELINE_PATH.exists(), reason="Baseline image not generated (run C.1 first)")
def test_sd3_single_gpu_vs_diffusers_baseline():
    """Run TetriServe SD3 on 1 GPU and compare against diffusers baseline."""
    rank = 0
    world_size = 1
    master_port = _free_port()
    request_queues = [mp.Queue()]

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
        model_config=ModelConfig(model=MODEL_ID),
        runtime_config=RuntimeConfig(
            dtype=torch.bfloat16,
            use_torch_compile=False,
            use_parallel_vae=False,
        ),
        parallel_config=ParallelConfig(
            sp_config=SequenceParallelConfig(ulysses_degree=1, ring_degree=1),
            dp_config=DataParallelConfig(dp_degree=1),
            pp_config=PipeFusionParallelConfig(pp_degree=1),
            tp_config=TensorParallelConfig(tp_degree=1),
            dit_parallel_size=1,
            vae_parallel_size=0,
        ),
        fast_attn_config=FastAttnConfig(use_fast_attn=False),
    )

    runner = FluidServeRunner(
        pretrained_model_name_or_path=MODEL_ID,
        engine_config=engine_config,
        request_queues=request_queues,
    )

    try:
        input_config = InputConfig(
            batch_size=1,
            height=HEIGHT,
            width=WIDTH,
            prompt=[PROMPT],
            num_inference_steps=STEPS,
            output_type="pil",
            seed=SEED,
        )

        sched = ScheduledRequests(
            req_ids=[0],
            sub_req_ids=[-1],
            parallel_ranks=[0],
            attn_ulysses_degree=1,
            attn_ring_degree=1,
            sp_degree=1,
            latent_length=None,
            input_config=input_config,
            origin_steps=[STEPS],
            start_step=0,
            end_step=STEPS,
            is_last_steps=[True],
            arrival_time=0,
            latency_threshold=None,
            estimated_time=0,
        )

        output_images = runner.generate(reqs=sched, output_type="pil")
        tetriserve_image = output_images[0]

        assert isinstance(tetriserve_image, Image.Image), f"Expected PIL Image, got {type(tetriserve_image)}"

        # Save TetriServe output for visual inspection
        output_dir = Path(__file__).parent / "fixtures" / "tetriserve_output"
        output_dir.mkdir(parents=True, exist_ok=True)
        tetriserve_image.save(output_dir / "sd3_sp1_512x512_seed42.png")

        # Compare against diffusers baseline
        # Note: PSNR ~26dB expected because TetriServe uses flash-attn while
        # diffusers uses F.scaled_dot_product_attention — numerically different
        # but mathematically equivalent. SSIM > 0.90 confirms visual similarity.
        baseline_image = Image.open(BASELINE_PATH)
        match, metrics = compare_images(tetriserve_image, baseline_image, threshold_psnr=25.0, threshold_ssim=0.90)

        print(f"\nComparison results:")
        print(f"  PSNR: {metrics['psnr']:.2f} dB (threshold: 25.0)")
        print(f"  SSIM: {metrics['ssim']:.4f} (threshold: 0.90)")
        print(f"  Match: {match}")

        assert match, (
            f"TetriServe output doesn't match diffusers baseline. "
            f"PSNR={metrics['psnr']:.2f}dB, SSIM={metrics['ssim']:.4f}"
        )

    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
