"""
Test C.7: Port fixes to FLUX pipeline and verify.

1. Generate FLUX diffusers baseline (single GPU)
2. Run TetriServe FLUX SP=1, compare vs baseline
3. Run TetriServe FLUX SP=2, compare vs SP=1

Uses FLUX.1-schnell (open model, 4-step default).

Requires: CUDA_VISIBLE_DEVICES with at least 1 GPU (SP=1) or 2 GPUs (SP=2)
"""

import os
import socket
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import multiprocessing as py_mp
import PIL.Image
import numpy as np

FLUX_MODEL = "black-forest-labs/FLUX.1-schnell"
PROMPT = "A beautiful sunset over mountains"
SEED = 42
HEIGHT = 512
WIDTH = 512
STEPS = 4  # schnell is a 4-step model

FIXTURES_DIR = Path(__file__).parent / "fixtures"
BASELINE_PATH = FIXTURES_DIR / "diffusers_baseline" / "flux_schnell_512x512_seed42.png"
OUTPUT_DIR = FIXTURES_DIR / "tetriserve_output"


def _free_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


# ─── Baseline generation ─────────────────────────────────────────────

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_generate_flux_baseline():
    """Generate FLUX diffusers baseline image."""
    if BASELINE_PATH.exists():
        pytest.skip("Baseline already exists")

    from diffusers import FluxPipeline

    pipe = FluxPipeline.from_pretrained(FLUX_MODEL, torch_dtype=torch.bfloat16)
    pipe = pipe.to("cuda")

    image = pipe(
        PROMPT,
        num_inference_steps=STEPS,
        generator=torch.Generator("cuda").manual_seed(SEED),
        height=HEIGHT,
        width=WIDTH,
        guidance_scale=0.0,  # schnell uses guidance_scale=0
    ).images[0]

    BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    image.save(BASELINE_PATH)

    # Verify reproducibility
    image2 = pipe(
        PROMPT,
        num_inference_steps=STEPS,
        generator=torch.Generator("cuda").manual_seed(SEED),
        height=HEIGHT,
        width=WIDTH,
        guidance_scale=0.0,
    ).images[0]

    diff = np.abs(np.array(image).astype(float) - np.array(image2).astype(float)).max()
    assert diff == 0, f"FLUX baseline not reproducible: max diff={diff}"
    print(f"FLUX baseline saved: {BASELINE_PATH} ({image.size})")

    del pipe
    torch.cuda.empty_cache()


# ─── SP=1 test ────────────────────────────────────────────────────────

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not BASELINE_PATH.exists(), reason="Generate baseline first")
def test_flux_sp1_vs_baseline():
    """Run TetriServe FLUX on 1 GPU and compare against diffusers baseline."""
    from tetriserve.distributed import init_distributed_environment
    from tetriserve.engine.runner import FluidServeRunner
    from tetriserve.config.request import ScheduledRequests
    from tetriserve.xfuser.config.config import (
        EngineConfig, ModelConfig, RuntimeConfig, ParallelConfig,
        SequenceParallelConfig, InputConfig, DataParallelConfig,
        PipeFusionParallelConfig, TensorParallelConfig, FastAttnConfig,
    )
    from tests.utils.image_compare import compare_images

    rank, world_size = 0, 1
    master_port = _free_port()

    os.environ.update(RANK="0", WORLD_SIZE="1", LOCAL_RANK="0",
                      MASTER_ADDR="localhost", MASTER_PORT=str(master_port))
    init_distributed_environment(rank, rank, world_size,
                                f"tcp://localhost:{master_port}", "nccl")
    torch.cuda.set_device(rank)

    ec = EngineConfig(
        model_config=ModelConfig(model=FLUX_MODEL),
        runtime_config=RuntimeConfig(dtype=torch.bfloat16, use_torch_compile=False, use_parallel_vae=False),
        parallel_config=ParallelConfig(
            sp_config=SequenceParallelConfig(ulysses_degree=1, ring_degree=1),
            dp_config=DataParallelConfig(dp_degree=1),
            pp_config=PipeFusionParallelConfig(pp_degree=1),
            tp_config=TensorParallelConfig(tp_degree=1),
            dit_parallel_size=1, vae_parallel_size=0),
        fast_attn_config=FastAttnConfig(use_fast_attn=False))

    runner = FluidServeRunner(FLUX_MODEL, ec, request_queues=[mp.Queue()])

    try:
        ic = InputConfig(batch_size=1, height=HEIGHT, width=WIDTH, prompt=[PROMPT],
                         num_inference_steps=STEPS, output_type="pil", seed=SEED)
        sched = ScheduledRequests(
            req_ids=[0], sub_req_ids=[-1], parallel_ranks=[0],
            attn_ulysses_degree=1, attn_ring_degree=1, sp_degree=1,
            latent_length=None, input_config=ic, origin_steps=[STEPS],
            start_step=0, end_step=STEPS, is_last_steps=[True],
            arrival_time=0, latency_threshold=None, estimated_time=0)

        images = runner.generate(reqs=sched, output_type="pil")
        ts_image = images[0]
        assert isinstance(ts_image, PIL.Image.Image)

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        ts_image.save(OUTPUT_DIR / "flux_sp1_512x512_seed42.png")

        baseline = PIL.Image.open(BASELINE_PATH)
        # SP=1 vs diffusers: flash-attn vs PyTorch attention causes minor numerical diffs
        match, metrics = compare_images(ts_image, baseline, threshold_psnr=25.0, threshold_ssim=0.90)
        print(f"\nFLUX SP=1 vs diffusers: PSNR={metrics['psnr']:.2f}dB, SSIM={metrics['ssim']:.4f}")
        assert match, f"PSNR={metrics['psnr']:.2f}dB, SSIM={metrics['ssim']:.4f}"
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


# ─── SP=2 test ────────────────────────────────────────────────────────

def _sp2_worker(rank, world_size, master_port, request_queues, sync_queues, result_queue):
    from tetriserve.distributed import init_distributed_environment
    from tetriserve.engine.runner import FluidServeRunner
    from tetriserve.config.request import ScheduledRequests
    from tetriserve.xfuser.config.config import (
        EngineConfig, ModelConfig, RuntimeConfig, ParallelConfig,
        SequenceParallelConfig, InputConfig, DataParallelConfig,
        PipeFusionParallelConfig, TensorParallelConfig, FastAttnConfig,
    )

    os.environ.update(RANK=str(rank), WORLD_SIZE=str(world_size), LOCAL_RANK=str(rank),
                      MASTER_ADDR="localhost", MASTER_PORT=str(master_port))
    init_distributed_environment(rank, rank, world_size,
                                f"tcp://localhost:{master_port}", "nccl")
    torch.cuda.set_device(rank)

    ec = EngineConfig(
        model_config=ModelConfig(model=FLUX_MODEL),
        runtime_config=RuntimeConfig(dtype=torch.bfloat16, use_torch_compile=False, use_parallel_vae=False),
        parallel_config=ParallelConfig(
            sp_config=SequenceParallelConfig(ulysses_degree=world_size, ring_degree=1),
            dp_config=DataParallelConfig(dp_degree=1),
            pp_config=PipeFusionParallelConfig(pp_degree=1),
            tp_config=TensorParallelConfig(tp_degree=1),
            dit_parallel_size=world_size, vae_parallel_size=0),
        fast_attn_config=FastAttnConfig(use_fast_attn=False))

    runner = FluidServeRunner(FLUX_MODEL, ec, request_queues=request_queues, sync_queues=sync_queues)

    try:
        ic = InputConfig(batch_size=1, height=HEIGHT, width=WIDTH, prompt=[PROMPT],
                         num_inference_steps=STEPS, output_type="pil", seed=SEED)
        sched = ScheduledRequests(
            req_ids=[0], sub_req_ids=[-1], parallel_ranks=list(range(world_size)),
            attn_ulysses_degree=world_size, attn_ring_degree=1, sp_degree=world_size,
            latent_length=None, input_config=ic, origin_steps=[STEPS],
            start_step=0, end_step=STEPS, is_last_steps=[True],
            arrival_time=0, latency_threshold=None, estimated_time=0)

        images = runner.generate(reqs=sched, output_type="pil")

        if rank == 0:
            image = images[0]
            if isinstance(image, PIL.Image.Image):
                OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                path = OUTPUT_DIR / "flux_sp2_512x512_seed42.png"
                image.save(path)
                result_queue.put(("success", str(path)))
            else:
                result_queue.put(("error", f"Rank 0 got {type(image)}"))
        else:
            result_queue.put(("ok_secondary", None))
    except Exception as e:
        import traceback
        result_queue.put(("error", f"Rank {rank}: {traceback.format_exc()}"))
    finally:
        runner.latent_manager.stop_sender_thread()
        if dist.is_initialized():
            dist.destroy_process_group()


def test_flux_sp2_vs_sp1():
    """Run TetriServe FLUX on 2 GPUs (SP=2) and compare against SP=1."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if torch.cuda.device_count() < 2:
        pytest.skip("Need 2 GPUs")

    sp1_path = OUTPUT_DIR / "flux_sp1_512x512_seed42.png"
    if not sp1_path.exists():
        pytest.skip("Run SP=1 test first")

    world_size = 2
    master_port = _free_port()
    ctx = py_mp.get_context("spawn")
    rq = [ctx.Queue() for _ in range(world_size)]
    sq = [ctx.Queue() for _ in range(world_size)]
    result_queue = ctx.Queue()

    mp.spawn(_sp2_worker, args=(world_size, master_port, rq, sq, result_queue),
             nprocs=world_size, join=True)

    results = [result_queue.get(timeout=30) for _ in range(world_size)]
    for status, msg in results:
        if status == "error":
            raise RuntimeError(f"Worker failed: {msg}")

    sp2_path = None
    for status, msg in results:
        if status == "success":
            sp2_path = msg
            break
    assert sp2_path, "Rank 0 did not produce an image"

    from tests.utils.image_compare import compare_images
    sp2_image = PIL.Image.open(sp2_path)
    sp1_image = PIL.Image.open(sp1_path)
    # SP=2 all-to-all in bfloat16 adds per-layer noise over 4 steps × 19 layers
    match, metrics = compare_images(sp2_image, sp1_image, threshold_psnr=20.0, threshold_ssim=0.85)

    print(f"\nFLUX SP=2 vs SP=1: PSNR={metrics['psnr']:.2f}dB, SSIM={metrics['ssim']:.4f}")
    assert match, f"PSNR={metrics['psnr']:.2f}dB, SSIM={metrics['ssim']:.4f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
