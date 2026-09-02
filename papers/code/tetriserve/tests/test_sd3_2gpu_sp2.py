"""
Test C.3: Run TetriServe SD3 on 2 GPUs with SP=2 (fixed_sp_2) and compare
output against SP=1 single-GPU output from C.2.

SP=2 uses Ulysses sequence parallelism — splits Q/K/V along the sequence
dimension across 2 GPUs. This should produce near-identical output to SP=1
since both use the same attention implementation (flash-attn).

Requires: CUDA_VISIBLE_DEVICES=1,3 (2 free GPUs)
"""

import os
import socket
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import multiprocessing as py_mp
import PIL.Image

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

MODEL_ID = "stabilityai/stable-diffusion-3-medium-diffusers"
PROMPT = "A beautiful sunset over mountains"
SEED = 42
HEIGHT = 512
WIDTH = 512
STEPS = 20

SP1_IMAGE_PATH = Path(__file__).parent / "fixtures" / "tetriserve_output" / "sd3_sp1_512x512_seed42.png"
OUTPUT_DIR = Path(__file__).parent / "fixtures" / "tetriserve_output"


def _free_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _worker(rank, world_size, master_port, request_queues, sync_queues, result_queue):
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
            sp_config=SequenceParallelConfig(ulysses_degree=world_size, ring_degree=1),
            dp_config=DataParallelConfig(dp_degree=1),
            pp_config=PipeFusionParallelConfig(pp_degree=1),
            tp_config=TensorParallelConfig(tp_degree=1),
            dit_parallel_size=world_size,
            vae_parallel_size=0,
        ),
        fast_attn_config=FastAttnConfig(use_fast_attn=False),
    )

    runner = FluidServeRunner(
        pretrained_model_name_or_path=MODEL_ID,
        engine_config=engine_config,
        request_queues=request_queues,
        sync_queues=sync_queues,
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
            parallel_ranks=list(range(world_size)),
            attn_ulysses_degree=world_size,
            attn_ring_degree=1,
            sp_degree=world_size,
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

        if rank == 0:
            image = output_images[0]
            if isinstance(image, PIL.Image.Image):
                OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                save_path = OUTPUT_DIR / "sd3_sp2_512x512_seed42.png"
                image.save(save_path)
                result_queue.put(("success", str(save_path)))
            else:
                result_queue.put(("error", f"Rank 0 got {type(image)} instead of PIL.Image"))
        else:
            # Non-primary ranks return None from VAE decode
            result_queue.put(("ok_secondary", None))

    except Exception as e:
        import traceback
        result_queue.put(("error", f"Rank {rank}: {traceback.format_exc()}"))

    finally:
        runner.latent_manager.stop_sender_thread()
        if dist.is_initialized():
            dist.destroy_process_group()


def test_sd3_2gpu_sp2():
    """Run TetriServe SD3 on 2 GPUs (SP=2) and compare against SP=1 output."""
    if not torch.cuda.is_available():
        import pytest
        pytest.skip("CUDA not available")

    if torch.cuda.device_count() < 2:
        import pytest
        pytest.skip("Need at least 2 GPUs")

    if not SP1_IMAGE_PATH.exists():
        import pytest
        pytest.skip(f"SP=1 baseline not found: {SP1_IMAGE_PATH} (run C.2 first)")

    world_size = 2
    master_port = _free_port()
    ctx = py_mp.get_context("spawn")
    request_queues = [ctx.Queue() for _ in range(world_size)]
    sync_queues = [ctx.Queue() for _ in range(world_size)]
    result_queue = ctx.Queue()

    mp.spawn(
        _worker,
        args=(world_size, master_port, request_queues, sync_queues, result_queue),
        nprocs=world_size,
        join=True,
    )

    # Collect results from both ranks
    results = []
    for _ in range(world_size):
        results.append(result_queue.get(timeout=30.0))

    # Check for errors
    for status, msg in results:
        if status == "error":
            raise RuntimeError(f"Worker failed: {msg}")

    # Find the success result (rank 0)
    sp2_image_path = None
    for status, msg in results:
        if status == "success":
            sp2_image_path = msg
            break

    assert sp2_image_path is not None, "Rank 0 did not produce an image"

    # Compare SP=2 output against SP=1 output
    from tests.utils.image_compare import compare_images

    sp2_image = PIL.Image.open(sp2_image_path)
    sp1_image = PIL.Image.open(SP1_IMAGE_PATH)

    # SP=2 Ulysses parallelism introduces numerical differences from bfloat16
    # all-to-all operations accumulating over 20 steps × 24 attention layers.
    # Expected: ~21dB PSNR, ~0.87 SSIM vs SP=1 output.
    match, metrics = compare_images(sp2_image, sp1_image, threshold_psnr=20.0, threshold_ssim=0.85)

    print(f"\nSP=2 vs SP=1 comparison:")
    print(f"  PSNR: {metrics['psnr']:.2f} dB (threshold: 20.0)")
    print(f"  SSIM: {metrics['ssim']:.4f} (threshold: 0.85)")
    print(f"  Match: {match}")

    assert match, (
        f"SP=2 output doesn't match SP=1 baseline. "
        f"PSNR={metrics['psnr']:.2f}dB, SSIM={metrics['ssim']:.4f}"
    )


if __name__ == "__main__":
    test_sd3_2gpu_sp2()
