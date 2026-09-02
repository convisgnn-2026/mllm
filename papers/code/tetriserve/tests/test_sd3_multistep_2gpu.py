"""
Test C.5: Multi-step inference on 2 GPUs.

Split 20 steps into 2 rounds (steps 0-9, steps 10-19) on 2 GPUs with SP=2.
Round 1 produces a LatentFuture, round 2 consumes it and produces the final image.
Compare against single-round SP=2 output (from C.3).

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
from tetriserve.light_ray.latent_future import LatentFuture, TensorInfo
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
TOTAL_STEPS = 20
SPLIT = 10  # steps 0-9 in round 1, steps 10-19 in round 2

SP2_IMAGE_PATH = Path(__file__).parent / "fixtures" / "tetriserve_output" / "sd3_sp2_512x512_seed42.png"
OUTPUT_DIR = Path(__file__).parent / "fixtures" / "tetriserve_output"
SUB_REQ_ID = -100  # unique sub_req_id for this test


def _free_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _worker(rank, world_size, master_port, request_queues, sync_queues,
            metadata_queue, result_queue):
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(master_port)

    init_distributed_environment(
        rank=rank, local_rank=rank, world_size=world_size,
        distributed_init_method=f"tcp://localhost:{master_port}", backend="nccl",
    )
    torch.cuda.set_device(rank)

    engine_config = EngineConfig(
        model_config=ModelConfig(model=MODEL_ID),
        runtime_config=RuntimeConfig(
            dtype=torch.bfloat16, use_torch_compile=False, use_parallel_vae=False,
        ),
        parallel_config=ParallelConfig(
            sp_config=SequenceParallelConfig(ulysses_degree=world_size, ring_degree=1),
            dp_config=DataParallelConfig(dp_degree=1),
            pp_config=PipeFusionParallelConfig(pp_degree=1),
            tp_config=TensorParallelConfig(tp_degree=1),
            dit_parallel_size=world_size, vae_parallel_size=0,
        ),
        fast_attn_config=FastAttnConfig(use_fast_attn=False),
    )

    runner = FluidServeRunner(
        pretrained_model_name_or_path=MODEL_ID, engine_config=engine_config,
        request_queues=request_queues, sync_queues=sync_queues,
    )

    try:
        parallel_ranks = list(range(world_size))

        # ---- ROUND 1: steps 0 to SPLIT-1, output=latent ----
        round1_config = InputConfig(
            batch_size=1, height=HEIGHT, width=WIDTH, prompt=[PROMPT],
            num_inference_steps=SPLIT, output_type="latent", seed=SEED,
        )
        sched_r1 = ScheduledRequests(
            req_ids=[0], sub_req_ids=[SUB_REQ_ID], parallel_ranks=parallel_ranks,
            attn_ulysses_degree=world_size, attn_ring_degree=1, sp_degree=world_size,
            latent_length=None, input_config=round1_config,
            origin_steps=[TOTAL_STEPS], start_step=0, end_step=SPLIT,
            is_last_steps=[False], arrival_time=0, latency_threshold=None, estimated_time=0,
        )

        r1_outputs = runner.generate(reqs=sched_r1, output_type="latent")

        # Sync CUDA state between rounds
        torch.cuda.synchronize()

        # Rank 0 (comm_src_rank) gets the LatentFuture; share metadata to all ranks
        if rank == 0:
            latent_future = r1_outputs[0]
            assert isinstance(latent_future, LatentFuture), f"Expected LatentFuture, got {type(latent_future)}"
            # Serialize metadata for other ranks
            meta = {
                "sub_req_id": latent_future.sub_req_id,
                "src_rank": latent_future.src_rank,
                "step_range": latent_future.step_range,
                "tensors_info": {
                    name: {"shape": info.shape, "dtype": info.dtype, "device": info.device}
                    for name, info in latent_future.tensors_info.items()
                },
            }
            # Put metadata for each non-zero rank
            for _ in range(world_size - 1):
                metadata_queue.put(meta)
        else:
            meta = metadata_queue.get(timeout=30.0)
            latent_future = LatentFuture(
                sub_req_id=meta["sub_req_id"], src_rank=meta["src_rank"],
                tensors_info=meta["tensors_info"], step_range=meta["step_range"],
            )

        # ---- ROUND 2: steps SPLIT to TOTAL_STEPS-1, output=pil ----
        round2_config = InputConfig(
            batch_size=1, height=HEIGHT, width=WIDTH, prompt=[PROMPT],
            num_inference_steps=TOTAL_STEPS - SPLIT, output_type="pil", seed=SEED,
        )
        sched_r2 = ScheduledRequests(
            req_ids=[0], sub_req_ids=[SUB_REQ_ID], parallel_ranks=parallel_ranks,
            attn_ulysses_degree=world_size, attn_ring_degree=1, sp_degree=world_size,
            latent_length=None, input_config=round2_config,
            origin_steps=[TOTAL_STEPS], start_step=SPLIT, end_step=TOTAL_STEPS,
            is_last_steps=[True], arrival_time=0, latency_threshold=None, estimated_time=0,
        )

        r2_outputs = runner.generate(
            reqs=sched_r2, latent_future_dict={SUB_REQ_ID: latent_future}, output_type="pil",
        )

        if rank == 0:
            image = r2_outputs[0]
            if isinstance(image, PIL.Image.Image):
                OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                save_path = OUTPUT_DIR / "sd3_multistep_sp2_512x512_seed42.png"
                image.save(save_path)
                result_queue.put(("success", str(save_path)))
            else:
                result_queue.put(("error", f"Rank 0 got {type(image)} instead of PIL.Image"))
        else:
            result_queue.put(("ok_secondary", None))

    except Exception as e:
        import traceback
        result_queue.put(("error", f"Rank {rank}: {traceback.format_exc()}"))

    finally:
        runner.latent_manager.stop_sender_thread()
        if dist.is_initialized():
            dist.destroy_process_group()


def test_sd3_multistep_2gpu():
    """Run multi-step SD3 inference on 2 GPUs and compare against single-round SP=2."""
    if not torch.cuda.is_available():
        import pytest; pytest.skip("CUDA not available")
    if torch.cuda.device_count() < 2:
        import pytest; pytest.skip("Need at least 2 GPUs")
    if not SP2_IMAGE_PATH.exists():
        import pytest; pytest.skip(f"SP=2 baseline not found: {SP2_IMAGE_PATH} (run C.3 first)")

    world_size = 2
    master_port = _free_port()
    ctx = py_mp.get_context("spawn")
    request_queues = [ctx.Queue() for _ in range(world_size)]
    sync_queues = [ctx.Queue() for _ in range(world_size)]
    metadata_queue = ctx.Queue()
    result_queue = ctx.Queue()

    mp.spawn(
        _worker,
        args=(world_size, master_port, request_queues, sync_queues,
              metadata_queue, result_queue),
        nprocs=world_size,
        join=True,
    )

    results = []
    for _ in range(world_size):
        results.append(result_queue.get(timeout=30.0))

    for status, msg in results:
        if status == "error":
            raise RuntimeError(f"Worker failed: {msg}")

    ms_image_path = None
    for status, msg in results:
        if status == "success":
            ms_image_path = msg
            break

    assert ms_image_path is not None, "Rank 0 did not produce an image"

    from tests.utils.image_compare import compare_images

    ms_image = PIL.Image.open(ms_image_path)
    sp2_image = PIL.Image.open(SP2_IMAGE_PATH)

    # Multi-step should match single-round SP=2 closely — same distributed
    # attention, just split across 2 scheduling rounds with latent transfer.
    match, metrics = compare_images(ms_image, sp2_image, threshold_psnr=18.0, threshold_ssim=0.80)

    print(f"\nMulti-step SP=2 vs single-round SP=2 comparison:")
    print(f"  PSNR: {metrics['psnr']:.2f} dB (threshold: 18.0)")
    print(f"  SSIM: {metrics['ssim']:.4f} (threshold: 0.80)")
    print(f"  Match: {match}")

    assert match, (
        f"Multi-step output doesn't match single-round SP=2. "
        f"PSNR={metrics['psnr']:.2f}dB, SSIM={metrics['ssim']:.4f}"
    )


if __name__ == "__main__":
    test_sd3_multistep_2gpu()
