import os
import time
import torch
import multiprocessing as mp
import torch.distributed as dist
import torch.multiprocessing as mp

from tetriserve.light_ray.latent_manager import LatentManager
from tetriserve.light_ray.latent_future import LatentFuture, TensorInfo
from tetriserve.utils import get_device, set_device
from tetriserve.xfuser.logger import init_logger

logger = init_logger(__name__)


def test_worker_fn(rank, world_size, request_queues: list, use_gpu: bool = True):
    """
    Tests the mp.Queue based LatentManager flow.
    Rank 0 saves a latent.
    Rank 1 requests the latent using future.get(), which puts request on Rank 0's queue.
    Rank 0's background thread processes the queue and sends.
    Rank 1 receives.
    """
    # --- Initialization ---
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["MASTER_ADDR"] = "localhost"
    master_port = "29501"
    os.environ["MASTER_PORT"] = master_port

    set_device(gpu=use_gpu)  # Force CPU for testing queue logic reliably
    backend = "gloo" if not use_gpu else "nccl"  # Use gloo for CPU test
    try:
        dist.init_process_group(backend, rank=rank, world_size=world_size)
    except Exception as e:
        logger.error(f"[Rank {rank}] Failed to init process group with {backend}: {e}")
        return

    # Use torch.cuda.set_device if using CUDA
    if get_device().type == "cuda":
        torch.cuda.set_device(rank)
    logger.info(f"[Rank {rank}] Using device: {get_device()} with backend: {backend}")

    # Pass the list of queues to the manager
    latent_manager = LatentManager(rank, request_queues)
    test_sub_req_id = 101

    bs_list = [1, 2, 4]
    shape_list = []
    for bs in bs_list:
        for width, height in zip([512, 1024, 2048], [512, 1024, 2048]):
            shape_list.append((bs, height, width))

    for shape in shape_list:
        bs, height, width = shape

        if rank == 0:
            logger.info(f"[Rank 0] Starting test sender logic.")
            try:
                # 1. Save latent on rank 0
                tensors = {
                    "latents": torch.randn((bs, 4, height // 8, width // 8), dtype=torch.float16, device=get_device()),
                    "prompt_embeds": torch.randn((bs * 2, 333, 4096), dtype=torch.float16, device=get_device()),
                    "pooled_prompt_embeds": torch.randn((bs * 2, 2048), dtype=torch.float16, device=get_device()),
                    "timesteps": torch.rand((50,), dtype=torch.float32, device=get_device()),
                    "sigmas": torch.rand((51,), dtype=torch.float32, device=get_device()),
                }
                latent_manager.save_latent(sub_req_id=test_sub_req_id, tensors=tensors)

                # 2. Wait for Rank 1 to finish (signaled by barrier)
                dist.barrier()
                # Add a small delay to ensure sender thread processes potential final messages
                time.sleep(0.2)

            except Exception as e:
                logger.error(f"[Rank 0] TEST FAILED: Error during sender logic: {e}")
                # Still need to hit barrier if Rank 1 is waiting
                try:
                    dist.barrier()  # noqa
                except:
                    pass  # noqa
            finally:
                # 4. Signal completion by reaching the barrier
                dist.barrier()

        elif rank == 1:
            logger.info(f"[Rank 1] Starting test receiver logic.")
            try:
                dist.barrier()
                tensors_info = {
                    "latents": TensorInfo(shape=(bs, 4, height // 8, width // 8), dtype=torch.float16, device="cpu"),
                    "prompt_embeds": TensorInfo(shape=(bs * 2, 333, 4096), dtype=torch.float16, device="cpu"),
                    "pooled_prompt_embeds": TensorInfo(shape=(bs * 2, 2048), dtype=torch.float16, device="cpu"),
                    "timesteps": TensorInfo(shape=(50,), dtype=torch.float32, device="cpu"),
                    "sigmas": TensorInfo(shape=(51,), dtype=torch.float32, device="cpu"),
                }
                future = LatentFuture(sub_req_id=test_sub_req_id, src_rank=0, tensors_info=tensors_info)

                # 3. Request the latent. This puts request on Rank 0's queue and blocks on dist.recv.
                logger.info(f"[Rank 1] Calling future.get().")
                received_tensors = future.get()
                logger.info(f"[Rank 1] TEST PASSED.")

            except Exception as e:
                logger.error(f"[Rank 1] TEST FAILED: Error during receiver logic: {e}")
                import traceback

                logger.error(traceback.format_exc())
            finally:
                # 4. Signal completion by reaching the barrier
                dist.barrier()

    logger.info(f"[Rank {rank}] Stopping sender thread.")
    latent_manager.stop_sender_thread()
    time.sleep(0.5)

    try:
        dist.destroy_process_group()
    except Exception as e:
        logger.error(f"[Rank {rank}] Error destroying process group: {e}")
    logger.info(f"[Rank {rank}] Test worker finished.")


def _set_envs_and_config():
    mp.set_start_method("spawn", force=True)


def main_test():
    _set_envs_and_config()

    world_size = 2
    request_queues = [mp.Queue() for _ in range(world_size)]

    logger.info(f"Starting LatentManager test with {world_size} processes and multiprocessing.Queue...")

    processes = []
    # Use standard multiprocessing Process as torch_spawn expects fn as first arg
    for i in range(world_size):
        p = mp.Process(target=test_worker_fn, args=(i, world_size, request_queues))
        p.start()
        processes.append(p)

    exit_codes = {}
    for i, p in enumerate(processes):
        p.join()
        exit_codes[i] = p.exitcode

    logger.info("LatentManager test finished.")
    logger.info(f"Exit codes: {exit_codes}")
    # Check exit codes for success (0)
    if any(c != 0 for c in exit_codes.values() if c is not None):
        logger.error("TEST FAILED: One or more processes exited with non-zero status.")
    else:
        logger.info("All processes finished successfully.")


if __name__ == "__main__":
    main_test()
