import os
import time
import torch
import torch.distributed as dist
import multiprocessing as mp
import argparse
import traceback
import signal
import json
from datetime import datetime
from functools import partial

from tetriserve.utils import get_device, set_device
from tetriserve.light_ray.latent_manager import LatentManager, LatentFuture
from tetriserve.xfuser.logger import init_logger

logger = init_logger(__name__)


def _set_envs_and_config():
    mp.set_start_method("spawn", force=True)


from tetriserve.xfuser.core.distributed.parallel_state import init_distributed_environment


def setup(rank, world_size):
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12358"
    init_distributed_environment(
        local_rank=rank,
        rank=rank,
        world_size=world_size,
        distributed_init_method=f"tcp://localhost:{os.environ['MASTER_PORT']}",
        backend="nccl",
    )

    # # Initialize the process group
    # dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo", rank=rank, world_size=world_size)
    set_device(gpu=True)
    torch.cuda.set_device(rank)
    # logger.info(f"Rank {rank} initialized with device {get_device()}")


def cleanup():
    """Clean up the distributed environment."""
    if dist.is_initialized():
        logger.info("Destroying process group")
        dist.destroy_process_group()


def run_benchmark(rank, world_size, request_queues, num_iterations=10, output_file=None, mp_barrier=None):
    # Setup error handling
    def signal_handler(sig, frame):
        logger.error(f"Rank {rank}: Received signal {sig}, cleaning up and exiting")
        cleanup()
        os._exit(1)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    assert mp_barrier is not None

    try:
        setup(rank, world_size)

        latent_manager = LatentManager(rank, request_queues)

        # Synchronize to ensure all processes have initialized
        mp_barrier.wait()

        def run_single_config(bs, height, width, num_iterations):
            config_results = {
                "batch_size": bs,
                "height": height,
                "width": width,
                "iterations": num_iterations,
                "transfer_times": [],
                "total_time": 0,
                "throughput": 0,
                "total_size_bytes": 0,
            }

            try:
                if rank == 0:
                    # Sender rank
                    logger.info(f"Rank {rank}: Creating tensors...")
                    # Create tensors with the specified sizes
                    tensors = {
                        "latents": torch.randn(
                            (bs, 4, height // 8, width // 8), dtype=torch.float16, device=get_device()
                        ),
                        "prompt_embeds": torch.randn((bs * 2, 333, 4096), dtype=torch.float16, device=get_device()),
                        "pooled_prompt_embeds": torch.randn((bs * 2, 2048), dtype=torch.float16, device=get_device()),
                        "timesteps": torch.rand((50,), dtype=torch.float32, device=get_device()),
                        "sigmas": torch.rand((51,), dtype=torch.float32, device=get_device()),
                    }

                    # Calculate total size in MB
                    total_size_bytes = 0
                    for tensor_name, tensor in tensors.items():
                        tensor_size = tensor.element_size() * tensor.numel()
                        total_size_bytes += tensor_size
                        logger.info(
                            f"Tensor {tensor_name}: shape={tensor.shape}, dtype={tensor.dtype}, "
                            f"size={tensor_size / (1024 * 1024):.2f} MB"
                        )

                    config_results["total_size_bytes"] = total_size_bytes
                    logger.info(f"Total data size: {total_size_bytes / (1024 * 1024):.2f} MB")

                    # Save the latents for each iteration
                    futures = []
                    for i in range(num_iterations):
                        sub_req_id = i
                        logger.info(f"Rank {rank}: Saving tensor {i}...")
                        future = latent_manager.save_latent(sub_req_id, tensors)
                        futures.append(future)

                    logger.info(f"Rank {rank}: All tensors saved, waiting for requests...")
                    mp_barrier.wait()

                elif rank == 1:
                    # Receiver rank
                    mp_barrier.wait()

                    # Benchmark the p2p overhead
                    logger.info(f"Rank {rank}: Starting benchmark...")

                    # Calculate tensors_info to use for verification
                    tensors_info = {
                        "latents": {"shape": (bs, 4, height // 8, width // 8), "dtype": torch.float16},
                        "prompt_embeds": {"shape": (bs * 2, 333, 4096), "dtype": torch.float16},
                        "pooled_prompt_embeds": {"shape": (bs * 2, 2048), "dtype": torch.float16},
                        "timesteps": {"shape": (50,), "dtype": torch.float32},
                        "sigmas": {"shape": (51,), "dtype": torch.float32},
                    }

                    # Create futures directly without going through save_latent
                    futures = [
                        LatentFuture(sub_req_id=i, src_rank=0, tensors_info=tensors_info) for i in range(num_iterations)
                    ]

                    # Measure time for each get() call
                    total_time = 0
                    for i, future in enumerate(futures):
                        logger.info(f"Rank {rank}: Getting tensor {i}...")
                        start_time = time.time()

                        try:
                            received_tensors = future.get(rank, request_queues)
                            end_time = time.time()
                            logger.info(f"Rank {rank}: Got tensor {i}")

                            transfer_time = end_time - start_time
                            total_time += transfer_time
                            config_results["transfer_times"].append(transfer_time)

                            # Verify tensor shapes
                            for tensor_name, info in tensors_info.items():
                                if tensor_name in received_tensors:
                                    actual_shape = received_tensors[tensor_name].shape
                                    expected_shape = info["shape"]
                                    if actual_shape != expected_shape:
                                        logger.warning(
                                            f"Shape mismatch for {tensor_name}: expected {expected_shape}, got {actual_shape}"
                                        )

                            logger.info(f"Iteration {i}: Transfer took {transfer_time:.6f} seconds")
                        except Exception as e:
                            logger.error(f"Error getting tensor {i}: {e}")
                            logger.error(traceback.format_exc())
                            break

                    if total_time > 0 and i > 0:
                        avg_time = total_time / (i + 1)
                        logger.info(f"Average transfer time: {avg_time:.6f} seconds")

                        # Calculate throughput
                        total_size_bytes = sum(
                            torch.empty(info["shape"], dtype=info["dtype"]).element_size()
                            * torch.empty(info["shape"], dtype=info["dtype"]).numel()
                            for info in tensors_info.values()
                        )

                        throughput = (total_size_bytes * (i + 1)) / (total_time * 1024 * 1024)
                        logger.info(f"Total throughput: {throughput:.2f} MB/s")

                        config_results["total_time"] = total_time
                        config_results["throughput"] = throughput
                        config_results["total_size_bytes"] = total_size_bytes

                mp_barrier.wait()
                return config_results

            except Exception as e:
                logger.error(f"Rank {rank}: Exception during benchmark: {e}")
                logger.error(traceback.format_exc())
                return config_results

        warmup_iterations = 100
        run_single_config(1, 1024, 1024, warmup_iterations)

        # Run benchmarks for different configurations
        all_results = []
        # for bs in [1, 2]:
        # for bs in [1]:
        for bs in [4]:
            for height, width in [
                (256, 256),
                # (256, 512),
                (512, 512),
                # (512, 1024),
                (1024, 1024),
                # (1024, 2048),
                (2048, 2048),
            ]:
                results = run_single_config(bs, height, width, num_iterations)
                if rank == 1:  # Only collect results from receiver rank
                    all_results.append(results)

        # Save results to JSON file
        if rank == 1:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if output_file is not None:
                results_file = output_file
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(os.path.abspath(results_file)), exist_ok=True)

                with open(results_file, "w") as f:
                    json.dump({"timestamp": timestamp, "configurations": all_results}, f, indent=2)

                logger.info(f"Results saved to {results_file}")

        # Clean up regardless of success/failure
        logger.info(f"Rank {rank}: Stopping sender thread")
        latent_manager.stop_sender_thread()
        logger.info(f"Rank {rank}: Sender thread stopped")

    except Exception as e:
        logger.error(f"Rank {rank}: Exception in main process: {e}")
        logger.error(traceback.format_exc())
    finally:
        # Ensure cleanup happens
        logger.info(f"Rank {rank}: Running cleanup")
        cleanup()
        logger.info(f"Rank {rank}: Cleanup complete")


"""
python benchmark/latent_transfer/run_latent_manager_benchmark.py --iterations 40 --output_file ./tetriserve_benchmark/latent_transfer/latent_transfer_benchmark.json

python benchmark/latent_transfer/run_latent_manager_benchmark.py --iterations 40 --output_file log/benchmark/latent_transfer/latent_transfer_benchmark.json 2>&1 | tee log/benchmark/latent_transfer/latent_transfer_benchmark.log

CUDA_VISIBLE_DEVICES=0,2 python benchmark/latent_transfer/run_latent_manager_benchmark.py --iterations 20 --output_file log/benchmark/latent_transfer/latent_transfer_benchmark_gpu_0_2.json 2>&1 | tee log/benchmark/latent_transfer/latent_transfer_benchmark_gpu_0_2.log
"""


def main():
    parser = argparse.ArgumentParser(description="Benchmark LatentFuture.get p2p overhead")
    parser.add_argument("--iterations", type=int, default=10, help="Number of iterations")
    parser.add_argument("--output_file", type=str, help="Path to save benchmark results (optional)")
    args = parser.parse_args()

    world_size = 2  # We need at least 2 processes: a sender and a receiver

    _set_envs_and_config()

    request_queues = [mp.Queue() for _ in range(world_size)]
    mp_barrier = mp.Barrier(world_size)  # Create a barrier for world_size processes

    try:
        processes = []
        for i in range(world_size):
            p = mp.Process(
                target=run_benchmark,
                args=(i, world_size, request_queues, args.iterations, args.output_file, mp_barrier),
            )
            p.start()
            processes.append(p)

        # Wait for all processes to complete
        for p in processes:
            p.join(timeout=120)  # Add a 2-minute timeout per process
            if p.is_alive():
                logger.warning(f"Process {p.pid} is still running after timeout, terminating")
                p.terminate()
                p.join(5)  # Give it 5 seconds to terminate
                if p.is_alive():
                    logger.error(f"Process {p.pid} failed to terminate, killing")
                    os.kill(p.pid, signal.SIGKILL)

    except KeyboardInterrupt:
        logger.info("Caught keyboard interrupt, terminating processes")
        for p in processes:
            if p.is_alive():
                p.terminate()

    finally:
        # Ensure queues are properly cleaned up
        logger.info("Cleaning up queues")
        for q in request_queues:
            if hasattr(q, "cancel_join_thread"):
                q.cancel_join_thread()

        logger.info("Benchmark complete")


if __name__ == "__main__":
    main()
