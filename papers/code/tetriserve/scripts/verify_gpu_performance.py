#!/usr/bin/env python3
"""
Verify GPU performance differences using torch.distributed and tensor operations.
This script measures TFLOPS for matrix multiplication on each GPU.
"""

import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def benchmark_gpu(rank, world_size):
    """Benchmark performance on a single GPU"""

    # Initialize distributed environment
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    # Set device
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    print(f"GPU {rank}: Initialized")

    # Test different matrix sizes (similar to different Lat-Len in your workload)
    # Matrix multiplication: C = A @ B where A is MxK and B is KxN
    # FLOPs = 2 * M * N * K (multiply-add)
    test_configs = [
        # (M, K, N, name)
        (1024, 1024, 1024, "small"),  # ~2.1 GFLOPs per operation
        (2048, 2048, 2048, "medium"),  # ~17.2 GFLOPs per operation
        (4096, 4096, 4096, "large"),  # ~137.4 GFLOPs per operation
        (8192, 8192, 8192, "xlarge"),  # ~1099.5 GFLOPs per operation
    ]

    results = []

    for M, K, N, name in test_configs:
        # Create random matrices
        A = torch.randn(M, K, device=device, dtype=torch.float32)
        B = torch.randn(K, N, device=device, dtype=torch.float32)

        # Warmup
        print(f"GPU {rank}: Warming up {name} workload...")
        for _ in range(10):
            C = torch.matmul(A, B)
        torch.cuda.synchronize()

        # Benchmark
        num_iterations = 100 if name in ["small", "medium"] else 20

        start_time = time.time()
        for _ in range(num_iterations):
            C = torch.matmul(A, B)
        torch.cuda.synchronize()
        end_time = time.time()

        # Calculate performance
        elapsed_time = end_time - start_time
        flops_per_iter = 2.0 * M * N * K
        total_flops = flops_per_iter * num_iterations
        tflops = (total_flops / elapsed_time) / 1e12

        results.append({"size": name, "dims": f"{M}x{K}x{N}", "tflops": tflops, "time": elapsed_time})

        print(f"GPU {rank}: {name} ({M}x{K}x{N}) = {tflops:.2f} TFLOPS")

    # Gather results from all GPUs
    dist.barrier()

    # Communication benchmark
    print(f"\nGPU {rank}: Starting communication benchmark...")

    # Test different message sizes
    comm_sizes = [
        (1024 * 1024, "1MB"),  # 1 MB
        (16 * 1024 * 1024, "16MB"),  # 16 MB
        (64 * 1024 * 1024, "64MB"),  # 64 MB
        (256 * 1024 * 1024, "256MB"),  # 256 MB
    ]

    comm_results = []

    for size_bytes, size_name in comm_sizes:
        # Create data tensors
        send_tensor = torch.randn(size_bytes // 4, device=device, dtype=torch.float32)  # float32 = 4 bytes
        recv_tensors = [torch.zeros_like(send_tensor) for _ in range(world_size)]

        # Warmup
        for _ in range(5):
            dist.all_to_all_single(recv_tensors[rank], send_tensor)
        torch.cuda.synchronize()

        # Benchmark all-to-all
        num_iterations = 20
        start_time = time.time()

        for _ in range(num_iterations):
            output_tensor = torch.zeros(size_bytes // 4 * world_size, device=device, dtype=torch.float32)
            input_tensor = send_tensor.repeat(world_size)
            dist.all_to_all_single(output_tensor, input_tensor)
            torch.cuda.synchronize()

        end_time = time.time()
        elapsed_time = end_time - start_time

        # Calculate bandwidth (bidirectional)
        total_bytes = size_bytes * world_size * 2 * num_iterations  # Each GPU sends and receives
        bandwidth_gb_s = (total_bytes / elapsed_time) / (1024**3)

        comm_results.append({"size": size_name, "bandwidth": bandwidth_gb_s, "time": elapsed_time})

        print(f"GPU {rank}: All-to-all {size_name} = {bandwidth_gb_s:.2f} GB/s")

    dist.barrier()

    # Only rank 0 prints summary
    if rank == 0:
        print("\n" + "=" * 60)
        print("SUMMARY - Average TFLOPS per workload size:")
        print("=" * 60)

    # Collect results from all ranks
    for config_idx, (_, _, _, name) in enumerate(test_configs):
        tflops_list = [0.0] * world_size

        # Each rank broadcasts its result
        for r in range(world_size):
            if r == rank:
                tflops_tensor = torch.tensor([results[config_idx]["tflops"]], device=device)
            else:
                tflops_tensor = torch.zeros(1, device=device)

            dist.broadcast(tflops_tensor, src=r)
            tflops_list[r] = tflops_tensor.item()

        if rank == 0:
            # Analyze results
            lower_gpus = tflops_list[:4] if world_size >= 4 else tflops_list
            upper_gpus = tflops_list[4:] if world_size > 4 else []

            avg_lower = sum(lower_gpus) / len(lower_gpus) if lower_gpus else 0
            avg_upper = sum(upper_gpus) / len(upper_gpus) if upper_gpus else 0

            print(f"\n{name.upper()} workload:")
            print(f"  Individual GPUs: {[f'{t:.2f}' for t in tflops_list]}")

            if world_size >= 8:
                print(f"  Lower GPUs (0-3) avg: {avg_lower:.2f} TFLOPS")
                print(f"  Upper GPUs (4-7) avg: {avg_upper:.2f} TFLOPS")
                if avg_lower > 0:
                    ratio = (avg_upper / avg_lower) * 100
                    print(f"  Upper/Lower ratio: {ratio:.1f}%")

    # Communication results summary
    if rank == 0:
        print("\n" + "=" * 60)
        print("SUMMARY - All-to-All Communication Bandwidth:")
        print("=" * 60)

    # Collect communication results
    for comm_idx, (_, size_name) in enumerate(comm_sizes):
        bandwidth_list = [0.0] * world_size

        # Each rank broadcasts its result
        for r in range(world_size):
            if r == rank:
                bw_tensor = torch.tensor([comm_results[comm_idx]["bandwidth"]], device=device)
            else:
                bw_tensor = torch.zeros(1, device=device)

            dist.broadcast(bw_tensor, src=r)
            bandwidth_list[r] = bw_tensor.item()

        if rank == 0:
            print(f"\n{size_name} transfers:")
            print(f"  Individual GPUs: {[f'{b:.2f}' for b in bandwidth_list]} GB/s")

            if world_size >= 8:
                # Create bandwidth matrix to show GPU-to-GPU patterns
                print(f"\n  Bandwidth matrix hint (based on all-to-all pattern):")
                lower_to_lower = sum(bandwidth_list[:4]) / 4
                upper_to_upper = sum(bandwidth_list[4:]) / 4
                print(f"  Lower GPUs (0-3) avg: {lower_to_lower:.2f} GB/s")
                print(f"  Upper GPUs (4-7) avg: {upper_to_upper:.2f} GB/s")

                # Check if there's a significant difference
                if abs(lower_to_lower - upper_to_upper) > 5:  # More than 5 GB/s difference
                    print(f"  WARNING: Significant bandwidth difference detected!")

    # Cleanup
    dist.destroy_process_group()


def benchmark_gpu_combinations(world_size):
    """Test all C(n,4) combinations of GPUs"""
    import itertools

    if world_size < 4:
        print(f"Skipping C(n,4) test - need at least 4 GPUs (found {world_size})")
        return

    print(f"\n{'='*60}")
    print(f"Testing all C({world_size},4) = {len(list(itertools.combinations(range(world_size), 4)))} combinations")
    print(f"{'='*60}")

    # Test matrix size for quick benchmark
    M, K, N = 4096, 4096, 4096

    results = {}

    for combo in itertools.combinations(range(world_size), 4):
        combo_name = f"GPUs {combo}"
        print(f"\nTesting {combo_name}...")

        # Initialize process group for this subset
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12356"

        # Create a subcommunicator with just these 4 GPUs
        # For simplicity, we'll test each GPU independently
        compute_tflops = []
        comm_bandwidth = []

        for gpu_id in combo:
            torch.cuda.set_device(gpu_id)
            device = torch.device(f"cuda:{gpu_id}")

            # Compute test
            A = torch.randn(M, K, device=device, dtype=torch.float32)
            B = torch.randn(K, N, device=device, dtype=torch.float32)

            # Warmup
            for _ in range(5):
                C = torch.matmul(A, B)
            torch.cuda.synchronize()

            # Benchmark
            num_iterations = 10
            start_time = time.time()
            for _ in range(num_iterations):
                C = torch.matmul(A, B)
            torch.cuda.synchronize()
            end_time = time.time()

            elapsed_time = end_time - start_time
            flops_per_iter = 2.0 * M * N * K
            total_flops = flops_per_iter * num_iterations
            tflops = (total_flops / elapsed_time) / 1e12
            compute_tflops.append(tflops)

            # Simple memory bandwidth test (copy operation)
            size_bytes = 64 * 1024 * 1024  # 64MB
            data = torch.randn(size_bytes // 4, device=device, dtype=torch.float32)

            # Warmup
            for _ in range(5):
                data_copy = data.clone()
            torch.cuda.synchronize()

            # Benchmark
            start_time = time.time()
            for _ in range(20):
                data_copy = data.clone()
            torch.cuda.synchronize()
            end_time = time.time()

            bandwidth = (size_bytes * 20 / (end_time - start_time)) / (1024**3)
            comm_bandwidth.append(bandwidth)

        avg_tflops = sum(compute_tflops) / len(compute_tflops)
        avg_bandwidth = sum(comm_bandwidth) / len(comm_bandwidth)

        results[combo_name] = {"tflops": avg_tflops, "bandwidth": avg_bandwidth, "gpus": combo}

        print(f"  Avg TFLOPS: {avg_tflops:.2f}")
        print(f"  Avg Bandwidth: {avg_bandwidth:.2f} GB/s")

    # Analyze results
    print(f"\n{'='*60}")
    print("ANALYSIS - C(n,4) Performance Patterns:")
    print(f"{'='*60}")

    # Sort by TFLOPS
    sorted_by_tflops = sorted(results.items(), key=lambda x: x[1]["tflops"], reverse=True)

    print("\nTop 5 best performing combinations (by TFLOPS):")
    for i, (name, data) in enumerate(sorted_by_tflops[:5]):
        print(f"  {i+1}. {name}: {data['tflops']:.2f} TFLOPS")

    print("\nBottom 5 worst performing combinations (by TFLOPS):")
    for i, (name, data) in enumerate(sorted_by_tflops[-5:]):
        print(f"  {i+1}. {name}: {data['tflops']:.2f} TFLOPS")

    # Check for patterns
    lower_only = [r for name, r in results.items() if all(g < 4 for g in r["gpus"])]
    upper_only = [r for name, r in results.items() if all(g >= 4 for g in r["gpus"])]
    mixed = [r for name, r in results.items() if any(g < 4 for g in r["gpus"]) and any(g >= 4 for g in r["gpus"])]

    if lower_only:
        avg_lower = sum(r["tflops"] for r in lower_only) / len(lower_only)
        print(f"\nLower GPUs only (0-3) combinations: {avg_lower:.2f} TFLOPS avg")

    if upper_only:
        avg_upper = sum(r["tflops"] for r in upper_only) / len(upper_only)
        print(f"Upper GPUs only (4-7) combinations: {avg_upper:.2f} TFLOPS avg")

    if mixed:
        avg_mixed = sum(r["tflops"] for r in mixed) / len(mixed)
        print(f"Mixed GPU combinations: {avg_mixed:.2f} TFLOPS avg")


def main():
    # Detect available GPUs
    if not torch.cuda.is_available():
        print("Error: CUDA is not available")
        return

    world_size = torch.cuda.device_count()
    print(f"Found {world_size} GPUs")

    if world_size < 2:
        print("Error: Need at least 2 GPUs to run distributed benchmark")
        return

    # Run full benchmark on all GPUs
    mp.spawn(benchmark_gpu, args=(world_size,), nprocs=world_size, join=True)

    # Run C(n,4) combination tests
    benchmark_gpu_combinations(world_size)


if __name__ == "__main__":
    main()
