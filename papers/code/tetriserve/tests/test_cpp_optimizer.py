#!/usr/bin/env python3
"""
Test script to compare C++ and Python implementations of optimal_shape.
"""

import time
import sys
import os
import pytest

from tests.test_constants import SLO_DICT, SHAPE_LIST, BS_DISTRIBUTION
from tetriserve.utils import get_current_time

from tetriserve.scheduler.dyn_step_scheduler import DynStepScheduler
from tetriserve.config.request import UnScheduledRequest
from tetriserve.xfuser.config import InputConfig

try:
    from tetriserve.scheduler.cpp_optimizer import cpp_optimizer

    HAS_CPP_OPTIMIZER = True
except ImportError:
    HAS_CPP_OPTIMIZER = False

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

_SCALING_DATA_EXISTS = os.path.exists("log/benchmark/scaling_efficiency/sd3/e2e_scaling_efficiency.json")
_SKIP_REASON = (
    "C++ optimizer not compiled" if not HAS_CPP_OPTIMIZER
    else "scaling efficiency data not available" if not _SCALING_DATA_EXISTS
    else ""
)


@pytest.mark.skipif(not HAS_CPP_OPTIMIZER or not _SCALING_DATA_EXISTS, reason=_SKIP_REASON or "n/a")
def test_scheduler_integration():
    """Test integration with scheduler"""
    print("\n" + "=" * 60)
    print("Testing scheduler integration...")

    scheduler = DynStepScheduler(
        ranks=[0, 1, 2, 3], schedule_logic="fcfs_optimal_shape_schedule", use_cpp_optimizer=True
    )

    bs_shape_slo_list = []
    for bs, _ in BS_DISTRIBUTION:
        for shape in SHAPE_LIST:
            height, width = shape
            bs_shape_slo_list.append((bs, height, width, SLO_DICT[(bs, height, width)]))

    # Statistics for timing comparison
    cpp_times = []
    python_times = []
    num_iterations = 10

    print(f"\nRunning {num_iterations} iterations per test case for performance comparison...")
    print("-" * 80)

    for bs, height, width, slo in bs_shape_slo_list:

        # Create test request
        input_config = InputConfig(
            prompt="test prompt", height=height, width=width, num_inference_steps=50, batch_size=bs
        )

        # Test C++ implementation
        scheduler.use_cpp_optimizer = True
        start_time = time.perf_counter()
        for i in range(num_iterations):
            test_request = UnScheduledRequest(
                req_id=1,
                sub_req_ids=[1],
                latent_length=None,
                input_config=input_config,
                arrival_time=get_current_time(),
                latency_threshold=slo,
            )
            result_cpp = scheduler.get_optimal_shape(4, test_request)
        cpp_end_time = time.perf_counter()
        cpp_avg_time = (cpp_end_time - start_time) / num_iterations
        cpp_times.append(cpp_avg_time)

        # Test Python implementation
        scheduler.use_cpp_optimizer = False
        start_time = time.perf_counter()
        for i in range(num_iterations):
            test_request = UnScheduledRequest(
                req_id=1,
                sub_req_ids=[1],
                latent_length=None,
                input_config=input_config,
                arrival_time=get_current_time(),
                latency_threshold=slo,
            )
            result_python = scheduler.get_optimal_shape(4, test_request)
        python_end_time = time.perf_counter()
        python_avg_time = (python_end_time - start_time) / num_iterations
        python_times.append(python_avg_time)

        # Calculate speedup
        speedup = python_avg_time / cpp_avg_time if cpp_avg_time > 0 else float("inf")

        # Display results for this test case
        print(f"Test case: bs={bs}, size={height}x{width}, slo={slo:.3f}s")
        print(f"  C++ result:    {result_cpp[1]}")
        print(f"  Python result: {result_python[1]}")
        print(f"  C++ avg time:    {cpp_avg_time*1000:.4f} ms")
        print(f"  Python avg time: {python_avg_time*1000:.4f} ms")
        print(f"  Speedup:         {speedup:.2f}x")
        print(f"  Results match:   {result_cpp[1] == result_python[1]}")
        assert result_cpp[1] == result_python[1]
        print("-" * 80)

    # Overall statistics
    overall_cpp_avg = sum(cpp_times) / len(cpp_times)
    overall_python_avg = sum(python_times) / len(python_times)
    overall_speedup = overall_python_avg / overall_cpp_avg if overall_cpp_avg > 0 else float("inf")

    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"Total test cases:           {len(cpp_times)}")
    print(f"Iterations per test case:   {num_iterations}")
    print(f"Overall C++ avg time:       {overall_cpp_avg*1000:.4f} ms")
    print(f"Overall Python avg time:    {overall_python_avg*1000:.4f} ms")
    print(f"Overall speedup:            {overall_speedup:.2f}x")
    print(f"Total time saved per call:  {(overall_python_avg - overall_cpp_avg)*1000:.4f} ms")
    print("=" * 60)


@pytest.mark.skipif(not HAS_CPP_OPTIMIZER or not _SCALING_DATA_EXISTS, reason=_SKIP_REASON or "n/a")
def test_optimizer_debug():
    scheduler = DynStepScheduler(
        ranks=[0, 1, 2, 3], schedule_logic="fcfs_optimal_shape_schedule", use_cpp_optimizer=True
    )

    current_world_size = 4
    num_inference_steps = 50
    latency_threshold = 149.99999833106995
    time_1gpu = 5.831443125
    time_2gpu = 3.4689693749999995
    time_4gpu = 2.43340890625
    time_8gpu = -1.0

    total_gpu_hours, optimal_shape_list, total_time = cpp_optimizer.get_optimal_shape_for_scheduler(
        current_world_size, num_inference_steps, latency_threshold, time_1gpu, time_2gpu, time_4gpu, time_8gpu
    )

    python_total_gpu_hours, python_optimal_shape_list, python_total_time = scheduler.get_optimal_shape_python(
        current_world_size, num_inference_steps, latency_threshold, time_1gpu, time_2gpu, time_4gpu, time_8gpu
    )

    print(f"C++ total gpu hours: {total_gpu_hours}")
    print(f"C++ optimal shape list: {optimal_shape_list}")
    print(f"C++ total time: {total_time}")
    print(f"Python total gpu hours: {python_total_gpu_hours}")
    print(f"Python optimal shape list: {python_optimal_shape_list}")
    print(f"Python total time: {python_total_time}")


def main():
    test_scheduler_integration()
    # test_optimizer_debug()


if __name__ == "__main__":
    main()
