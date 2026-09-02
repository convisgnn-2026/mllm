#!/usr/bin/env python3
"""
Test script for _dynamic_programming_combo_search method.

This test creates realistic input arguments based on the scheduler's actual usage
and tests the dynamic programming combo search functionality.
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from tetriserve.scheduler.dyn_step_scheduler import DynStepScheduler
from tetriserve.config.request import UnScheduledRequest, InputConfig
from tetriserve.utils import get_current_time
import time


def create_test_unscheduled_requests():
    """Create test unscheduled requests with different configurations."""
    requests = []

    # Request 1: Small image, low batch size
    # input_config1 = InputConfig(
    #     batch_size=2,
    #     height=512,
    #     width=512,
    #     num_inference_steps=20
    # )
    # req1 = UnScheduledRequest(
    #     req_id=1,
    #     sub_req_ids=[1, 2],
    #     input_config=input_config1,
    #     arrival_time=get_current_time() - 5.0,  # Arrived 5 seconds ago
    #     latency_threshold=30.0,  # 30 second SLO
    #     origin_steps=20,
    #     latent_length=64
    # )
    # requests.append(req1)

    # Request 2: Medium image, medium batch size
    # input_config2 = InputConfig(
    #     batch_size=4,
    #     height=1024,
    #     width=1024,
    #     num_inference_steps=20
    # )
    # req2 = UnScheduledRequest(
    #     req_id=2,
    #     sub_req_ids=[3, 4, 5, 6],
    #     input_config=input_config2,
    #     arrival_time=get_current_time() - 3.0,  # Arrived 3 seconds ago
    #     latency_threshold=45.0,  # 45 second SLO
    #     origin_steps=20,
    #     latent_length=64
    # )
    # requests.append(req2)

    # Request 3: Medium-large image, small batch size
    input_config3 = InputConfig(batch_size=1, height=256, width=256, num_inference_steps=16)
    req3 = UnScheduledRequest(
        req_id=3,
        sub_req_ids=[7],
        input_config=input_config3,
        arrival_time=get_current_time(),  # Arrived now
        latency_threshold=1.197215654296875,
        origin_steps=16,
        latent_length=256 * 256 // 16 // 16,
    )
    requests.append(req3)

    return requests


def test_dynamic_programming_combo_search():
    """Test the _dynamic_programming_combo_search method with realistic inputs."""

    print("=" * 80)
    print("Testing _dynamic_programming_combo_search")
    print("=" * 80)

    # Create scheduler with 8 GPUs
    scheduler = DynStepScheduler(
        ranks=[0, 1, 2, 3, 4, 5, 6, 7],
        schedule_logic="dyn_slo_schedule_global",
        e2e_scaling_efficiency_path="log/benchmark/scaling_efficiency/sd3/A40/e2e_scaling_efficiency.json",
        use_cpp_optimizer=False,
    )

    # Create test requests
    unscheduled_reqs = create_test_unscheduled_requests()

    # Create realistic shape_state
    # Each request has remaining steps across different GPU configurations [1GPU, 2GPU, 4GPU, 8GPU]
    shape_state = [
        # [5, 3, 2, 0],  # Request 1: 5 steps on 1GPU, 3 on 2GPU, 2 on 4GPU, 0 on 8GPU
        # [0, 8, 4, 0],   # Request 2: 0 on 1GPU, 8 on 2GPU, 4 on 4GPU, 0 on 8GPU
        [16, 0, 0, 0]  # Request 3: 0 on 1GPU, 0 on 2GPU, 6 on 4GPU, 0 on 8GPU
    ]

    # Create realistic rank_time (current GPU completion times)
    # rank_time = [0.0, 2.5, 1.8, 3.2, 0.0, 1.5, 2.1, 0.0]
    rank_time = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    # Create realistic req_time_list (request completion times)
    # req_time_list = [5.2, 8.7, 12.3]
    req_time_list = [0, 0, 0]

    # Create realistic result_missed_list (already missed requests)
    result_missed_list = []

    # Create realistic gpu_time_state (GPU time for each request)
    gpu_time_state = [0.00020241487630208332, 0.00020241487630208332, 0.00020241487630208332]

    # Set round duration
    round_duration = 0.2802317895507812

    print(f"Input Arguments:")
    print(f"  shape_state: {shape_state}")
    print(f"  unscheduled_reqs count: {len(unscheduled_reqs)}")
    print(f"  rank_time: {rank_time}")
    print(f"  req_time_list: {req_time_list}")
    print(f"  result_missed_list: {result_missed_list}")
    print(f"  gpu_time_state: {gpu_time_state}")
    print(f"  round_duration: {round_duration}")
    print()

    # Print request details
    print("Request Details:")
    for i, req in enumerate(unscheduled_reqs):
        seq_len = req.input_config.height * req.input_config.width
        print(f"  Request {i}: ID={req.req_id}, seq_len={seq_len}, batch_size={req.input_config.batch_size}")
        print(f"    remaining_steps: {shape_state[i]}")
        print(f"    arrival_time: {req.arrival_time:.2f}")
        print(f"    latency_threshold: {req.latency_threshold:.2f}")
    print()

    # Call the method
    print("Calling _dynamic_programming_combo_search...")
    start_time = time.time()

    result = scheduler._dynamic_programming_combo_search(
        shape_state=shape_state,
        unscheduled_reqs=unscheduled_reqs,
        rank_time=rank_time,
        req_time_list=req_time_list,
        result_missed_list=result_missed_list,
        gpu_time_state=gpu_time_state,
        round_duration=round_duration,
    )

    # result = scheduler._exhaustive_combo_search(
    #     shape_state=shape_state,
    #     unscheduled_reqs=unscheduled_reqs,
    #     rank_time=rank_time,
    #     req_time_list=req_time_list,
    #     result_missed_list=result_missed_list,
    #     gpu_time_state=gpu_time_state,
    # )

    end_time = time.time()
    execution_time = end_time - start_time

    # Unpack results
    best_missed, best_missed_list, best_filled, best_gpu_time, best_time, best_rank_used, best_combo_list = result

    print(f"Execution completed in {execution_time:.4f} seconds")
    print()

    # Print results
    print("Results:")
    print(f"  best_missed: {best_missed}")
    print(f"  best_missed_list: {best_missed_list}")
    print(f"  best_filled: {best_filled}")
    print(f"  best_gpu_time: {best_gpu_time}")
    print(f"  best_time: {best_time}")
    print(f"  best_rank_used: {best_rank_used}")
    print(f"  best_combo_list: {best_combo_list}")
    print()

    # Print detailed combo analysis
    print("Detailed Combo Analysis:")
    for i, combo in enumerate(best_combo_list):
        req_idx, rank_group, req_step, req_time = combo
        req = unscheduled_reqs[req_idx]
        seq_len = req.input_config.height * req.input_config.width

        print(f"  Combo {i}:")
        print(f"    Request {req_idx} (ID: {req.req_id})")
        print(f"    Rank group: {rank_group}")
        print(f"    Steps to complete: {req_step}")
        print(f"    Estimated time: {req_time:.3f}s")
        print(f"    GPU count: {len(rank_group)}")
        print(f"    Sequence length: {seq_len}")
        print(f"    Batch size: {req.input_config.batch_size}")
        print()

    # Print summary statistics
    print("Summary Statistics:")
    print(f"  Total requests processed: {len(unscheduled_reqs)}")
    print(f"  Requests scheduled: {best_filled}")
    print(f"  Requests missed: {best_missed}")
    print(
        f"  GPU utilization: {best_rank_used}/{len(scheduler.ranks)} ({best_rank_used/len(scheduler.ranks)*100:.1f}%)"
    )
    print(f"  Round duration used: {round_duration}s")
    print(f"  Execution time: {execution_time:.4f}s")

    return result


if __name__ == "__main__":
    test_dynamic_programming_combo_search()
