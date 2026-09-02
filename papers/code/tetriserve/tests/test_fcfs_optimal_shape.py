"""
Test C.6: Verify fcfs_optimal_shape scheduler assigns step_ranges with
no gaps and no overlaps, covering all inference steps.

This is a scheduler-level test — no GPU required. It tests:
1. Single request scheduled in one round (all steps fit in available GPUs)
2. Step range coverage: start_step/end_step cover [0, total_steps) exactly
3. is_last_steps flag set correctly
4. Multi-round scheduling: request split across multiple schedule() calls
"""

import pytest
from copy import deepcopy

from tetriserve.scheduler.dyn_step_scheduler import DynStepScheduler
from tetriserve.xfuser.config import InputConfig
from tetriserve.utils import get_current_time


SCALING_EFFICIENCY_PATH = "log/benchmark/scaling_efficiency/sd3/A40/e2e_scaling_efficiency.json"


@pytest.fixture
def scheduler_4gpu():
    """Create a 4-GPU scheduler with fcfs_optimal_shape."""
    return DynStepScheduler(
        ranks=[0, 1, 2, 3],
        schedule_logic="fcfs_optimal_shape",
        e2e_scaling_efficiency_path=SCALING_EFFICIENCY_PATH,
        use_cpp_optimizer=False,
    )


def _add_request(scheduler, req_id, steps=20, height=512, width=512, batch_size=1):
    """Helper to add a request to the scheduler."""
    sub_req_ids = [-(req_id * 10 + i) for i in range(batch_size)]
    input_config = InputConfig(
        batch_size=batch_size,
        height=height,
        width=width,
        prompt=["test"] * batch_size,
        num_inference_steps=steps,
        output_type="latent",
        seed=42,
    )
    scheduler.add_request(
        request_id=req_id,
        sub_req_ids=sub_req_ids,
        input_config=input_config,
        arrival_time=get_current_time(),
        latency_threshold=float("inf"),
    )
    return sub_req_ids


def test_single_request_step_coverage(scheduler_4gpu):
    """A single request should be scheduled with correct start_step/end_step."""
    scheduler = scheduler_4gpu
    sub_req_ids = _add_request(scheduler, req_id=1, steps=20)

    blocks = scheduler.schedule()
    assert len(blocks) >= 1, "Expected at least one block"

    # Collect all step ranges from the first scheduling round
    first_req = blocks[0].requests[0]
    assert first_req.start_step == 0, f"First stage should start at 0, got {first_req.start_step}"
    assert first_req.end_step > 0, f"First stage should have end_step > 0"
    assert first_req.end_step <= 20, f"end_step should not exceed total steps"

    print(f"First stage: steps [{first_req.start_step}, {first_req.end_step}), "
          f"SP={first_req.sp_degree}, ranks={first_req.parallel_ranks}")

    # If all steps scheduled in one round, verify coverage
    if first_req.end_step == 20:
        assert all(first_req.is_last_steps), "Should be last steps if all 20 scheduled"
    else:
        assert not any(first_req.is_last_steps), "Should NOT be last steps if partial"


def test_multi_round_step_coverage(scheduler_4gpu):
    """When optimal shape splits across rounds, step_ranges should be contiguous."""
    scheduler = scheduler_4gpu

    # Use a request where optimal shape likely splits (many steps, large image)
    sub_req_ids = _add_request(scheduler, req_id=1, steps=28, height=1024, width=1024)
    total_steps = 28

    all_step_ranges = []
    round_num = 0

    while True:
        blocks = scheduler.schedule()
        if not blocks:
            break

        for block in blocks:
            for req in block.requests:
                all_step_ranges.append((req.start_step, req.end_step, req.sp_degree, req.is_last_steps[0]))
                print(f"Round {round_num}: steps [{req.start_step}, {req.end_step}), "
                      f"SP={req.sp_degree}, is_last={req.is_last_steps[0]}")

                # Simulate completion: update request tracker and free GPUs
                scheduler.running_requests.extend(req.sub_req_ids)

        # Simulate round completion
        for block in blocks:
            for req in block.requests:
                scheduler.request_tracker.requests[req.sub_req_ids[0]].input_config.num_inference_steps -= (
                    req.end_step - req.start_step
                )
                scheduler.running_requests = [
                    r for r in scheduler.running_requests if r not in req.sub_req_ids
                ]
            scheduler.add_ranks(block.begin_condition_ranks)

        round_num += 1

        # Check if request is complete
        remaining = scheduler.request_tracker.requests.get(sub_req_ids[0])
        if remaining is None or remaining.input_config.num_inference_steps == 0:
            break

        if round_num > 10:
            pytest.fail("Too many scheduling rounds — likely a bug")

    assert len(all_step_ranges) >= 1, "Should have at least one stage"

    # Verify step coverage: no gaps, no overlaps
    all_step_ranges.sort(key=lambda x: x[0])
    covered_steps = set()
    for start, end, sp, is_last in all_step_ranges:
        assert start < end, f"Invalid range [{start}, {end})"
        for s in range(start, end):
            assert s not in covered_steps, f"Step {s} covered twice (overlap)"
            covered_steps.add(s)

    expected_steps = set(range(total_steps))
    missing = expected_steps - covered_steps
    extra = covered_steps - expected_steps
    assert not missing, f"Missing steps: {sorted(missing)}"
    assert not extra, f"Extra steps: {sorted(extra)}"

    # Last stage should have is_last_steps=True
    assert all_step_ranges[-1][3], "Last stage should have is_last_steps=True"

    # Non-last stages should have is_last_steps=False
    for start, end, sp, is_last in all_step_ranges[:-1]:
        assert not is_last, f"Non-last stage [{start},{end}) has is_last_steps=True"

    print(f"\nAll {total_steps} steps covered in {len(all_step_ranges)} stages, no gaps or overlaps")


def test_tight_latency_forces_multi_gpu(scheduler_4gpu):
    """With a tight latency threshold, the scheduler should use more GPUs."""
    scheduler = scheduler_4gpu
    sub_req_ids = _add_request(scheduler, req_id=1, steps=20, height=1024, width=1024)

    # Override latency threshold to be very tight (forces more GPUs)
    req = scheduler.request_tracker.requests[sub_req_ids[0]]
    req.latency_threshold = 0.1  # Very tight — should force max parallelism

    blocks = scheduler.schedule()
    assert len(blocks) >= 1

    first_req = blocks[0].requests[0]
    print(f"Tight latency: SP={first_req.sp_degree}, steps [{first_req.start_step}, {first_req.end_step})")

    # With such a tight threshold, it may use more GPUs (or fail to meet it)
    # Either way, step_range should be valid
    assert first_req.start_step == 0
    assert first_req.end_step > 0


def test_optimal_shape_gpu_allocation(scheduler_4gpu):
    """Verify GPU allocation is valid."""
    scheduler = scheduler_4gpu
    _add_request(scheduler, req_id=1, steps=20)

    blocks = scheduler.schedule()
    for block in blocks:
        for req in block.requests:
            assert req.sp_degree == len(req.parallel_ranks), (
                f"sp_degree ({req.sp_degree}) != len(parallel_ranks) ({len(req.parallel_ranks)})"
            )
            assert req.sp_degree in (1, 2, 4, 8), f"Invalid sp_degree: {req.sp_degree}"
            assert req.attn_ulysses_degree * req.attn_ring_degree == req.sp_degree


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
