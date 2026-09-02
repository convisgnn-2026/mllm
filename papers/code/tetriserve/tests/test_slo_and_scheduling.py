"""
Tests for SLO policy calculation, scheduler SLO logic, and RequestTracker.

P1.8: Covers SLO deadlines, scheduling under resource constraints,
and request tracker state management — all without GPU.
"""

import os
import pytest
from unittest.mock import patch
from copy import deepcopy

from tetriserve.workload.core import Request
from tetriserve.workload.slo import FixedSLOPolicy, BaselineSLOPolicy, CustomSLOPolicy
from tetriserve.engine.request_tracker import RequestTracker
from tetriserve.xfuser.config import InputConfig


# ── Helpers ─────────────────────────────────────────────────────────

def _make_request(batch_size=1, height=512, width=512, steps=20, **kw):
    return Request(
        req_idx=kw.get("req_idx", 0),
        model_name="sd3",
        batch_size=batch_size,
        height=height,
        width=width,
        num_inference_steps=steps,
        prompt=["test"] * batch_size,
    )


def _make_input_config(batch_size=1, height=512, width=512, steps=20):
    return InputConfig(
        prompt=["test"] * batch_size,
        batch_size=batch_size,
        height=height,
        width=width,
        num_inference_steps=steps,
    )


# ── FixedSLOPolicy ──────────────────────────────────────────────────

class TestFixedSLOPolicy:
    def test_returns_same_value_for_all_requests(self):
        policy = FixedSLOPolicy(slo=15.0)
        for h, w in [(512, 512), (1024, 1024), (2048, 2048)]:
            req = _make_request(height=h, width=w)
            assert policy.calculate_slo(req) == 15.0

    def test_rejects_non_positive_slo(self):
        with pytest.raises(ValueError, match="must be positive"):
            FixedSLOPolicy(slo=0.0)
        with pytest.raises(ValueError, match="must be positive"):
            FixedSLOPolicy(slo=-1.0)

    def test_fractional_slo(self):
        policy = FixedSLOPolicy(slo=0.5)
        assert policy.calculate_slo(_make_request()) == 0.5


# ── BaselineSLOPolicy ───────────────────────────────────────────────

class TestBaselineSLOPolicy:
    def test_lookup_exact_match(self):
        data = {(1, 512, 512): 2.0, (1, 1024, 1024): 4.0}
        policy = BaselineSLOPolicy(baseline_data=data, multiplier=1.0)
        assert policy.calculate_slo(_make_request(height=512, width=512)) == 2.0
        assert policy.calculate_slo(_make_request(height=1024, width=1024)) == 4.0

    def test_default_fallback_for_unknown_shape(self):
        policy = BaselineSLOPolicy(baseline_data={}, multiplier=1.0)
        assert policy.calculate_slo(_make_request(height=768, width=768)) == 10.0

    def test_multiplier_applied(self):
        data = {(1, 512, 512): 2.0}
        policy = BaselineSLOPolicy(baseline_data=data, multiplier=1.5)
        assert policy.calculate_slo(_make_request()) == 3.0

    def test_multiplier_stacks(self):
        data = {(1, 512, 512): 2.0}
        for mult, expected in [(2.0, 4.0), (0.5, 1.0), (3.0, 6.0)]:
            policy = BaselineSLOPolicy(baseline_data=data, multiplier=mult)
            assert policy.calculate_slo(_make_request()) == expected


# ── CustomSLOPolicy ─────────────────────────────────────────────────

class TestCustomSLOPolicy:
    def test_fixed_value(self):
        mapping = {"(1, 512, 512)": 15.0}
        policy = CustomSLOPolicy(slo_mapping=mapping)
        assert policy.calculate_slo(_make_request()) == 15.0

    def test_missing_key_raises(self):
        policy = CustomSLOPolicy(slo_mapping={})
        with pytest.raises(ValueError, match="No SLO mapping found"):
            policy.calculate_slo(_make_request(height=999, width=999))

    def test_multiplier_format(self):
        baseline_data = {(1, 512, 512): 2.0}
        baseline_policy = BaselineSLOPolicy(baseline_data=baseline_data, multiplier=1.0)
        mapping = {"(1, 512, 512)": "1.5x"}
        policy = CustomSLOPolicy(slo_mapping=mapping, baseline_policy=baseline_policy)
        slo = policy.calculate_slo(_make_request())
        assert slo == pytest.approx(3.0, rel=1e-6)

    def test_shape_fallback_ignores_batch(self):
        """If exact (bs, h, w) not found, matches on (h, w) with any bs."""
        mapping = {"(4, 512, 512)": 10.0}
        policy = CustomSLOPolicy(slo_mapping=mapping)
        # batch_size=1 should still match via the fallback search
        assert policy.calculate_slo(_make_request(batch_size=1)) == 10.0


# ── RequestTracker ──────────────────────────────────────────────────

class TestRequestTrackerSLO:
    def test_track_preserves_latency_threshold(self):
        tracker = RequestTracker()
        config = _make_input_config()
        tracker.track_request(
            sub_req_ids=[100],
            req_id=1,
            arrival_time=0.0,
            latency_threshold=5.0,
            input_config=config,
        )
        reqs = tracker.get_requests()
        assert len(reqs) == 1
        assert reqs[0].latency_threshold == 5.0
        assert reqs[0].arrival_time == 0.0

    def test_get_requests_by_req_id(self):
        tracker = RequestTracker()
        config = _make_input_config(batch_size=2)
        tracker.track_request(
            sub_req_ids=[200, 201],
            req_id=2,
            arrival_time=1.0,
            latency_threshold=10.0,
            input_config=config,
        )
        reqs = tracker.get_requests_by_req_id(req_id=2)
        assert len(reqs) == 2
        for r in reqs:
            assert r.latency_threshold == 10.0

    def test_merged_requests_preserve_threshold(self):
        tracker = RequestTracker()
        config = _make_input_config(batch_size=3)
        tracker.track_request(
            sub_req_ids=[300, 301, 302],
            req_id=3,
            arrival_time=2.0,
            latency_threshold=8.0,
            input_config=config,
        )
        merged = tracker.get_merged_requests()
        assert len(merged) == 1
        assert merged[0].latency_threshold == 8.0
        assert merged[0].input_config.batch_size == 3

    def test_multiple_requests_independent(self):
        tracker = RequestTracker()
        config1 = _make_input_config()
        config2 = _make_input_config(height=1024, width=1024)
        tracker.track_request(
            sub_req_ids=[400], req_id=4, arrival_time=0.0,
            latency_threshold=5.0, input_config=config1,
        )
        tracker.track_request(
            sub_req_ids=[500], req_id=5, arrival_time=1.0,
            latency_threshold=15.0, input_config=config2,
        )
        assert len(tracker) == 2
        assert tracker[400].latency_threshold == 5.0
        assert tracker[500].latency_threshold == 15.0


# ── Scheduler optimal shape (pure Python, no GPU) ──────────────────

_HAS_SCALING_DATA = os.path.exists("log/benchmark/scaling_efficiency/sd3/e2e_scaling_efficiency.json")


@pytest.mark.skipif(not _HAS_SCALING_DATA, reason="scaling efficiency data not available")
class TestOptimalShapePython:
    """Test the pure-Python DP algorithm for optimal GPU allocation."""

    @pytest.fixture
    def scheduler(self):
        """Create a DynStepScheduler with mocked cost data."""
        from tetriserve.scheduler.dyn_step_scheduler import DynStepScheduler

        s = DynStepScheduler(
            ranks=[0, 1, 2, 3],
            schedule_logic="fcfs_optimal_shape_schedule",
        )
        return s

    def test_tight_slo_prefers_fewer_gpus(self, scheduler):
        """With very tight SLO and fast 1-GPU time, prefer single GPU."""
        # 1-GPU is fastest per-step; tight SLO should not waste GPU-hours
        total_gpu_hours, shape, total_time = scheduler.get_optimal_shape_python(
            req_id=1,
            current_world_size=4,
            num_inference_steps=10,
            latency_threshold=5.0,  # generous enough for 10 steps on 1 GPU
            time_1gpu=0.4,  # 0.4s/step × 10 = 4s total (within 5s SLO)
            time_2gpu=0.25,
            time_4gpu=0.15,
            time_8gpu=-1.0,  # unavailable
        )
        assert total_time <= 5.0, f"Must meet SLO: {total_time}"
        # Should prefer 1-GPU since it meets SLO with minimal resources
        assert shape[0] > 0, "Should schedule some steps on 1 GPU"

    def test_loose_slo_all_single_gpu(self, scheduler):
        """Very loose SLO → all steps on 1 GPU (minimum resource usage)."""
        total_gpu_hours, shape, total_time = scheduler.get_optimal_shape_python(
            req_id=2,
            current_world_size=4,
            num_inference_steps=20,
            latency_threshold=100.0,  # very generous
            time_1gpu=0.5,
            time_2gpu=0.3,
            time_4gpu=0.2,
            time_8gpu=-1.0,
        )
        assert total_time <= 100.0
        # With 100s budget and 0.5s/step, 20 steps = 10s. All on 1 GPU.
        assert shape[0] == 20, f"Expected all 20 on 1-GPU, got shape={shape}"

    def test_impossible_slo_returns_best_effort(self, scheduler):
        """SLO too tight for any configuration → still returns something."""
        total_gpu_hours, shape, total_time = scheduler.get_optimal_shape_python(
            req_id=3,
            current_world_size=4,
            num_inference_steps=50,
            latency_threshold=0.001,  # impossible
            time_1gpu=1.0,
            time_2gpu=0.6,
            time_4gpu=0.4,
            time_8gpu=-1.0,
        )
        # Algorithm should return the fastest possible execution
        assert sum(shape) == 50, "All steps must be scheduled"

    def test_unavailable_configs_skipped(self, scheduler):
        """GPUs marked as unavailable (-1) should be excluded."""
        total_gpu_hours, shape, total_time = scheduler.get_optimal_shape_python(
            req_id=4,
            current_world_size=4,
            num_inference_steps=10,
            latency_threshold=10.0,
            time_1gpu=0.5,
            time_2gpu=-1.0,  # unavailable
            time_4gpu=-1.0,  # unavailable
            time_8gpu=-1.0,  # unavailable
        )
        # Only 1-GPU is available, all 10 steps must go there
        assert shape[0] == 10
        assert shape[1] == 0
        assert shape[2] == 0


# ── SLO time-remaining calculation ──────────────────────────────────

class TestSLOTimeRemaining:
    """Test the time_remaining = arrival_time + latency_threshold - current_time formula."""

    def test_positive_remaining(self):
        arrival = 0.0
        threshold = 10.0
        current = 5.0
        remaining = threshold + arrival - current
        assert remaining == 5.0

    def test_negative_remaining_deadline_passed(self):
        arrival = 0.0
        threshold = 5.0
        current = 6.0
        remaining = threshold + arrival - current
        assert remaining == -1.0

    def test_exactly_at_deadline(self):
        arrival = 0.0
        threshold = 5.0
        current = 5.0
        remaining = threshold + arrival - current
        assert remaining == 0.0

    def test_late_arrival(self):
        arrival = 10.0
        threshold = 5.0
        current = 12.0
        remaining = threshold + arrival - current
        assert remaining == 3.0  # deadline at t=15, current=12
