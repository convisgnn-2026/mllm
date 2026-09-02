"""
Statistics computation for workload results.

Simple, pure functions for calculating metrics.
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import numpy as np

from .core import Workload, Request


@dataclass
class WorkloadStats:
    """
    Statistics for a completed workload.

    All metrics in seconds unless otherwise specified.
    """

    # Basic info
    model_name: str
    num_requests: int

    # Throughput metrics
    throughput: float  # requests/second
    request_rate: float  # actual arrival rate

    # Latency metrics (seconds)
    latency_mean: float
    latency_std: float
    latency_median: float
    latency_p50: float
    latency_p90: float
    latency_p95: float
    latency_p99: float
    latency_min: float
    latency_max: float

    # Detailed data
    latencies: np.ndarray  # All latencies
    start_times: np.ndarray  # Request start times
    completion_times: np.ndarray  # Request completion times

    # Shape distribution
    shape_distribution: Dict[Tuple[int, int], int]  # (h, w) -> count

    # SLO metrics
    slo_meet_ratio: Optional[float] = None
    slo_violations: Optional[int] = None

    def __str__(self) -> str:
        """Pretty print statistics."""
        lines = [
            f"Workload Statistics: {self.model_name}",
            f"  Requests: {self.num_requests}",
            f"  Throughput: {self.throughput:.2f} req/s",
            f"  Request Rate: {self.request_rate:.2f} req/s",
            f"  Latency: mean={self.latency_mean:.2f}s, std={self.latency_std:.2f}s",
            f"  Latency: p50={self.latency_p50:.2f}s, p90={self.latency_p90:.2f}s, p99={self.latency_p99:.2f}s",
        ]

        if self.slo_meet_ratio is not None:
            lines.append(f"  SLO Meet Ratio: {self.slo_meet_ratio:.2%}")
            lines.append(f"  SLO Violations: {self.slo_violations}/{self.num_requests}")

        return "\n".join(lines)


def compute_statistics(workload: Workload, warmup_seconds: float = 5.0) -> WorkloadStats:
    """
    Compute statistics for completed workload.

    Args:
        workload: Completed workload
        warmup_seconds: Seconds to skip at start/end for warmup

    Returns:
        WorkloadStats with all metrics

    Raises:
        ValueError: If workload has no completed requests
    """
    # Get requests excluding warmup period
    requests = workload.skip_warmup(warmup_seconds)

    if not requests:
        raise ValueError("No requests found (after warmup filtering)")

    # Check that requests have completion times
    completed = [r for r in requests if r.completion_time is not None]
    if not completed:
        raise ValueError("No completed requests found")

    # Extract times
    start_times = np.array([r.submit_time for r in completed])
    completion_times = np.array([r.completion_time for r in completed])
    latencies = completion_times - start_times

    # Compute throughput
    duration = max(completion_times) - min(start_times)
    throughput = len(completed) / duration if duration > 0 else 0

    # Compute request rate
    arrival_duration = start_times[-1] - start_times[0]
    request_rate = len(completed) / arrival_duration if arrival_duration > 0 else 0

    # Compute latency statistics
    sorted_latencies = np.sort(latencies)

    latency_stats = {
        "mean": float(np.mean(latencies)),
        "std": float(np.std(latencies)),
        "median": float(np.median(latencies)),
        "p50": float(np.percentile(sorted_latencies, 50)),
        "p90": float(np.percentile(sorted_latencies, 90)),
        "p95": float(np.percentile(sorted_latencies, 95)),
        "p99": float(np.percentile(sorted_latencies, 99)),
        "min": float(np.min(latencies)),
        "max": float(np.max(latencies)),
    }

    # Compute shape distribution
    shape_dist = defaultdict(int)
    for req in completed:
        shape = (req.height, req.width)
        shape_dist[shape] += 1

    # Compute SLO metrics if SLOs are set
    slo_meet_ratio = None
    slo_violations = None

    if completed[0].slo is not None:
        slo_met_count = sum(1 for req in completed if req.slo_met)
        slo_meet_ratio = slo_met_count / len(completed)
        slo_violations = len(completed) - slo_met_count

    return WorkloadStats(
        model_name=workload.get_model_name(),
        num_requests=len(completed),
        throughput=throughput,
        request_rate=request_rate,
        latency_mean=latency_stats["mean"],
        latency_std=latency_stats["std"],
        latency_median=latency_stats["median"],
        latency_p50=latency_stats["p50"],
        latency_p90=latency_stats["p90"],
        latency_p95=latency_stats["p95"],
        latency_p99=latency_stats["p99"],
        latency_min=latency_stats["min"],
        latency_max=latency_stats["max"],
        latencies=latencies,
        start_times=start_times,
        completion_times=completion_times,
        shape_distribution=dict(shape_dist),
        slo_meet_ratio=slo_meet_ratio,
        slo_violations=slo_violations,
    )


def print_statistics(stats: WorkloadStats):
    """
    Print statistics in readable format.

    Args:
        stats: WorkloadStats to print
    """
    print(str(stats))

    # Print detailed shape distribution
    print("\n  Shape Distribution:")
    for (h, w), count in sorted(stats.shape_distribution.items()):
        pct = 100.0 * count / stats.num_requests
        print(f"    {h}x{w}: {count} ({pct:.1f}%)")

    # Print SLO details if available
    if stats.slo_meet_ratio is not None:
        print(f"\n  SLO Details:")
        print(f"    Met: {stats.num_requests - stats.slo_violations}")
        print(f"    Violated: {stats.slo_violations}")
        print(f"    Ratio: {stats.slo_meet_ratio:.2%}")


def get_per_request_info(workload: Workload, warmup_seconds: float = 5.0) -> List[Dict]:
    """
    Extract per-request information for analysis.

    Args:
        workload: Workload to extract from
        warmup_seconds: Seconds to skip at start/end

    Returns:
        List of dicts with per-request info (only completed requests)
    """
    requests = workload.skip_warmup(warmup_seconds)

    # Filter to only completed requests (consistent with compute_statistics)
    completed = [r for r in requests if r.completion_time is not None]

    return [
        {
            "req_idx": req.req_idx,
            "batch_size": req.batch_size,
            "height": req.height,
            "width": req.width,
            "resolution": req.resolution,
            "steps": req.num_inference_steps,
            "arrival_time": req.arrival_time,
            "submit_time": req.submit_time,
            "completion_time": req.completion_time,
            "latency": req.latency,
            "slo": req.slo,
            "slo_met": req.slo_met,
        }
        for req in completed  # ← Changed from 'requests' to 'completed'
    ]
