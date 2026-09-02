"""
Clean Workload Package - No global state, no caching, just simple code.

Design Principles:
- KISS: Keep It Simple, Stupid
- No global mutable state
- No hidden caching
- Explicit is better than implicit
- Easy to test
- Type-safe

Basic Usage:
    >>> from tetriserve.workload import (
    ...     WorkloadBuilder,
    ...     PoissonProcess,
    ...     ShapeDistribution,
    ...     FixedSLOPolicy,
    ...     compute_statistics
    ... )
    >>>
    >>> # Create workload
    >>> workload = (
    ...     WorkloadBuilder()
    ...     .with_model("sd3")
    ...     .with_arrival_process(PoissonProcess(rate=1.0))
    ...     .with_duration(60)
    ...     .with_shape_distribution(ShapeDistribution.evenly())
    ...     .with_slo_policy(FixedSLOPolicy(10.0))
    ...     .build()
    ... )
    >>>
    >>> # Visualize workload (before execution)
    >>> print_workload_summary(workload)
    >>> plot_all(workload, output_dir="plots", show=True)
    >>>
    >>> # Execute workload (your execution code here)
    >>> # ...
    >>>
    >>> # Compute statistics
    >>> stats = compute_statistics(workload)
    >>> print(stats)

Advanced Usage:
    See examples in builder.py and slo.py
"""

# Core classes
from .core import Request, Workload

# Arrival processes
from .arrival import (
    ArrivalProcess,
    PoissonProcess,
    GammaProcess,
    DeterministicProcess,
)

# Distributions
from .distribution import ShapeDistribution, BatchSizeDistribution

# Population
from .population import (
    populate_requests,
    populate_requests_with_prompts,
    populate_fixed_requests,
)

# SLO management
from .slo import (
    SLOPolicy,
    FixedSLOPolicy,
    BaselineSLOPolicy,
    CustomSLOPolicy,
    apply_slo_policy,
    load_slo_policy,
)

# Statistics
from .statistics import (
    WorkloadStats,
    compute_statistics,
    print_statistics,
    get_per_request_info,
)

# Builder (recommended way to create workloads)
from .builder import WorkloadBuilder, create_simple_workload

# I/O
from .io import dump_workload, load_workload

# Constants
from .constants import (
    STEPS,
    SHAPE_LIST,
    BS_DISTRIBUTION,
    RATE_LIST,
    STEPS_LIST,
)

# Visualization (lazy import to avoid heavy dependencies)
# Import these when you need them:
#   from tetriserve.workload.visualization import plot_all, print_workload_summary
# Or use the convenience function below:


def plot_all(*args, **kwargs):
    """Lazy import wrapper for plot_all."""
    from .visualization import plot_all as _plot_all

    return _plot_all(*args, **kwargs)


def plot_arrival_pattern(*args, **kwargs):
    """Lazy import wrapper for plot_arrival_pattern."""
    from .visualization import plot_arrival_pattern as _plot

    return _plot(*args, **kwargs)


def plot_request_shapes(*args, **kwargs):
    """Lazy import wrapper for plot_request_shapes."""
    from .visualization import plot_request_shapes as _plot

    return _plot(*args, **kwargs)


def plot_workload_timeline(*args, **kwargs):
    """Lazy import wrapper for plot_workload_timeline."""
    from .visualization import plot_workload_timeline as _plot

    return _plot(*args, **kwargs)


def plot_burstiness_analysis(*args, **kwargs):
    """Lazy import wrapper for plot_burstiness_analysis."""
    from .visualization import plot_burstiness_analysis as _plot

    return _plot(*args, **kwargs)


def print_workload_summary(*args, **kwargs):
    """Lazy import wrapper for print_workload_summary."""
    from .visualization import print_workload_summary as _print

    return _print(*args, **kwargs)


__all__ = [
    # Core
    "Request",
    "Workload",
    # Arrival
    "ArrivalProcess",
    "PoissonProcess",
    "GammaProcess",
    "DeterministicProcess",
    # Distribution
    "ShapeDistribution",
    "BatchSizeDistribution",
    # Population
    "populate_requests",
    "populate_requests_with_prompts",
    "populate_fixed_requests",
    # SLO
    "SLOPolicy",
    "FixedSLOPolicy",
    "BaselineSLOPolicy",
    "CustomSLOPolicy",
    "apply_slo_policy",
    "load_slo_policy",
    # Statistics
    "WorkloadStats",
    "compute_statistics",
    "print_statistics",
    "get_per_request_info",
    # Builder
    "WorkloadBuilder",
    "create_simple_workload",
    # I/O
    "dump_workload",
    "load_workload",
    # Constants
    "STEPS",
    "SHAPE_LIST",
    "BS_DISTRIBUTION",
    "RATE_LIST",
    "STEPS_LIST",
    # Visualization
    "plot_arrival_pattern",
    "plot_request_shapes",
    "plot_workload_timeline",
    "plot_burstiness_analysis",
    "plot_all",
    "print_workload_summary",
]

__version__ = "2.0.0-clean"
