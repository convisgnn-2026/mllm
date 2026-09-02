"""
Workload visualization functions.

Simple plotting utilities to visualize workload characteristics:
- Arrival patterns (burstiness)
- Request shapes over time
- Batch sizes
- SLO distributions
"""

from typing import Optional, List, Tuple
from pathlib import Path
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.figure import Figure

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None

from .core import Workload, Request

# Use standard logging to avoid heavy import chain
import logging

logger = logging.getLogger(__name__)


def _check_matplotlib():
    """Check if matplotlib is available."""
    if not HAS_MATPLOTLIB:
        raise ImportError("Matplotlib is required for visualization. " "Install it with: pip install matplotlib")


def plot_arrival_pattern(
    workload: Workload, output_path: Optional[str] = None, show: bool = True, figsize: Tuple[int, int] = (12, 8)
):
    """
    Plot arrival pattern to visualize burstiness.

    Creates two subplots:
    1. Cumulative arrivals over time (shows rate)
    2. Inter-arrival times (shows burstiness)

    Args:
        workload: Workload to visualize
        output_path: Path to save figure (if None, doesn't save)
        show: Whether to display the plot
        figsize: Figure size (width, height)

    Returns:
        matplotlib Figure object
    """
    _check_matplotlib()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

    # Plot 1: Cumulative arrivals
    cumulative_count = np.arange(1, len(workload.arrival_times) + 1)
    ax1.plot(workload.arrival_times, cumulative_count, "b-", linewidth=2)
    ax1.set_xlabel("Time (seconds)", fontsize=12)
    ax1.set_ylabel("Cumulative Requests", fontsize=12)
    ax1.set_title(
        f"Arrival Pattern: {workload.get_model_name()} " f"(Rate={workload.mean_rate:.2f}/s, CV={workload.cv:.2f})",
        fontsize=14,
        fontweight="bold",
    )
    ax1.grid(True, alpha=0.3)

    # Add expected line for comparison (constant rate)
    if len(workload.arrival_times) > 1:
        expected_line = cumulative_count[0] + workload.mean_rate * (workload.arrival_times - workload.arrival_times[0])
        ax1.plot(
            workload.arrival_times, expected_line, "r--", label=f"Expected (rate={workload.mean_rate:.2f}/s)", alpha=0.7
        )
        ax1.legend()

    # Plot 2: Inter-arrival times (shows burstiness)
    if len(workload.arrival_times) > 1:
        intervals = workload.arrival_times[1:] - workload.arrival_times[:-1]
        interval_times = workload.arrival_times[1:]

        ax2.scatter(interval_times, intervals, alpha=0.6, s=20)
        ax2.axhline(
            y=1.0 / workload.mean_rate,
            color="r",
            linestyle="--",
            label=f"Mean interval ({1.0/workload.mean_rate:.3f}s)",
        )
        ax2.set_xlabel("Time (seconds)", fontsize=12)
        ax2.set_ylabel("Inter-arrival Time (seconds)", fontsize=12)
        ax2.set_title(f"Burstiness: CV={workload.cv:.2f} " f"(CV=1: Poisson, CV>1: Bursty, CV<1: Regular)", fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # Add stats box
        stats_text = (
            f"Mean: {np.mean(intervals):.3f}s\n"
            f"Std: {np.std(intervals):.3f}s\n"
            f"Min: {np.min(intervals):.3f}s\n"
            f"Max: {np.max(intervals):.3f}s"
        )
        ax2.text(
            0.02,
            0.98,
            stats_text,
            transform=ax2.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            fontsize=10,
        )

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved arrival pattern plot to {output_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def plot_request_shapes(
    workload: Workload, output_path: Optional[str] = None, show: bool = True, figsize: Tuple[int, int] = (14, 10)
):
    """
    Plot request characteristics over time.

    Creates subplots showing:
    1. Request shapes (resolution) over time
    2. Batch sizes over time
    3. SLO thresholds (if available)
    4. Shape distribution histogram

    Args:
        workload: Workload to visualize
        output_path: Path to save figure
        show: Whether to display the plot
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    _check_matplotlib()

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    ax1, ax2, ax3, ax4 = axes.flatten()

    # Extract data
    arrivals = workload.arrival_times
    resolutions = [req.resolution for req in workload.requests]
    batch_sizes = [req.batch_size for req in workload.requests]
    shapes = [(req.height, req.width) for req in workload.requests]

    # Define colors for different resolutions
    resolution_colors = {
        256 * 256: "blue",
        512 * 512: "green",
        1024 * 1024: "orange",
        2048 * 2048: "red",
    }

    # Plot 1: Resolutions over time
    for resolution, color_key in resolution_colors.items():
        mask = [r == resolution for r in resolutions]
        if any(mask):
            masked_arrivals = arrivals[mask]
            masked_resolutions = np.array(resolutions)[mask]
            size = int(resolution**0.5)
            ax1.scatter(
                masked_arrivals,
                masked_resolutions,
                label=f"{size}x{size}",
                alpha=0.6,
                s=50,
                color=resolution_colors[resolution],
            )

    ax1.set_xlabel("Arrival Time (seconds)", fontsize=11)
    ax1.set_ylabel("Resolution (pixels²)", fontsize=11)
    ax1.set_title("Request Resolutions Over Time", fontsize=12, fontweight="bold")
    ax1.set_yscale("log")
    ax1.legend(title="Shape", loc="best")
    ax1.grid(True, alpha=0.3)

    # Plot 2: Batch sizes over time
    unique_batch_sizes = sorted(set(batch_sizes))
    for bs in unique_batch_sizes:
        mask = [b == bs for b in batch_sizes]
        if any(mask):
            masked_arrivals = arrivals[mask]
            masked_bs = np.array(batch_sizes)[mask]
            ax2.scatter(masked_arrivals, masked_bs, label=f"BS={bs}", alpha=0.6, s=50)

    ax2.set_xlabel("Arrival Time (seconds)", fontsize=11)
    ax2.set_ylabel("Batch Size", fontsize=11)
    ax2.set_title("Batch Sizes Over Time", fontsize=12, fontweight="bold")
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.3)

    # Plot 3: SLO thresholds (if available)
    slos = [req.slo for req in workload.requests if req.slo is not None]
    if slos:
        for resolution, color_key in resolution_colors.items():
            mask = [
                (r == resolution and s is not None) for r, s in zip(resolutions, [req.slo for req in workload.requests])
            ]
            if any(mask):
                masked_arrivals = arrivals[mask]
                masked_slos = np.array([req.slo for req in workload.requests])[mask]
                size = int(resolution**0.5)
                ax3.scatter(
                    masked_arrivals,
                    masked_slos,
                    label=f"{size}x{size}",
                    alpha=0.6,
                    s=50,
                    color=resolution_colors[resolution],
                )

        ax3.set_xlabel("Arrival Time (seconds)", fontsize=11)
        ax3.set_ylabel("SLO (seconds)", fontsize=11)
        ax3.set_title("SLO Thresholds by Shape", fontsize=12, fontweight="bold")
        ax3.legend(loc="best")
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, "No SLO data available", ha="center", va="center", transform=ax3.transAxes, fontsize=14)
        ax3.set_title("SLO Thresholds", fontsize=12, fontweight="bold")

    # Plot 4: Shape distribution histogram
    from collections import Counter

    shape_counts = Counter(shapes)
    shape_labels = [f"{h}x{w}" for h, w in sorted(shape_counts.keys())]
    shape_values = [shape_counts[shape] for shape in sorted(shape_counts.keys())]
    colors = [resolution_colors.get(h * w, "gray") for h, w in sorted(shape_counts.keys())]

    bars = ax4.bar(range(len(shape_labels)), shape_values, color=colors, alpha=0.7)
    ax4.set_xticks(range(len(shape_labels)))
    ax4.set_xticklabels(shape_labels)
    ax4.set_ylabel("Count", fontsize=11)
    ax4.set_title("Shape Distribution", fontsize=12, fontweight="bold")
    ax4.grid(True, alpha=0.3, axis="y")

    # Add percentage labels on bars
    total = sum(shape_values)
    for bar, count in zip(bars, shape_values):
        height = bar.get_height()
        pct = 100.0 * count / total
        ax4.text(bar.get_x() + bar.get_width() / 2.0, height, f"{pct:.1f}%", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved request shapes plot to {output_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def plot_workload_timeline(
    workload: Workload, output_path: Optional[str] = None, show: bool = True, figsize: Tuple[int, int] = (16, 10)
):
    """
    Plot complete workload timeline with all information.

    Shows each request as a horizontal bar with:
    - Arrival time
    - Shape (color-coded)
    - SLO deadline (if available)
    - Completion time (if executed)

    Args:
        workload: Workload to visualize
        output_path: Path to save figure
        show: Whether to display the plot
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    _check_matplotlib()

    fig, ax = plt.subplots(figsize=figsize)

    # Resolution colors
    resolution_colors = {
        256 * 256: "blue",
        512 * 512: "green",
        1024 * 1024: "orange",
        2048 * 2048: "red",
    }

    # Plot each request
    for i, req in enumerate(workload.requests):
        arrival = req.arrival_time
        resolution = req.resolution
        color = resolution_colors.get(resolution, "gray")

        # Plot arrival point
        ax.plot(arrival, i, "o", color=color, markersize=8, alpha=0.7)

        # Plot SLO deadline if available
        if req.slo is not None:
            slo_deadline = arrival + req.slo
            ax.plot([arrival, slo_deadline], [i, i], color=color, linestyle="--", alpha=0.3, linewidth=2)
            ax.plot(slo_deadline, i, "|", color="red", markersize=15, markeredgewidth=2, alpha=0.5)

        # Plot completion if available
        if req.completion_time is not None:
            ax.plot(
                [req.submit_time, req.completion_time],
                [i, i],
                color=color,
                linewidth=3,
                alpha=0.8,
                solid_capstyle="butt",
            )

            # Mark if SLO was violated
            if req.slo is not None and not req.slo_met:
                # Red background for SLO violation
                rect = patches.Rectangle(
                    (arrival + req.slo, i - 0.3),
                    req.completion_time - (arrival + req.slo),
                    0.6,
                    facecolor="red",
                    alpha=0.2,
                )
                ax.add_patch(rect)

    # Customize plot
    ax.set_xlabel("Time (seconds)", fontsize=12)
    ax.set_ylabel("Request Index", fontsize=12)
    ax.set_title(
        f"Workload Timeline: {workload.get_model_name()} "
        f"({len(workload)} requests, rate={workload.mean_rate:.2f}/s, "
        f"CV={workload.cv:.2f})",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_ylim(-1, len(workload))
    ax.grid(True, alpha=0.3, axis="x")

    # Create legend
    from matplotlib.lines import Line2D

    legend_elements = []

    # Add shape legend
    for resolution, color in sorted(resolution_colors.items()):
        size = int(resolution**0.5)
        legend_elements.append(
            Line2D([0], [0], marker="o", color="w", markerfacecolor=color, markersize=8, label=f"{size}x{size}")
        )

    # Add status legend
    legend_elements.extend(
        [
            Line2D([0], [0], marker="o", color="gray", markersize=8, label="Arrival"),
            Line2D([0], [0], color="gray", linestyle="--", label="SLO Window"),
            Line2D(
                [0],
                [0],
                marker="|",
                color="red",
                markersize=10,
                markeredgewidth=2,
                linestyle="None",
                label="SLO Deadline",
            ),
        ]
    )

    if workload.get_completed_requests():
        legend_elements.append(Line2D([0], [0], color="gray", linewidth=3, label="Execution"))

    ax.legend(handles=legend_elements, loc="upper left", ncol=2, fontsize=10, framealpha=0.9)

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved workload timeline to {output_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def plot_burstiness_analysis(
    workload: Workload, output_path: Optional[str] = None, show: bool = True, figsize: Tuple[int, int] = (14, 10)
):
    """
    Detailed burstiness analysis.

    Creates multiple plots to analyze arrival patterns:
    1. Inter-arrival time distribution (histogram)
    2. Inter-arrival time over time (scatter)
    3. Rolling window statistics
    4. Q-Q plot against exponential distribution

    Args:
        workload: Workload to analyze
        output_path: Path to save figure
        show: Whether to display the plot
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    _check_matplotlib()

    if len(workload.arrival_times) <= 1:
        logger.warning("Need at least 2 requests for burstiness analysis")
        return None

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    ax1, ax2, ax3, ax4 = axes.flatten()

    intervals = workload.arrival_times[1:] - workload.arrival_times[:-1]
    mean_interval = np.mean(intervals)

    # Plot 1: Histogram of inter-arrival times
    ax1.hist(intervals, bins=30, alpha=0.7, color="skyblue", edgecolor="black")
    ax1.axvline(mean_interval, color="red", linestyle="--", linewidth=2, label=f"Mean={mean_interval:.3f}s")
    ax1.set_xlabel("Inter-arrival Time (seconds)", fontsize=11)
    ax1.set_ylabel("Frequency", fontsize=11)
    ax1.set_title("Inter-arrival Time Distribution", fontsize=12, fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")

    # Plot 2: Inter-arrival times over time (shows temporal patterns)
    interval_times = workload.arrival_times[1:]
    ax2.plot(interval_times, intervals, "o-", alpha=0.6, markersize=4)
    ax2.axhline(mean_interval, color="red", linestyle="--", label=f"Mean={mean_interval:.3f}s")
    ax2.set_xlabel("Time (seconds)", fontsize=11)
    ax2.set_ylabel("Inter-arrival Time (seconds)", fontsize=11)
    ax2.set_title("Inter-arrival Times Over Time", fontsize=12, fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Rolling window statistics (shows burstiness over time)
    window_size = min(10, len(intervals) // 5)  # Adaptive window
    if window_size >= 2:
        rolling_mean = np.convolve(intervals, np.ones(window_size) / window_size, mode="valid")
        rolling_std = np.array(
            [np.std(intervals[i : i + window_size]) for i in range(len(intervals) - window_size + 1)]
        )
        rolling_times = interval_times[window_size - 1 :]

        ax3_twin = ax3.twinx()

        line1 = ax3.plot(rolling_times, rolling_mean, "b-", linewidth=2, label="Rolling Mean", alpha=0.8)
        line2 = ax3_twin.plot(rolling_times, rolling_std, "r-", linewidth=2, label="Rolling Std Dev", alpha=0.8)

        ax3.set_xlabel("Time (seconds)", fontsize=11)
        ax3.set_ylabel("Mean Inter-arrival (seconds)", color="b", fontsize=11)
        ax3_twin.set_ylabel("Std Dev (seconds)", color="r", fontsize=11)
        ax3.set_title(f"Burstiness Over Time (window={window_size})", fontsize=12, fontweight="bold")
        ax3.tick_params(axis="y", labelcolor="b")
        ax3_twin.tick_params(axis="y", labelcolor="r")

        # Combined legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax3.legend(lines, labels, loc="best")
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(
            0.5,
            0.5,
            "Not enough data for rolling statistics",
            ha="center",
            va="center",
            transform=ax3.transAxes,
            fontsize=12,
        )

    # Plot 4: Q-Q plot against exponential (Poisson process)
    from scipy import stats as scipy_stats

    sorted_intervals = np.sort(intervals)
    theoretical_quantiles = scipy_stats.expon.ppf(np.linspace(0.01, 0.99, len(intervals)), scale=mean_interval)

    ax4.scatter(theoretical_quantiles, sorted_intervals, alpha=0.6, s=20)

    # Add diagonal line (perfect exponential)
    max_val = max(theoretical_quantiles.max(), sorted_intervals.max())
    ax4.plot([0, max_val], [0, max_val], "r--", linewidth=2, label="Perfect Exponential")

    ax4.set_xlabel("Theoretical Quantiles (Exponential)", fontsize=11)
    ax4.set_ylabel("Sample Quantiles", fontsize=11)
    ax4.set_title(f"Q-Q Plot: Exponential Distribution " f"(CV={workload.cv:.2f})", fontsize=12, fontweight="bold")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Add interpretation text
    if workload.cv < 0.9:
        interpretation = "More regular than Poisson"
    elif workload.cv > 1.1:
        interpretation = "More bursty than Poisson"
    else:
        interpretation = "Similar to Poisson"

    ax4.text(
        0.02,
        0.98,
        f"CV={workload.cv:.2f}\n{interpretation}",
        transform=ax4.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7),
        fontsize=10,
    )

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved burstiness analysis to {output_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def plot_all(
    workload: Workload, output_path: Optional[str] = None, show: bool = False, figsize: Tuple[int, int] = (16, 10)
):
    """
    Create comprehensive visualization with all plots in one figure.

    Creates a single figure with 4 subplots showing:
    - Top row: Arrival pattern (cumulative & inter-arrival times)
    - Bottom row: Request characteristics (shapes over time & distribution)

    Args:
        workload: Workload to visualize
        output_path: Path to save figure (if None, doesn't save)
        show: Whether to display the plot
        figsize: Figure size (width, height)

    Returns:
        matplotlib Figure object with all subplots
    """
    _check_matplotlib()

    # Create figure with 2 rows, 2 columns
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Define colors for different resolutions
    resolution_colors = {
        256 * 256: "blue",
        512 * 512: "green",
        1024 * 1024: "orange",
        2048 * 2048: "red",
    }

    # Extract data
    arrivals = workload.arrival_times
    resolutions = [req.resolution for req in workload.requests]
    batch_sizes = [req.batch_size for req in workload.requests]
    shapes = [(req.height, req.width) for req in workload.requests]

    # ========== TOP ROW: Arrival Pattern ==========

    # Subplot 1: Cumulative arrivals
    ax1 = fig.add_subplot(gs[0, 0])
    cumulative_count = np.arange(1, len(arrivals) + 1)
    ax1.plot(arrivals, cumulative_count, "b-", linewidth=2, label="Actual")

    # Add expected line
    if len(arrivals) > 1:
        expected_line = cumulative_count[0] + workload.mean_rate * (arrivals - arrivals[0])
        ax1.plot(arrivals, expected_line, "r--", label=f"Expected (rate={workload.mean_rate:.2f}/s)", alpha=0.7)

    ax1.set_xlabel("Time (seconds)", fontsize=10)
    ax1.set_ylabel("Cumulative Requests", fontsize=10)
    ax1.set_title("Cumulative Arrivals", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Instantaneous request rate (shows bursts as peaks)
    ax2 = fig.add_subplot(gs[0, 1])
    if len(arrivals) > 5:  # Need enough data for rolling window
        # Calculate instantaneous rate using rolling window
        window_size = max(5, len(arrivals) // 20)  # Adaptive window

        # Use rolling window to count requests per time unit
        time_points = []
        instantaneous_rates = []

        for i in range(window_size, len(arrivals)):
            # Count requests in sliding window
            window_start = arrivals[i - window_size]
            window_end = arrivals[i]
            window_duration = window_end - window_start

            if window_duration > 0:
                rate = window_size / window_duration  # requests per second
                time_points.append(window_end)
                instantaneous_rates.append(rate)

        # Plot instantaneous rate
        ax2.plot(time_points, instantaneous_rates, "b-", linewidth=2, alpha=0.8, label="Instantaneous Rate")
        ax2.fill_between(time_points, instantaneous_rates, alpha=0.3)

        # Add mean line
        ax2.axhline(
            y=workload.mean_rate, color="r", linestyle="--", label=f"Mean Rate={workload.mean_rate:.2f}/s", linewidth=2
        )

        # Add ±1 std deviation bands
        if len(instantaneous_rates) > 1:
            rate_std = np.std(instantaneous_rates)
            ax2.axhline(y=workload.mean_rate + rate_std, color="orange", linestyle=":", alpha=0.5, linewidth=1.5)
            ax2.axhline(y=workload.mean_rate - rate_std, color="orange", linestyle=":", alpha=0.5, linewidth=1.5)
            ax2.fill_between(
                [arrivals[0], arrivals[-1]],
                [workload.mean_rate - rate_std] * 2,
                [workload.mean_rate + rate_std] * 2,
                color="orange",
                alpha=0.1,
                label="±1 Std Dev",
            )

        ax2.set_xlabel("Time (seconds)", fontsize=10)
        ax2.set_ylabel("Request Rate (req/s)", fontsize=10)
        ax2.set_title(f"Burstiness: CV={workload.cv:.2f} (Peaks = Bursts)", fontsize=11, fontweight="bold")
        ax2.legend(fontsize=9, loc="best")
        ax2.grid(True, alpha=0.3)

        # Add interpretation
        if workload.cv < 0.9:
            interp = "Regular"
            color = "green"
        elif workload.cv > 1.1:
            interp = "Bursty"
            color = "red"
        else:
            interp = "Poisson-like"
            color = "blue"

        ax2.text(
            0.98,
            0.98,
            f"{interp}",
            transform=ax2.transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor=color, alpha=0.3),
            fontsize=11,
            fontweight="bold",
        )
    else:
        ax2.text(
            0.5,
            0.5,
            "Need more requests for rate analysis",
            ha="center",
            va="center",
            transform=ax2.transAxes,
            fontsize=12,
        )

    # ========== BOTTOM ROW: Request Characteristics ==========

    # Subplot 3: Request resolutions over time
    ax3 = fig.add_subplot(gs[1, 0])
    for resolution, color in resolution_colors.items():
        mask = [r == resolution for r in resolutions]
        if any(mask):
            masked_arrivals = arrivals[mask]
            masked_resolutions = np.array(resolutions)[mask]
            size = int(resolution**0.5)
            ax3.scatter(masked_arrivals, masked_resolutions, label=f"{size}x{size}", alpha=0.6, s=40, color=color)

    ax3.set_xlabel("Arrival Time (seconds)", fontsize=10)
    ax3.set_ylabel("Resolution (pixels²)", fontsize=10)
    ax3.set_title("Request Shapes Over Time", fontsize=11, fontweight="bold")
    ax3.set_yscale("log")
    ax3.legend(title="Shape", fontsize=9, ncol=2)
    ax3.grid(True, alpha=0.3)

    # Subplot 4: Shape distribution
    ax4 = fig.add_subplot(gs[1, 1])
    from collections import Counter

    shape_counts = Counter(shapes)
    shape_labels = [f"{h}x{w}" for h, w in sorted(shape_counts.keys())]
    shape_values = [shape_counts[shape] for shape in sorted(shape_counts.keys())]
    colors = [resolution_colors.get(h * w, "gray") for h, w in sorted(shape_counts.keys())]

    bars = ax4.bar(range(len(shape_labels)), shape_values, color=colors, alpha=0.7)
    ax4.set_xticks(range(len(shape_labels)))
    ax4.set_xticklabels(shape_labels, fontsize=10)
    ax4.set_ylabel("Count", fontsize=10)
    ax4.set_title("Shape Distribution", fontsize=11, fontweight="bold")
    ax4.grid(True, alpha=0.3, axis="y")

    # Add percentage labels
    total = sum(shape_values)
    for bar, count in zip(bars, shape_values):
        height = bar.get_height()
        pct = 100.0 * count / total
        ax4.text(bar.get_x() + bar.get_width() / 2.0, height, f"{pct:.1f}%", ha="center", va="bottom", fontsize=9)

    # Add overall title
    fig.suptitle(
        f"Workload Analysis: {workload.get_model_name()} | "
        f"{len(workload)} requests | "
        f"Rate={workload.mean_rate:.2f}/s | "
        f"CV={workload.cv:.2f}",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved comprehensive visualization to {output_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def print_workload_summary(workload: Workload):
    """
    Print a text summary of the workload.

    Args:
        workload: Workload to summarize
    """
    print("=" * 70)
    print(f"Workload Summary: {workload.get_model_name()}")
    print("=" * 70)

    print(f"\n📊 Basic Statistics:")
    print(f"  Total Requests: {len(workload)}")
    print(f"  Duration: {workload.arrival_times[-1] - workload.arrival_times[0]:.2f}s")
    print(f"  Mean Rate: {workload.mean_rate:.2f} requests/second")
    print(f"  CV: {workload.cv:.2f}", end="")

    if workload.cv < 0.9:
        print(" (More regular than Poisson)")
    elif workload.cv > 1.1:
        print(" (More bursty than Poisson)")
    else:
        print(" (Similar to Poisson)")

    # Inter-arrival times
    if len(workload.arrival_times) > 1:
        intervals = workload.arrival_times[1:] - workload.arrival_times[:-1]
        print(f"\n⏱️  Inter-arrival Times:")
        print(f"  Mean: {np.mean(intervals):.3f}s")
        print(f"  Std: {np.std(intervals):.3f}s")
        print(f"  Min: {np.min(intervals):.3f}s")
        print(f"  Max: {np.max(intervals):.3f}s")

    # Shape distribution
    from collections import Counter

    shapes = Counter((req.height, req.width) for req in workload.requests)

    print(f"\n🖼️  Shape Distribution:")
    for (h, w), count in sorted(shapes.items()):
        pct = 100.0 * count / len(workload)
        resolution = h * w
        print(f"  {h}x{w} ({resolution:,} pixels): {count} ({pct:.1f}%)")

    # Batch sizes
    batch_sizes = Counter(req.batch_size for req in workload.requests)
    print(f"\n📦 Batch Size Distribution:")
    for bs, count in sorted(batch_sizes.items()):
        pct = 100.0 * count / len(workload)
        print(f"  Batch {bs}: {count} ({pct:.1f}%)")

    # Steps
    steps = Counter(req.num_inference_steps for req in workload.requests)
    print(f"\n🔢 Inference Steps:")
    for step, count in sorted(steps.items()):
        pct = 100.0 * count / len(workload)
        print(f"  {step} steps: {count} ({pct:.1f}%)")

    # SLO information
    slos = [req.slo for req in workload.requests if req.slo is not None]
    if slos:
        print(f"\n⏰ SLO Information:")
        print(f"  Requests with SLO: {len(slos)}/{len(workload)}")
        print(f"  SLO Range: {min(slos):.2f}s - {max(slos):.2f}s")
        print(f"  Mean SLO: {np.mean(slos):.2f}s")

    # Completion information
    completed = workload.get_completed_requests()
    if completed:
        latencies = [req.latency for req in completed]
        print(f"\n✅ Execution Results:")
        print(f"  Completed: {len(completed)}/{len(workload)}")
        print(f"  Mean Latency: {np.mean(latencies):.2f}s")
        print(f"  P90 Latency: {np.percentile(latencies, 90):.2f}s")

        if slos:
            slo_met_count = sum(1 for req in completed if req.slo_met)
            slo_meet_ratio = slo_met_count / len(completed)
            print(f"  SLO Meet Ratio: {slo_meet_ratio:.1%}")

    print("=" * 70)
