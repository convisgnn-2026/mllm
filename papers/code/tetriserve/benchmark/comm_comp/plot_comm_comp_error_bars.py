import json
import matplotlib.pyplot as plt
import numpy as np


def load_and_plot_timing_data(json_file_path):
    """Load timing data from JSON and create error bar plot"""

    # Load the JSON data
    with open(json_file_path, "r") as f:
        data = json.load(f)

    # Extract data for plotting
    resolutions = []
    comp_times = []
    comp_stds = []
    comm_times = []
    comm_stds = []

    for entry in data:
        height = entry["height"]
        width = entry["width"]
        resolution_str = f"{height}×{width}"

        resolutions.append(resolution_str)
        comp_times.append(entry["comp_time_avg"])
        comp_stds.append(entry["comp_time_std"])
        comm_times.append(entry["comm_time_avg"])
        comm_stds.append(entry["comm_time_std"])

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))

    x = np.arange(len(resolutions))
    width = 0.35

    # Create stacked bars with error bars
    # Bottom bars for computation
    bars_comp = ax.bar(
        x, comp_times, width, label="Computation", yerr=comp_stds, capsize=5, color="#1f77b4", alpha=0.8, hatch="///"
    )

    # Top bars for communication (stacked on top of computation)
    bars_comm = ax.bar(
        x,
        comm_times,
        width,
        bottom=comp_times,
        label="Communication",
        yerr=comm_stds,
        capsize=5,
        color="#ff7f0e",
        alpha=0.8,
    )

    # Customize the plot
    ax.set_xlabel("Height × Width", fontsize=14, fontweight="bold")
    ax.set_ylabel("End-to-End Time (s)", fontsize=14, fontweight="bold")
    ax.set_title(
        "Communication and Computation Time Breakdown\nwith Error Bars (FLUX Model, P=1)",
        fontsize=16,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(resolutions, fontsize=12)
    ax.legend(fontsize=12)

    # Add grid for better readability
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_axisbelow(True)

    # Adjust layout
    plt.tight_layout()

    # Print statistics
    print("Timing Statistics:")
    print("=" * 50)
    for i, resolution in enumerate(resolutions):
        print(f"{resolution}:")
        print(f"  Computation: {comp_times[i]:.4f} ± {comp_stds[i]:.4f} s")
        print(f"  Communication: {comm_times[i]:.4f} ± {comm_stds[i]:.4f} s")
        print(f"  Total E2E: {comp_times[i] + comm_times[i]:.4f} s")
        print()

    return fig, ax


def plot_multi_parallel_comparison_original(json_files_dict):
    """Create the original comprehensive plot without axis break"""

    # Colors for different P values
    colors = {
        1: "#90EE90",  # Light green for P=1
        2: "#87CEEB",  # Light blue for P=2
        4: "#FFB366",  # Light orange for P=4
        8: "#DDA0DD",  # Light purple for P=8
    }

    all_resolutions = set()
    all_data = {}

    # Load data from all files
    for p_value, file_path in json_files_dict.items():
        try:
            with open(file_path, "r") as f:
                data = json.load(f)

            all_data[p_value] = {}
            for entry in data:
                resolution = f"{entry['height']}×{entry['width']}"
                all_resolutions.add(resolution)
                all_data[p_value][resolution] = {
                    "comp_avg": entry["comp_time_avg"],
                    "comp_std": entry["comp_time_std"],
                    "comm_avg": entry["comm_time_avg"],
                    "comm_std": entry["comm_time_std"],
                }
        except FileNotFoundError:
            print(f"Warning: File {file_path} not found, skipping P={p_value}")
            continue

    # Sort resolutions by image size
    resolution_order = ["256×256", "512×512", "1024×1024", "2048×2048"]
    resolutions = [r for r in resolution_order if r in all_resolutions]

    x = np.arange(len(resolutions))
    width = 0.18

    # Create single figure (original style)
    fig, ax = plt.subplots(figsize=(14, 10))

    # Plot bars for each P value
    for i, p_value in enumerate(sorted(all_data.keys())):
        if p_value not in all_data:
            continue

        comp_times = []
        comp_stds = []
        comm_times = []
        comm_stds = []

        for resolution in resolutions:
            if resolution in all_data[p_value]:
                data_point = all_data[p_value][resolution]
                comp_times.append(data_point["comp_avg"])
                comp_stds.append(data_point["comp_std"])
                comm_times.append(data_point["comm_avg"])
                comm_stds.append(data_point["comm_std"])
            else:
                comp_times.append(0)
                comp_stds.append(0)
                comm_times.append(0)
                comm_stds.append(0)

        # Plot computation bars with hatching pattern
        bars_comp = ax.bar(
            x + i * width,
            comp_times,
            width,
            label=f"P={p_value}",
            yerr=comp_stds,
            capsize=5,
            color=colors[p_value],
            alpha=0.8,
            hatch="///",  # Forward slashes for computation
            edgecolor="black",
            linewidth=0.5,
        )

        # Plot communication on top (if any)
        if any(comm_times):
            bars_comm = ax.bar(
                x + i * width,
                comm_times,
                width,
                bottom=comp_times,
                yerr=comm_stds,
                capsize=5,
                color="#808080",
                alpha=0.8,
                hatch="\\\\\\",  # Backslashes for communication
                edgecolor="black",
                linewidth=0.5,
            )

    # Customize the plot
    ax.set_xlabel("Height × Width", fontsize=16, fontweight="bold")
    ax.set_ylabel("End-to-End Time (s)", fontsize=16, fontweight="bold")
    ax.set_title(
        "Communication and Computation Time Breakdown\nFLUX Model - Different Parallel Configurations",
        fontsize=18,
        fontweight="bold",
        pad=20,
    )

    # Set x-axis
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(resolutions, fontsize=14)

    # Create custom legend with proper labels
    from matplotlib.patches import Patch

    legend_elements = []

    # Add hatched pattern for computation and communication
    legend_elements.append(Patch(facecolor="white", hatch="///", edgecolor="black", label="Computation"))
    legend_elements.append(
        Patch(facecolor="#808080", hatch="\\\\\\", alpha=0.8, edgecolor="black", label="Communication")
    )

    # Add P value labels
    for p_value in sorted(all_data.keys()):
        legend_elements.append(Patch(facecolor=colors[p_value], alpha=0.8, edgecolor="black", label=f"P={p_value}"))

    # Position legend outside the plot area
    fig.legend(
        handles=legend_elements,
        fontsize=11,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        fancybox=True,
        shadow=True,
        ncol=1,
    )

    # Add grid
    ax.grid(True, alpha=0.3, axis="y", linestyle="-", linewidth=0.5)
    ax.set_axisbelow(True)

    # Set y-axis to start from 0
    ax.set_ylim(bottom=0)

    # Improve layout
    plt.tight_layout()
    return fig, ax


def plot_multi_parallel_comparison(json_files_dict):
    """Create a comprehensive plot with axis break comparing different parallel configurations"""

    # Colors for different P values
    colors = {
        1: "#90EE90",  # Light green for P=1
        2: "#87CEEB",  # Light blue for P=2
        4: "#FFB366",  # Light orange for P=4
        8: "#DDA0DD",  # Light purple for P=8
    }

    all_resolutions = set()
    all_data = {}

    # Load data from all files
    for p_value, file_path in json_files_dict.items():
        try:
            with open(file_path, "r") as f:
                data = json.load(f)

            all_data[p_value] = {}
            for entry in data:
                resolution = f"{entry['height']}×{entry['width']}"
                all_resolutions.add(resolution)
                all_data[p_value][resolution] = {
                    "comp_avg": entry["comp_time_avg"],
                    "comp_std": entry["comp_time_std"],
                    "comm_avg": entry["comm_time_avg"],
                    "comm_std": entry["comm_time_std"],
                }
        except FileNotFoundError:
            print(f"Warning: File {file_path} not found, skipping P={p_value}")
            continue

    # Sort resolutions by image size
    resolution_order = ["256×256", "512×512", "1024×1024", "2048×2048"]
    resolutions = [r for r in resolution_order if r in all_resolutions]

    x = np.arange(len(resolutions))
    width = 0.18

    # Create figure with axis break
    # Calculate proportional height ratios based on data ranges
    lower_range = 30 - 0  # 30 units (0-30)
    middle_range = 60 - 45  # 15 units (45-60)
    upper_range = 105 - 90  # 15 units (90-105)
    total_range = lower_range + middle_range + upper_range  # 60 units

    # Make height ratios proportional to data ranges
    lower_ratio = lower_range / total_range  # 30/60 = 0.5
    middle_ratio = middle_range / total_range  # 15/60 = 0.25
    upper_ratio = upper_range / total_range  # 15/60 = 0.25

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(14, 12), gridspec_kw={"height_ratios": [upper_ratio, middle_ratio, lower_ratio], "hspace": 0.05}
    )

    # Plot data on all three axes
    for ax in [ax1, ax2, ax3]:
        for i, p_value in enumerate(sorted(all_data.keys())):
            if p_value not in all_data:
                continue

            comp_times = []
            comp_stds = []
            comm_times = []
            comm_stds = []

            for resolution in resolutions:
                if resolution in all_data[p_value]:
                    data_point = all_data[p_value][resolution]
                    comp_times.append(data_point["comp_avg"])
                    comp_stds.append(data_point["comp_std"])
                    comm_times.append(data_point["comm_avg"])
                    comm_stds.append(data_point["comm_std"])
                else:
                    comp_times.append(0)
                    comp_stds.append(0)
                    comm_times.append(0)
                    comm_stds.append(0)

            # Plot computation bars with hatching pattern
            bars_comp = ax.bar(
                x + i * width,
                comp_times,
                width,
                label=f"P={p_value}" if ax == ax1 else "",
                yerr=comp_stds,
                capsize=5,
                color=colors[p_value],
                alpha=0.8,
                hatch="///",  # Forward slashes for computation
                edgecolor="black",
                linewidth=0.5,
            )

            # Plot communication on top (if any)
            if any(comm_times):
                bars_comm = ax.bar(
                    x + i * width,
                    comm_times,
                    width,
                    bottom=comp_times,
                    yerr=comm_stds,
                    capsize=5,
                    color="#808080",
                    alpha=0.8,
                    hatch="\\\\\\",  # Backslashes for communication
                    edgecolor="black",
                    linewidth=0.5,
                )

    # Set axis limits - show 0-30, 45-60, 90-105 (hide 30-45 and 60-90)
    ax1.set_ylim(90, 105)  # Upper part - shows P=1 2048×2048 (around 99.87s)
    ax2.set_ylim(45, 60)  # Middle part - shows intermediate values
    ax3.set_ylim(0, 30)  # Lower part - shows parallel configurations

    # Hide spines between subplots
    ax1.spines["bottom"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax2.spines["bottom"].set_visible(False)
    ax3.spines["top"].set_visible(False)

    ax1.xaxis.tick_top()
    ax1.tick_params(labeltop=False)
    ax2.tick_params(labeltop=False, labelbottom=False)
    ax3.xaxis.tick_bottom()

    # Add break symbols between ax1 and ax2
    d = 0.015  # size of diagonal lines
    kwargs = dict(transform=ax1.transAxes, color="k", clip_on=False)
    ax1.plot((-d, +d), (-d, +d), **kwargs)
    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)

    kwargs.update(transform=ax2.transAxes)
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    # Add break symbols between ax2 and ax3
    kwargs = dict(transform=ax2.transAxes, color="k", clip_on=False)
    ax2.plot((-d, +d), (-d, +d), **kwargs)
    ax2.plot((1 - d, 1 + d), (-d, +d), **kwargs)

    kwargs.update(transform=ax3.transAxes)
    ax3.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax3.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    # Customize axes
    ax3.set_xlabel("Height × Width", fontsize=16, fontweight="bold")
    fig.text(0.02, 0.5, "End-to-End Time (s)", fontsize=16, fontweight="bold", rotation=90, ha="center", va="center")
    ax1.set_title(
        "Communication and Computation Time Breakdown with Axis Break\nFLUX Model - Different Parallel Configurations (Hiding 30-45s, 60-90s)",
        fontsize=18,
        fontweight="bold",
        pad=20,
    )

    # Set x-axis
    ax3.set_xticks(x + width * 1.5)
    ax3.set_xticklabels(resolutions, fontsize=14)
    ax1.set_xticks(x + width * 1.5)
    ax1.set_xticklabels([])
    ax2.set_xticks(x + width * 1.5)
    ax2.set_xticklabels([])

    # Create custom legend with proper labels
    from matplotlib.patches import Patch

    legend_elements = []

    # Add hatched pattern for computation and communication
    legend_elements.append(Patch(facecolor="white", hatch="///", edgecolor="black", label="Computation"))
    legend_elements.append(
        Patch(facecolor="#808080", hatch="\\\\\\", alpha=0.8, edgecolor="black", label="Communication")
    )

    # Add P value labels
    for p_value in sorted(all_data.keys()):
        legend_elements.append(Patch(facecolor=colors[p_value], alpha=0.8, edgecolor="black", label=f"P={p_value}"))

    # Position legend outside the plot area to avoid interference
    fig.legend(
        handles=legend_elements,
        fontsize=11,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        fancybox=True,
        shadow=True,
        ncol=1,
    )

    # Add grids
    for ax in [ax1, ax2, ax3]:
        ax.grid(True, alpha=0.3, axis="y", linestyle="-", linewidth=0.5)
        ax.set_axisbelow(True)

    # Improve layout
    plt.tight_layout()
    return fig, (ax1, ax2, ax3)


if __name__ == "__main__":
    # Create comprehensive plot with all parallel configurations
    json_files_dict = {
        1: "log/FluidServe_logs/benchmark/comm_comp/flux/detailed_comm_comp_time_1.json",
        2: "log/FluidServe_logs/benchmark/comm_comp/flux/detailed_comm_comp_time_2.json",
        4: "log/FluidServe_logs/benchmark/comm_comp/flux/detailed_comm_comp_time_4.json",
        8: "log/FluidServe_logs/benchmark/comm_comp/flux/detailed_comm_comp_time_8.json",
    }

    # Generate original plot without axis break
    print("Creating original comprehensive comparison plot without axis break...")
    fig_original, ax_original = plot_multi_parallel_comparison_original(json_files_dict)
    plt.figure(fig_original.number)
    plt.savefig("flux_timing_error_bars_original.png", dpi=300, bbox_inches="tight")
    print("Original plot saved as 'flux_timing_error_bars_original.png'")

    # Generate improved plot with axis break
    print("\nCreating comprehensive comparison plot with axis break...")
    fig2, (ax1, ax2, ax3) = plot_multi_parallel_comparison(json_files_dict)
    plt.figure(fig2.number)
    plt.savefig("flux_timing_error_bars_axis_break.png", dpi=300, bbox_inches="tight")
    plt.savefig("flux_timing_error_bars_axis_break.pdf", dpi=300, bbox_inches="tight")
    print("Plot with axis break saved as 'flux_timing_error_bars_axis_break.png'")

    # Print summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS FOR ALL PARALLEL CONFIGURATIONS")
    print("=" * 80)

    for p_value, file_path in json_files_dict.items():
        with open(file_path, "r") as f:
            data = json.load(f)

        print(f"\nP={p_value} Configuration:")
        print("-" * 40)
        for entry in data:
            resolution = f"{entry['height']}×{entry['width']}"
            comp_time = entry["comp_time_avg"]
            comp_std = entry["comp_time_std"]
            comm_time = entry["comm_time_avg"]
            comm_std = entry["comm_time_std"]
            total_time = comp_time + comm_time

            print(
                f"  {resolution:>10}: {total_time:6.2f}s (±{comp_std:5.3f}) "
                f"[Comp: {comp_time:6.2f}s, Comm: {comm_time:6.3f}s]"
            )

    plt.show()
