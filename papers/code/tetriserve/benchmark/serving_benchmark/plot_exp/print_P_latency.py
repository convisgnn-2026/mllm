import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib import font_manager as fm

# Ensure Arial is registered with Matplotlib; fail if unavailable.
arial_candidates = [
    "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf",
    "/usr/share/fonts/truetype/msttcorefonts/arial.ttf",
    "/usr/share/fonts/truetype/msttcorefonts/Arial.TTF",
    "/usr/share/fonts/truetype/msttcorefonts/arial.TTF",
]
arial_path = next((p for p in arial_candidates if Path(p).exists()), None)
if arial_path:
    fm.fontManager.addfont(arial_path)
    print(f"Using Arial at: {arial_path}")
else:
    raise RuntimeError(
        "Arial font file not found. Ensure msttcorefonts (or Arial TTF) is installed "
        "and available at a known path, then rerun this script."
    )

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]

json_file = "tetriserve_benchmark/serving_benchmark/results/benchmark_flux_H100_tetris_dp_fake_engine/benchmark_results_all_slos.json"

with open(json_file, "r") as f:
    data = json.load(f)

schedule_logic_list = [
    "data_parallel",
    "fixed_sp_2",
    "fixed_sp_4",
    "fixed_sp_8",
    "per_resolution_schedule_1_1_2_8",
    "dyn_slo_schedule_global_dp",
]
shape_dist_list = ["evenly", "long_seqlen_high"]
slo_config_list = [
    "custom_1_1.0x_slo_H100_flux",
    "custom_1_1.1x_slo_H100_flux",
    "custom_1_1.2x_slo_H100_flux",
    "custom_1_1.3x_slo_H100_flux",
    "custom_1_1.4x_slo_H100_flux",
    "custom_1_1.5x_slo_H100_flux",
]

# Short names for better display
schedule_logic_short = {
    "data_parallel": "DP",
    "fixed_sp_2": "SP-2",
    "fixed_sp_4": "SP-4",
    "fixed_sp_8": "SP-8",
    "per_resolution_schedule_1_1_2_8": "PerRes",
    "dyn_slo_schedule_global_dp": "DynSLO",
}

slo_config_short = {
    "custom_1_1.0x_slo_H100_flux": "1.0x",
    "custom_1_1.1x_slo_H100_flux": "1.1x",
    "custom_1_1.2x_slo_H100_flux": "1.2x",
    "custom_1_1.3x_slo_H100_flux": "1.3x",
    "custom_1_1.4x_slo_H100_flux": "1.4x",
    "custom_1_1.5x_slo_H100_flux": "1.5x",
}

# Organize data for comparison
results = {}

for item in data:
    shape_dist = item["config"]["shape_dist"]
    slo_config = item["config"]["slo_config"]
    schedule_logic = item["config"]["schedule_logic"]

    key = (shape_dist, slo_config)
    if key not in results:
        results[key] = {}

    results[key][schedule_logic] = {
        "mean": item["stats"]["latency_mean"],
        "std": item["stats"]["latency_std"],
        "p50": item["stats"]["latency_p50"],
        "p90": item["stats"]["latency_p90"],
        "p95": item["stats"]["latency_p95"],
        "p99": item["stats"]["latency_p99"],
    }

# Print formatted comparison tables
print("=" * 120)
print("LATENCY COMPARISON ACROSS SCHEDULING STRATEGIES")
print("=" * 120)

for shape_dist in shape_dist_list:
    print(f"\n{'='*120}")
    print(f"SHAPE DISTRIBUTION: {shape_dist.upper()}")
    print(f"{'='*120}\n")

    for slo_config in slo_config_list:
        key = (shape_dist, slo_config)
        if key not in results:
            continue

        print(f"\nSLO Config: {slo_config_short[slo_config]}")
        print("-" * 120)

        # Table header
        header = f"{'Schedule Logic':<20} {'Mean':<12} {'Std':<12} {'P50':<12} {'P90':<12} {'P95':<12} {'P99':<12}"
        print(header)
        print("-" * 120)

        # Table rows
        for schedule_logic in schedule_logic_list:
            if schedule_logic in results[key]:
                stats = results[key][schedule_logic]
                row = (
                    f"{schedule_logic_short[schedule_logic]:<20} "
                    f"{stats['mean']:>10.2f}s  "
                    f"{stats['std']:>10.2f}s  "
                    f"{stats['p50']:>10.2f}s  "
                    f"{stats['p90']:>10.2f}s  "
                    f"{stats['p95']:>10.2f}s  "
                    f"{stats['p99']:>10.2f}s"
                )
                print(row)

        print()

# Generate visualizations
output_dir = Path("tetriserve_benchmark/serving_benchmark/results/benchmark_flux_H100_tetris_dp_fake_engine/plots")
output_dir.mkdir(exist_ok=True)

# Plot 1: Heatmaps for each shape_dist showing schedule_logic vs slo_config for P99 latency
for shape_dist in shape_dist_list:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Latency Comparison - {shape_dist}", fontsize=16, fontweight="bold")

    metrics = ["mean", "p90", "p99"]
    metric_names = ["Mean Latency", "P90 Latency", "P99 Latency"]

    for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        ax = axes[idx]

        # Create matrix for heatmap
        matrix = []
        for schedule_logic in schedule_logic_list:
            row = []
            for slo_config in slo_config_list:
                key = (shape_dist, slo_config)
                if key in results and schedule_logic in results[key]:
                    row.append(results[key][schedule_logic][metric])
                else:
                    row.append(np.nan)
            matrix.append(row)

        matrix = np.array(matrix)

        # Create heatmap
        im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn_r")

        # Set ticks and labels
        ax.set_xticks(np.arange(len(slo_config_list)))
        ax.set_yticks(np.arange(len(schedule_logic_list)))
        ax.set_xticklabels([slo_config_short[s] for s in slo_config_list], fontweight="bold")
        ax.set_yticklabels([schedule_logic_short[s] for s in schedule_logic_list], fontweight="bold")

        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Add values in cells
        for i in range(len(schedule_logic_list)):
            for j in range(len(slo_config_list)):
                if not np.isnan(matrix[i, j]):
                    text = ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center", color="black", fontsize=8)

        ax.set_title(metric_name)
        ax.set_xlabel("SLO Multiplier")
        ax.set_ylabel("Schedule Logic")

        # Add colorbar
        plt.colorbar(im, ax=ax, label="Latency (s)")

    plt.tight_layout()
    plt.savefig(output_dir / f"latency_heatmap_{shape_dist}.png", dpi=150, bbox_inches="tight")
    print(f"Saved heatmap: {output_dir / f'latency_heatmap_{shape_dist}.png'}")

# Plot 2: Line plots comparing schedule_logic for each shape_dist
for shape_dist in shape_dist_list:
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f"Schedule Logic Comparison - {shape_dist}", fontsize=16, fontweight="bold")

    metrics = [("mean", "Mean Latency"), ("p90", "P90 Latency"), ("p95", "P95 Latency"), ("p99", "P99 Latency")]

    for idx, (metric, metric_name) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]

        for schedule_logic in schedule_logic_list:
            latencies = []
            x_labels = []
            for slo_config in slo_config_list:
                key = (shape_dist, slo_config)
                if key in results and schedule_logic in results[key]:
                    latencies.append(results[key][schedule_logic][metric])
                    x_labels.append(slo_config_short[slo_config])

            if latencies:
                ax.plot(x_labels, latencies, marker="o", label=schedule_logic_short[schedule_logic], linewidth=2)

        ax.set_xlabel("SLO Multiplier", fontsize=11)
        ax.set_ylabel("Latency (s)", fontsize=11)
        ax.set_title(metric_name, fontsize=12)
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f"latency_lines_{shape_dist}.png", dpi=150, bbox_inches="tight")
    print(f"Saved line plot: {output_dir / f'latency_lines_{shape_dist}.png'}")

# Plot 3: Bar charts for specific SLO configs
for shape_dist in shape_dist_list:
    # Select a few representative SLO configs for cleaner visualization
    selected_slos = ["custom_1_1.0x_slo_H100_flux", "custom_1_1.2x_slo_H100_flux", "custom_1_1.5x_slo_H100_flux"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"P99 Latency Comparison - {shape_dist}", fontsize=16, fontweight="bold")

    for idx, slo_config in enumerate(selected_slos):
        ax = axes[idx]
        key = (shape_dist, slo_config)

        if key in results:
            schedules = []
            latencies = []

            for schedule_logic in schedule_logic_list:
                if schedule_logic in results[key]:
                    schedules.append(schedule_logic_short[schedule_logic])
                    latencies.append(results[key][schedule_logic]["p99"])

            bars = ax.bar(schedules, latencies, color=plt.cm.Set3(np.arange(len(schedules))))

            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0, height, f"{height:.1f}s", ha="center", va="bottom", fontsize=9
                )

            ax.set_xlabel("Schedule Logic", fontsize=11)
            ax.set_ylabel("P99 Latency (s)", fontsize=11)
            ax.set_title(f"SLO: {slo_config_short[slo_config]}", fontsize=12)
            ax.tick_params(axis="x", rotation=45)
            ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_dir / f"latency_bars_{shape_dist}.png", dpi=150, bbox_inches="tight")
    print(f"Saved bar chart: {output_dir / f'latency_bars_{shape_dist}.png'}")

print(f"\n{'='*120}")
print(f"All plots saved to: {output_dir}")
print(f"{'='*120}")
