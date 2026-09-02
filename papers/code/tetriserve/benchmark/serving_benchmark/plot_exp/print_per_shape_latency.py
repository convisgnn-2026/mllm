import json
import matplotlib.pyplot as plt
from cycler import cycler
import numpy as np
from pathlib import Path
from collections import defaultdict
from matplotlib import font_manager as fm
import pandas as pd

# Ensure Arial is registered with Matplotlib; fail if unavailable.
# Register Arial Regular
arial_candidates = [
    "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf",
    "/usr/share/fonts/truetype/msttcorefonts/arial.ttf",
    "/usr/share/fonts/truetype/msttcorefonts/Arial.TTF",
    "/usr/share/fonts/truetype/msttcorefonts/arial.TTF",
]
arial_path = next((p for p in arial_candidates if Path(p).exists()), None)
if not arial_path:
    raise RuntimeError(
        "Arial font file not found. Ensure msttcorefonts (or Arial TTF) is installed "
        "and available at a known path, then rerun this script."
    )

fm.fontManager.addfont(arial_path)
print(f"Using Arial Regular at: {arial_path}")

# Register Arial Bold (essential for bold tick labels!)
arial_bold_candidates = [
    "/usr/share/fonts/truetype/msttcorefonts/Arial_Bold.ttf",
    "/usr/share/fonts/truetype/msttcorefonts/arialbd.ttf",
    "/usr/share/fonts/truetype/msttcorefonts/ARIALBD.TTF",
]
arial_bold_path = next((p for p in arial_bold_candidates if Path(p).exists()), None)
if arial_bold_path:
    fm.fontManager.addfont(arial_bold_path)
    print(f"Using Arial Bold at: {arial_bold_path}")
else:
    print("WARNING: Arial Bold not found. Bold text may not render properly.")

# Register Arial Italic (optional but good to have)
arial_italic_candidates = [
    "/usr/share/fonts/truetype/msttcorefonts/Arial_Italic.ttf",
    "/usr/share/fonts/truetype/msttcorefonts/ariali.ttf",
]
arial_italic_path = next((p for p in arial_italic_candidates if Path(p).exists()), None)
if arial_italic_path:
    fm.fontManager.addfont(arial_italic_path)
    print(f"Using Arial Italic at: {arial_italic_path}")

# Register Arial Bold Italic (optional)
arial_bold_italic_candidates = [
    "/usr/share/fonts/truetype/msttcorefonts/Arial_Bold_Italic.ttf",
    "/usr/share/fonts/truetype/msttcorefonts/arialbi.ttf",
]
arial_bold_italic_path = next((p for p in arial_bold_italic_candidates if Path(p).exists()), None)
if arial_bold_italic_path:
    fm.fontManager.addfont(arial_bold_italic_path)
    print(f"Using Arial Bold Italic at: {arial_bold_italic_path}")

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]

json_file = "tetriserve_benchmark/serving_benchmark/results/benchmark_flux_H100_tetris_dp_fake_engine/benchmark_results_all_slos.json"
# Generate visualizations
# output_dir = Path("tetriserve_benchmark/serving_benchmark/results/benchmark_flux_H100_tetris_dp_fake_engine/plots/per_shape")
# output_dir.mkdir(exist_ok=True, parents=True)

# json_file = (
#     "tetriserve_benchmark/serving_benchmark/results/benchmark_flux_H100_tetris_dp_real/benchmark_results_all_slos.json"
# )
# Generate visualizations
# output_dir = Path("tetriserve_benchmark/serving_benchmark/results/benchmark_flux_H100_tetris_dp_real/plots/per_shape")
output_dir = Path(
    "tetriserve_benchmark/serving_benchmark/results/benchmark_flux_H100_tetris_dp_fake_engine/plots/per_shape"
)
output_dir.mkdir(exist_ok=True, parents=True)


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

# Resolution shapes we care about
resolution_shapes = ["256x256", "512x512", "1024x1024", "2048x2048"]

# Short names for better display
schedule_logic_short = {
    "data_parallel": "SP=1",
    "fixed_sp_2": "SP=2",
    "fixed_sp_4": "SP=4",
    "fixed_sp_8": "SP=8",
    "per_resolution_schedule_1_1_2_8": "RSSP",
    "dyn_slo_schedule_global_dp": "TetriServe",
}

schedule_logic_color = {
    "data_parallel": "#87CEEB",  # xDiT (SP=1)
    "fixed_sp_2": "#A9A9A9",  # xDiT (SP=2)
    "fixed_sp_4": "#696969",  # xDiT (SP=4)
    "fixed_sp_8": "#000000",  # xDiT (SP=8)
    "per_resolution_schedule_1_1_2_8": "#4B0082",  # RSSP / Static SP
    "dyn_slo_schedule_global_dp": "#FF8C00",  # TetriServe (ours)
}

# Apply consistent color cycle following schedule order
plt.rcParams["axes.prop_cycle"] = cycler(color=[schedule_logic_color[s] for s in schedule_logic_list])

slo_config_short = {
    "custom_1_1.0x_slo_H100_flux": "1.0x",
    "custom_1_1.1x_slo_H100_flux": "1.1x",
    "custom_1_1.2x_slo_H100_flux": "1.2x",
    "custom_1_1.3x_slo_H100_flux": "1.3x",
    "custom_1_1.4x_slo_H100_flux": "1.4x",
    "custom_1_1.5x_slo_H100_flux": "1.5x",
}

shape_dist_display = {"evenly": "Uniform Mixes", "long_seqlen_high": "Skewed Mixes"}


def compute_percentile(latencies, p):
    """Compute percentile for a list of latencies"""
    if not latencies:
        return np.nan
    return np.percentile(latencies, p)


# Organize data by shape
# Structure: results[shape_dist][slo_config][schedule_logic][resolution] = list of latencies
#            results_slo_met[...] = latencies where slo_met is True (per schedule_logic)
#            results_slo_union[shape_dist][slo_config][resolution] = latencies where slo_met is True (union of all schedules)
#            results_by_req[shape_dist][slo_config][schedule_logic][resolution][req_idx] = latency (all requests)
#            union_req_ids[shape_dist][slo_config][resolution] = set of req_idx that met SLO in any schedule
#            slo_targets[shape_dist][slo_config][schedule_logic][resolution] = slo value
results = {}
results_slo_met = {}
results_slo_union = {}
results_by_req = {}
union_req_ids = {}
slo_targets = {}

for item in data:
    shape_dist = item["config"]["shape_dist"]
    slo_config = item["config"]["slo_config"]
    schedule_logic = item["config"]["schedule_logic"]

    if shape_dist not in results:
        results[shape_dist] = {}
        results_slo_met[shape_dist] = {}
        results_slo_union[shape_dist] = {}
        results_by_req[shape_dist] = {}
        union_req_ids[shape_dist] = {}
        slo_targets[shape_dist] = {}
    if slo_config not in results[shape_dist]:
        results[shape_dist][slo_config] = {}
        results_slo_met[shape_dist][slo_config] = {}
        results_slo_union[shape_dist][slo_config] = defaultdict(list)
        results_by_req[shape_dist][slo_config] = {}
        union_req_ids[shape_dist][slo_config] = defaultdict(set)
        slo_targets[shape_dist][slo_config] = {}
    if schedule_logic not in results[shape_dist][slo_config]:
        results[shape_dist][slo_config][schedule_logic] = defaultdict(list)
        results_slo_met[shape_dist][slo_config][schedule_logic] = defaultdict(list)
        results_by_req[shape_dist][slo_config][schedule_logic] = defaultdict(dict)
        slo_targets[shape_dist][slo_config][schedule_logic] = {}

    # Extract per-request info and organize by shape
    for req in item["stats"]["per_request_info"]:
        resolution = f"{req['height']}x{req['width']}"
        latency = req["latency"]
        req_idx = req["req_idx"]
        results[shape_dist][slo_config][schedule_logic][resolution].append(latency)
        results_by_req[shape_dist][slo_config][schedule_logic][resolution][req_idx] = latency
        if req.get("slo_met"):
            results_slo_met[shape_dist][slo_config][schedule_logic][resolution].append(latency)
            results_slo_union[shape_dist][slo_config][resolution].append(latency)
            union_req_ids[shape_dist][slo_config][resolution].add(req_idx)
        if resolution not in slo_targets[shape_dist][slo_config][schedule_logic]:
            slo_val = req.get("slo")
            if slo_val is not None:
                slo_targets[shape_dist][slo_config][schedule_logic][resolution] = slo_val * 1.07

# Compute statistics for each shape
# Structure: stats[shape_dist][slo_config][schedule_logic][resolution] = {mean, p50, p90, p95, p99}
stats = {}

for shape_dist in results:
    stats[shape_dist] = {}
    for slo_config in results[shape_dist]:
        stats[shape_dist][slo_config] = {}
        for schedule_logic in results[shape_dist][slo_config]:
            stats[shape_dist][slo_config][schedule_logic] = {}
            for resolution in results[shape_dist][slo_config][schedule_logic]:
                latencies = results[shape_dist][slo_config][schedule_logic][resolution]
                stats[shape_dist][slo_config][schedule_logic][resolution] = {
                    "mean": np.mean(latencies),
                    "std": np.std(latencies),
                    "p50": compute_percentile(latencies, 50),
                    "p90": compute_percentile(latencies, 90),
                    "p95": compute_percentile(latencies, 95),
                    "p99": compute_percentile(latencies, 99),
                    "count": len(latencies),
                }

# Print formatted comparison tables
print("=" * 140)
print("PER-SHAPE LATENCY COMPARISON ACROSS SCHEDULING STRATEGIES")
print("=" * 140)

for shape_dist in shape_dist_list:
    if shape_dist not in stats:
        continue

    print(f"\n{'='*140}")
    print(f"SHAPE DISTRIBUTION: {shape_dist_display[shape_dist].upper()}")
    print(f"{'='*140}\n")

    for slo_config in slo_config_list:
        if slo_config not in stats[shape_dist]:
            continue

        print(f"\nSLO Config: {slo_config_short[slo_config]}")
        print("=" * 140)

        for resolution in resolution_shapes:
            print(f"\n  Resolution: {resolution}")
            print("  " + "-" * 136)

            # Table header
            header = f"  {'Schedule Logic':<20} {'Count':<8} {'Mean':<12} {'Std':<12} {'P50':<12} {'P90':<12} {'P95':<12} {'P99':<12}"
            print(header)
            print("  " + "-" * 136)

            # Table rows
            for schedule_logic in schedule_logic_list:
                if schedule_logic in stats[shape_dist][slo_config]:
                    if resolution in stats[shape_dist][slo_config][schedule_logic]:
                        s = stats[shape_dist][slo_config][schedule_logic][resolution]
                        row = (
                            f"  {schedule_logic_short[schedule_logic]:<20} "
                            f"{s['count']:<8} "
                            f"{s['mean']:>10.2f}s  "
                            f"{s['std']:>10.2f}s  "
                            f"{s['p50']:>10.2f}s  "
                            f"{s['p90']:>10.2f}s  "
                            f"{s['p95']:>10.2f}s  "
                            f"{s['p99']:>10.2f}s"
                        )
                        print(row)

            print()

# # Plot 0: Overall percentiles (p95, p99, p99.9) across all shapes (print only)
# for shape_dist in shape_dist_list:
#     if shape_dist not in results:
#         continue

#     for slo_config in slo_config_list:
#         if slo_config not in results[shape_dist]:
#             continue

#         print(f"\nOverall percentiles - {shape_dist_display[shape_dist]}, SLO {slo_config_short[slo_config]}")
#         header = f"{'Schedule Logic':<20} {'P95 (s)':>12} {'P99 (s)':>12} {'P99.9 (s)':>12}"
#         print(header)
#         print("-" * len(header))

#         for schedule_logic in schedule_logic_list:
#             res_dict = results[shape_dist][slo_config].get(schedule_logic, {})
#             latencies = []
#             for res_vals in res_dict.values():
#                 latencies.extend(res_vals)

#             if latencies:
#                 p95 = np.percentile(latencies, 95)
#                 p99 = np.percentile(latencies, 99)
#                 p999 = np.percentile(latencies, 99.9)
#             else:
#                 p95 = p99 = p999 = np.nan

#             row = f"{schedule_logic_short[schedule_logic]:<20} {p95:>12.2f} {p99:>12.2f} {p999:>12.2f}"
#             print(row)


# # Plot 1: Heatmaps for each (shape_dist, resolution) showing schedule_logic vs slo_config
# # Create separate heatmaps for each metric
# metrics_to_plot = ["mean", "p90", "p95", "p99"]
# metric_names = ["Mean Latency", "P90 Latency", "P95 Latency", "P99 Latency"]

# for shape_dist in shape_dist_list:
#     if shape_dist not in stats:
#         continue

#     for metric, metric_name in zip(metrics_to_plot, metric_names):
#         fig, axes = plt.subplots(2, 2, figsize=(20, 16))
#         # fig.suptitle(f'{metric_name} by Resolution - {shape_dist_display[shape_dist]}', fontsize=24, fontweight='bold')

#         for idx, resolution in enumerate(resolution_shapes):
#             ax = axes[idx // 2, idx % 2]

#             # Create matrix for heatmap
#             matrix = []
#             for schedule_logic in schedule_logic_list:
#                 row = []
#                 for slo_config in slo_config_list:
#                     if (
#                         slo_config in stats[shape_dist]
#                         and schedule_logic in stats[shape_dist][slo_config]
#                         and resolution in stats[shape_dist][slo_config][schedule_logic]
#                     ):
#                         row.append(stats[shape_dist][slo_config][schedule_logic][resolution][metric])
#                     else:
#                         row.append(np.nan)
#                 matrix.append(row)

#             matrix = np.array(matrix)

#             # Create heatmap
#             im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn_r")

#             # Set ticks and labels
#             ax.set_xticks(np.arange(len(slo_config_list)))
#             ax.set_yticks(np.arange(len(schedule_logic_list)))
#             ax.set_xticklabels([slo_config_short[s] for s in slo_config_list], fontsize=25)
#             ax.set_yticklabels([schedule_logic_short[s] for s in schedule_logic_list], fontsize=25)

#             # Rotate x labels
#             plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

#             # Add values in cells
#             for i in range(len(schedule_logic_list)):
#                 for j in range(len(slo_config_list)):
#                     if not np.isnan(matrix[i, j]):
#                         text = ax.text(
#                             j, i, f"{matrix[i, j]:.1f}", ha="center", va="center", color="black", fontsize=25
#                         )

#             ax.set_title(f"{resolution}", fontsize=35, fontweight="bold")
#             ax.set_xlabel("SLO Scale", fontsize=30)
#             # Remove y-axis label
#             ax.set_ylabel("")

#             # Add colorbar
#             plt.colorbar(im, ax=ax, label=f"{metric_name} (s)")

#         plt.tight_layout()
#         plt.savefig(output_dir / f"per_shape_heatmap_{metric}_{shape_dist}.png", dpi=150, bbox_inches="tight")
#         plt.savefig(output_dir / f"per_shape_heatmap_{metric}_{shape_dist}.pdf", dpi=300, bbox_inches="tight")
#         print(f"Saved per-shape {metric} heatmap: {output_dir / f'per_shape_heatmap_{metric}_{shape_dist}.png'}")

# # Plot 2: Bar charts with 4 sub-plots (one per resolution) showing schedule logics across SLO configs
# for shape_dist in shape_dist_list:
#     if shape_dist not in stats:
#         continue

#     for metric, metric_name in zip(metrics_to_plot, metric_names):
#         fig, axes = plt.subplots(2, 2, figsize=(20, 16))
#         fig.suptitle(f'{metric_name} by Resolution', fontsize=40, fontweight='bold', y=0.998)

#         # Create handles for legend (will be collected from first subplot)
#         handles, labels = None, None

#         for idx, resolution in enumerate(resolution_shapes):
#             ax = axes[idx // 2, idx % 2]

#             # Number of SLO configs and schedule logics
#             x = np.arange(len(slo_config_list))
#             width = 0.13  # Width of each bar

#             # Plot bars for each schedule logic
#             for i, schedule_logic in enumerate(schedule_logic_list):
#                 latencies = []
#                 for slo_config in slo_config_list:
#                     if (
#                         slo_config in stats[shape_dist]
#                         and schedule_logic in stats[shape_dist][slo_config]
#                         and resolution in stats[shape_dist][slo_config][schedule_logic]
#                     ):
#                         latencies.append(stats[shape_dist][slo_config][schedule_logic][resolution][metric])
#                     else:
#                         latencies.append(0)  # Use 0 for missing data

#                 # Offset for grouped bars
#                 offset = (i - len(schedule_logic_list) / 2 + 0.5) * width
#                 bars = ax.bar(x + offset, latencies, width,
#                              label=schedule_logic_short[schedule_logic],
#                              alpha=0.8)

#             ax.set_xlabel('SLO Scale', fontsize=30)
#             ax.set_ylabel(f'{metric_name} (s)', fontsize=30)
#             ax.set_title(f'{resolution}', fontsize=35, fontweight='bold')
#             ax.set_xticks(x)
#             ax.set_xticklabels([slo_config_short[s] for s in slo_config_list], fontsize=25)
#             ax.tick_params(axis='y', labelsize=25)
#             ax.grid(True, alpha=0.3, axis='y')

#             # Collect legend handles from first subplot
#             if idx == 0:
#                 handles, labels = ax.get_legend_handles_labels()

#         # Add a single legend at the figure level
#         if handles and labels:
#             fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.97),
#                       ncol=6, fontsize=22, frameon=True, fancybox=True, shadow=True)

#         plt.tight_layout(rect=[0, 0, 1, 0.90])  # Leave more space for title and legend at top
#         plt.savefig(output_dir / f"per_shape_bargraph_{metric}_{shape_dist}.png", dpi=150, bbox_inches="tight")
#         plt.savefig(output_dir / f"per_shape_bargraph_{metric}_{shape_dist}.pdf", dpi=300, bbox_inches="tight")
#         print(f"Saved per-shape {metric} bar graph: {output_dir / f'per_shape_bargraph_{metric}_{shape_dist}.png'}")

# # Plot 3: Overall P99 Latency (without per-shape calculation) - 2 subplots side by side
# overall_stats = {}
# for item in data:
#     shape_dist = item["config"]["shape_dist"]
#     slo_config = item["config"]["slo_config"]
#     schedule_logic = item["config"]["schedule_logic"]

#     if shape_dist not in overall_stats:
#         overall_stats[shape_dist] = {}
#     if slo_config not in overall_stats[shape_dist]:
#         overall_stats[shape_dist][slo_config] = {}

#     overall_stats[shape_dist][slo_config][schedule_logic] = {
#         'p99_latency': item['stats']['latency_p99'],
#         'mean_latency': item['stats']['latency_mean'],
#         'throughput': item['stats']['throughput']
#     }

# # Plot overall P99 latency
# fig, axes = plt.subplots(1, 2, figsize=(20, 8))
# fig.suptitle('Overall P99 Latency', fontsize=40, fontweight='bold', y=1.0)

# for idx, shape_dist in enumerate(shape_dist_list):
#     if shape_dist not in overall_stats:
#         continue

#     ax = axes[idx]

#     # Number of SLO configs and schedule logics
#     x = np.arange(len(slo_config_list))
#     width = 0.13

#     # Plot bars for each schedule logic
#     for i, schedule_logic in enumerate(schedule_logic_list):
#         latencies = []
#         for slo_config in slo_config_list:
#             if (slo_config in overall_stats[shape_dist] and
#                 schedule_logic in overall_stats[shape_dist][slo_config]):
#                 latencies.append(overall_stats[shape_dist][slo_config][schedule_logic]['p99_latency'])
#             else:
#                 latencies.append(0)

#         offset = (i - len(schedule_logic_list) / 2 + 0.5) * width
#         bars = ax.bar(x + offset, latencies, width,
#                      label=schedule_logic_short[schedule_logic],
#                      alpha=0.8)

#     ax.set_xlabel('SLO Scale', fontsize=30)
#     ax.set_ylabel('P99 Latency (s)', fontsize=30)
#     ax.set_title(f'{shape_dist_display[shape_dist]}', fontsize=35, fontweight='bold')
#     ax.set_xticks(x)
#     ax.set_xticklabels([slo_config_short[s] for s in slo_config_list], fontsize=25)
#     ax.tick_params(axis='y', labelsize=25)
#     ax.grid(True, alpha=0.3, axis='y')

#     # Collect legend handles from first subplot
#     if idx == 0:
#         handles, labels = ax.get_legend_handles_labels()

# # Add a single legend at the figure level
# if handles and labels:
#     fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.94),
#               ncol=6, fontsize=22, frameon=True, fancybox=True, shadow=True)

# plt.tight_layout(rect=[0, 0, 1, 0.86])
# plt.savefig(output_dir / f"overall_p99_latency.png", dpi=150, bbox_inches="tight")
# plt.savefig(output_dir / f"overall_p99_latency.pdf", dpi=300, bbox_inches="tight")
# print(f"Saved overall P99 latency bar graph: {output_dir / 'overall_p99_latency.png'}")

# # Plot overall mean latency
# fig, axes = plt.subplots(1, 2, figsize=(20, 8))
# fig.suptitle('Overall Mean Latency', fontsize=40, fontweight='bold', y=1.0)

# for idx, shape_dist in enumerate(shape_dist_list):
#     if shape_dist not in overall_stats:
#         continue

#     ax = axes[idx]

#     # Number of SLO configs and schedule logics
#     x = np.arange(len(slo_config_list))
#     width = 0.13

#     # Plot bars for each schedule logic
#     for i, schedule_logic in enumerate(schedule_logic_list):
#         latencies = []
#         for slo_config in slo_config_list:
#             if (slo_config in overall_stats[shape_dist] and
#                 schedule_logic in overall_stats[shape_dist][slo_config]):
#                 latencies.append(overall_stats[shape_dist][slo_config][schedule_logic]['mean_latency'])
#             else:
#                 latencies.append(0)

#         offset = (i - len(schedule_logic_list) / 2 + 0.5) * width
#         bars = ax.bar(x + offset, latencies, width,
#                      label=schedule_logic_short[schedule_logic],
#                      alpha=0.8)

#     ax.set_xlabel('SLO Scale', fontsize=30)
#     ax.set_ylabel('Mean Latency (s)', fontsize=30)
#     ax.set_title(f'{shape_dist_display[shape_dist]}', fontsize=35, fontweight='bold')
#     ax.set_xticks(x)
#     ax.set_xticklabels([slo_config_short[s] for s in slo_config_list], fontsize=25)
#     ax.tick_params(axis='y', labelsize=25)
#     ax.grid(True, alpha=0.3, axis='y')

#     # Collect legend handles from first subplot
#     if idx == 0:
#         handles, labels = ax.get_legend_handles_labels()

# # Add a single legend at the figure level
# if handles and labels:
#     fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.94),
#               ncol=6, fontsize=22, frameon=True, fancybox=True, shadow=True)

# plt.tight_layout(rect=[0, 0, 1, 0.86])
# plt.savefig(output_dir / f"overall_mean_latency.png", dpi=150, bbox_inches="tight")
# plt.savefig(output_dir / f"overall_mean_latency.pdf", dpi=300, bbox_inches="tight")
# print(f"Saved overall mean latency bar graph: {output_dir / 'overall_mean_latency.png'}")

# # Plot overall throughput (TPT)
# fig, axes = plt.subplots(1, 2, figsize=(20, 8))
# fig.suptitle('Overall Throughput', fontsize=40, fontweight='bold', y=1.0)

# for idx, shape_dist in enumerate(shape_dist_list):
#     if shape_dist not in overall_stats:
#         continue

#     ax = axes[idx]

#     # Number of SLO configs and schedule logics
#     x = np.arange(len(slo_config_list))
#     width = 0.13

#     # Plot bars for each schedule logic
#     for i, schedule_logic in enumerate(schedule_logic_list):
#         throughputs = []
#         for slo_config in slo_config_list:
#             if (slo_config in overall_stats[shape_dist] and
#                 schedule_logic in overall_stats[shape_dist][slo_config]):
#                 throughputs.append(overall_stats[shape_dist][slo_config][schedule_logic]['throughput'])
#             else:
#                 throughputs.append(0)

#         offset = (i - len(schedule_logic_list) / 2 + 0.5) * width
#         bars = ax.bar(x + offset, throughputs, width,
#                      label=schedule_logic_short[schedule_logic],
#                      alpha=0.8)

#     ax.set_xlabel('SLO Scale', fontsize=30)
#     ax.set_ylabel('Throughput (req/s)', fontsize=30)
#     ax.set_title(f'{shape_dist_display[shape_dist]}', fontsize=35, fontweight='bold')
#     ax.set_xticks(x)
#     ax.set_xticklabels([slo_config_short[s] for s in slo_config_list], fontsize=25)
#     ax.tick_params(axis='y', labelsize=25)
#     ax.grid(True, alpha=0.3, axis='y')

#     # Collect legend handles from first subplot
#     if idx == 0:
#         handles, labels = ax.get_legend_handles_labels()

# # Add a single legend at the figure level
# if handles and labels:
#     fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.94),
#               ncol=6, fontsize=22, frameon=True, fancybox=True, shadow=True)

# plt.tight_layout(rect=[0, 0, 1, 0.86])
# plt.savefig(output_dir / f"overall_throughput.png", dpi=150, bbox_inches="tight")
# plt.savefig(output_dir / f"overall_throughput.pdf", dpi=300, bbox_inches="tight")
# print(f"Saved overall throughput bar graph: {output_dir / 'overall_throughput.png'}")


# # Plot 4: Per-shape CDF of latency
# # One figure per (shape_dist, slo_config); 4 subplots (resolutions), lines = schedule logic
# for shape_dist in shape_dist_list:
#     if shape_dist not in results:
#         continue

#     for slo_config in slo_config_list:
#         if slo_config not in results[shape_dist]:
#             continue

#         fig, axes = plt.subplots(2, 2, figsize=(20, 16))
#         fig.suptitle(
#             f"CDF of Latency - {shape_dist_display[shape_dist]}, SLO {slo_config_short[slo_config]}",
#             fontsize=40,
#             fontweight="bold",
#             y=0.998,
#         )

#         handles, labels = None, None

#         for idx, resolution in enumerate(resolution_shapes):
#             ax = axes[idx // 2, idx % 2]
#             slo_line_added = False

#             for schedule_logic in schedule_logic_list:
#                 latencies = (
#                     results[shape_dist][slo_config]
#                     .get(schedule_logic, {})
#                     .get(resolution, [])
#                 )
#                 if not latencies:
#                     continue

#                 lat_sorted = np.sort(latencies)
#                 y = np.arange(1, len(lat_sorted) + 1) / len(lat_sorted)
#                 ax.plot(
#                     lat_sorted,
#                     y,
#                     label=schedule_logic_short[schedule_logic],
#                     linewidth=2,
#                 )

#             # Add SLO reference line if available for this resolution
#             slo_val = None
#             for schedule_logic in schedule_logic_list:
#                 slo_val = (
#                     slo_targets[shape_dist][slo_config]
#                     .get(schedule_logic, {})
#                     .get(resolution)
#                 )
#                 if slo_val is not None:
#                     break

#             if slo_val is not None:
#                 ax.axvline(
#                     slo_val,
#                     color="k",
#                     linestyle="--",
#                     linewidth=2,
#                     label="SLO" if not slo_line_added else None,
#                 )
#                 slo_line_added = True

#             ax.set_title(f"{resolution}", fontsize=35, fontweight="bold")
#             ax.set_xlabel("Latency (s)", fontsize=30)
#             ax.set_ylabel("CDF", fontsize=30)
#             ax.tick_params(axis="both", labelsize=25)
#             ax.grid(True, alpha=0.3)
#             ax.set_ylim(0.2, 1.01)

#             if handles is None:
#                 handles, labels = ax.get_legend_handles_labels()

#         if handles and labels:
#             fig.legend(
#                 handles,
#                 labels,
#                 loc="upper center",
#                 bbox_to_anchor=(0.5, 0.97),
#                 ncol=6,
#                 fontsize=22,
#                 frameon=True,
#                 fancybox=True,
#                 shadow=True,
#             )

#         plt.tight_layout(rect=[0, 0, 1, 0.90])
#         plt.savefig(
#             output_dir / f"per_shape_cdf_{shape_dist}_{slo_config}.png",
#             dpi=150,
#             bbox_inches="tight",
#         )
#         plt.savefig(
#             output_dir / f"per_shape_cdf_{shape_dist}_{slo_config}.pdf",
#             dpi=300,
#             bbox_inches="tight",
#         )
#         print(
#             f"Saved per-shape CDF: {output_dir / f'per_shape_cdf_{shape_dist}_{slo_config}.png'}"
#         )

# # Plot 5: Per-shape CDF of latency (SLO-met only)
# # One figure per (shape_dist, slo_config); 4 subplots (resolutions), lines = schedule logic
# for shape_dist in shape_dist_list:
#     if shape_dist not in results_slo_met:
#         continue

#     for slo_config in slo_config_list:
#         if slo_config not in results_slo_met[shape_dist]:
#             continue

#         fig, axes = plt.subplots(2, 2, figsize=(20, 16))
#         fig.suptitle(
#             f"CDF of Latency (SLO-met) - {shape_dist_display[shape_dist]}, SLO {slo_config_short[slo_config]}",
#             fontsize=40,
#             fontweight="bold",
#             y=0.998,
#         )

#         handles, labels = None, None

#         for idx, resolution in enumerate(resolution_shapes):
#             ax = axes[idx // 2, idx % 2]
#             slo_line_added = False

#             for schedule_logic in schedule_logic_list:
#                 latencies = (
#                     results_slo_met[shape_dist][slo_config]
#                     .get(schedule_logic, {})
#                     .get(resolution, [])
#                 )
#                 if not latencies:
#                     continue

#                 lat_sorted = np.sort(latencies)
#                 y = np.arange(1, len(lat_sorted) + 1) / len(lat_sorted)
#                 ax.plot(
#                     lat_sorted,
#                     y,
#                     label=schedule_logic_short[schedule_logic],
#                     linewidth=2,
#                 )

#             # Add SLO reference line if available for this resolution
#             slo_val = None
#             for schedule_logic in schedule_logic_list:
#                 slo_val = (
#                     slo_targets[shape_dist][slo_config]
#                     .get(schedule_logic, {})
#                     .get(resolution)
#                 )
#                 if slo_val is not None:
#                     break

#             if slo_val is not None:
#                 ax.axvline(
#                     slo_val,
#                     color="k",
#                     linestyle="--",
#                     linewidth=2,
#                     label="SLO" if not slo_line_added else None,
#                 )
#                 slo_line_added = True

#             ax.set_title(f"{resolution}", fontsize=35, fontweight="bold")
#             ax.set_xlabel("Latency (s)", fontsize=30)
#             ax.set_ylabel("CDF", fontsize=30)
#             ax.tick_params(axis="both", labelsize=25)
#             ax.grid(True, alpha=0.3)
#             ax.set_ylim(0.2, 1.01)

#             if handles is None:
#                 handles, labels = ax.get_legend_handles_labels()

#         if handles and labels:
#             fig.legend(
#                 handles,
#                 labels,
#                 loc="upper center",
#                 bbox_to_anchor=(0.5, 0.97),
#                 ncol=6,
#                 fontsize=22,
#                 frameon=True,
#                 fancybox=True,
#                 shadow=True,
#             )

#         plt.tight_layout(rect=[0, 0, 1, 0.90])
#         plt.savefig(
#             output_dir / f"per_shape_cdf_slo_met_{shape_dist}_{slo_config}.png",
#             dpi=150,
#             bbox_inches="tight",
#         )
#         plt.savefig(
#             output_dir / f"per_shape_cdf_slo_met_{shape_dist}_{slo_config}.pdf",
#             dpi=300,
#             bbox_inches="tight",
#         )
#         print(
#             f"Saved per-shape SLO-met CDF: {output_dir / f'per_shape_cdf_slo_met_{shape_dist}_{slo_config}.png'}"
#         )

# # Plot 5c: Per-shape CDF of latency (per-schedule, filtered to union SLO-met requests)
# # One figure per (shape_dist, slo_config); 4 subplots (resolutions), lines = schedule logic but only requests that met SLO in any schedule
# for shape_dist in shape_dist_list:
#     if shape_dist not in union_req_ids:
#         continue

#     for slo_config in slo_config_list:
#         if slo_config not in union_req_ids[shape_dist]:
#             continue

#         fig, axes = plt.subplots(2, 2, figsize=(20, 16))
#         fig.suptitle(
#             f"CDF of {shape_dist_display[shape_dist]}, SLO {slo_config_short[slo_config]}",
#             fontsize=40,
#             fontweight="bold",
#             y=0.998,
#         )

#         handles, labels = None, None

#         for idx, resolution in enumerate(resolution_shapes):
#             ax = axes[idx // 2, idx % 2]
#             req_ids_union = union_req_ids[shape_dist][slo_config].get(resolution, set())

#             for schedule_logic in schedule_logic_list:
#                 req_map = (
#                     results_by_req[shape_dist][slo_config]
#                     .get(schedule_logic, {})
#                     .get(resolution, {})
#                 )
#                 latencies = [lat for rid, lat in req_map.items() if rid in req_ids_union]
#                 if not latencies:
#                     continue

#                 lat_sorted = np.sort(latencies)
#                 y = np.arange(1, len(lat_sorted) + 1) / len(lat_sorted)
#                 ax.plot(
#                     lat_sorted,
#                     y,
#                     label=schedule_logic_short[schedule_logic],
#                     linewidth=2,
#                 )

#             ax.set_title(f"{resolution}", fontsize=35, fontweight="bold")
#             ax.set_xlabel("Latency (s)", fontsize=30)
#             ax.set_ylabel("CDF", fontsize=30)
#             ax.tick_params(axis="both", labelsize=25)
#             ax.grid(True, alpha=0.3)
#             ax.set_ylim(0.2, 1.01)

#             if handles is None:
#                 handles, labels = ax.get_legend_handles_labels()

#         if handles and labels:
#             fig.legend(
#                 handles,
#                 labels,
#                 loc="upper center",
#                 bbox_to_anchor=(0.5, 0.97),
#                 ncol=6,
#                 fontsize=22,
#                 frameon=True,
#                 fancybox=True,
#                 shadow=True,
#             )

#         plt.tight_layout(rect=[0, 0, 1, 0.90])
#         plt.savefig(
#             output_dir / f"per_shape_cdf_per_sched_union_filtered_{shape_dist}_{slo_config}.png",
#             dpi=150,
#             bbox_inches="tight",
#         )
#         plt.savefig(
#             output_dir / f"per_shape_cdf_per_sched_union_filtered_{shape_dist}_{slo_config}.pdf",
#             dpi=300,
#             bbox_inches="tight",
#         )
#         print(
#             f"Saved per-shape per-schedule union-filtered CDF: {output_dir / f'per_shape_cdf_per_sched_union_filtered_{shape_dist}_{slo_config}.png'}"
#         )

# Plot 5d: Overall CDF (per-schedule, union SLO-met) across all resolutions
for shape_dist in shape_dist_list:
    if shape_dist not in union_req_ids:
        continue

    for slo_config in slo_config_list:
        if slo_config not in union_req_ids[shape_dist]:
            continue

        fig, ax = plt.subplots(figsize=(10, 10))
        # fig.suptitle(
        #     f"CDF of {shape_dist_display[shape_dist]}, SLO {slo_config_short[slo_config]}",
        #     fontsize=32,
        #     fontweight="bold",
        #     y=0.88,
        # )

        # Collect union req ids across all resolutions
        union_ids_all_res = set()
        for ids in union_req_ids[shape_dist][slo_config].values():
            union_ids_all_res.update(ids)

        # Data for CSV export
        csv_data = []

        for schedule_logic in schedule_logic_list:
            # Flatten latencies for reqs in union across resolutions
            latencies = []
            sched_res_map = results_by_req[shape_dist][slo_config].get(schedule_logic, {})
            for req_map in sched_res_map.values():
                for rid, lat in req_map.items():
                    if rid in union_ids_all_res:
                        latencies.append(lat)
                        # Store data for CSV export
                        csv_data.append(
                            {
                                "Schedule_Logic": schedule_logic_short[schedule_logic],
                                "Request_ID": rid,
                                "Latency_s": lat,
                                "Shape_Dist": shape_dist_display[shape_dist],
                                "SLO_Config": slo_config_short[slo_config],
                            }
                        )

            if not latencies:
                continue

            lat_sorted = np.sort(latencies)
            y = np.arange(1, len(lat_sorted) + 1) / len(lat_sorted)
            ax.plot(
                lat_sorted,
                y,
                label=schedule_logic_short[schedule_logic],
                linewidth=4,
            )

        ax.set_xlabel("Latency (s)", fontsize=40, fontweight="bold", fontname="Arial")
        ax.set_ylabel("CDF", fontsize=40, fontweight="bold", fontname="Arial")
        ax.tick_params(axis="both", labelsize=30, pad=8)
        ax.grid(True, alpha=0.3)
        # ax.set_ylim(0.2, 1.01)
        ax.set_ylim(0, 1.0)
        ax.set_xlim(0, 17)
        ax.margins(x=0.06, y=0.06)

        ax.legend(loc="lower right", fontsize=31, frameon=True, fancybox=True, shadow=True)

        plt.tight_layout(rect=[0, 0, 1, 0.90])

        # Apply tick label fontweight after tight_layout (ensure Arial Bold is used)
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontname("Arial")
            lbl.set_fontweight("bold")

        plt.savefig(
            output_dir / f"overall_cdf_per_sched_union_filtered_{shape_dist}_{slo_config}.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.savefig(
            output_dir / f"overall_cdf_per_sched_union_filtered_{shape_dist}_{slo_config}.pdf",
            dpi=300,
            bbox_inches="tight",
        )
        print(
            f"Saved overall per-schedule union-filtered CDF: {output_dir / f'overall_cdf_per_sched_union_filtered_{shape_dist}_{slo_config}.png'}"
        )

        # Export data to CSV
        if csv_data:
            export_df = pd.DataFrame(csv_data)
            csv_path = output_dir / f"overall_cdf_per_sched_union_filtered_{shape_dist}_{slo_config}_data.csv"
            export_df.to_csv(csv_path, index=False)
            print(f"Data exported to {csv_path}")

        plt.close()

# # Plot 5: Overall (all shapes) CDF of latency
# # One figure per (shape_dist, slo_config); single axes, lines = schedule logic
# for shape_dist in shape_dist_list:
#     if shape_dist not in results:
#         continue

#     for slo_config in slo_config_list:
#         if slo_config not in results[shape_dist]:
#             continue

#         fig, ax = plt.subplots(figsize=(12, 10))
#         fig.suptitle(
#             f"Overall CDF of Latency - {shape_dist_display[shape_dist]}, SLO {slo_config_short[slo_config]}",
#             fontsize=32,
#             fontweight="bold",
#             y=0.97,
#         )

#         slo_vals_overall = []
#         for schedule_logic in schedule_logic_list:
#             res_dict = results[shape_dist][slo_config].get(schedule_logic, {})
#             # Flatten latencies across all resolutions
#             latencies = []
#             for res_vals in res_dict.values():
#                 latencies.extend(res_vals)

#             if not latencies:
#                 continue

#             lat_sorted = np.sort(latencies)
#             y = np.arange(1, len(lat_sorted) + 1) / len(lat_sorted)
#             ax.plot(
#                 lat_sorted,
#                 y,
#                 label=schedule_logic_short[schedule_logic],
#                 linewidth=2,
#             )

#             # Collect slo values across resolutions for this schedule
#             for res_key, slo_val in slo_targets[shape_dist][slo_config].get(schedule_logic, {}).items():
#                 if slo_val is not None:
#                     slo_vals_overall.append(slo_val)

#         ax.set_xlabel("Latency (s)", fontsize=28)
#         ax.set_ylabel("CDF", fontsize=28)
#         ax.tick_params(axis="both", labelsize=22)
#         ax.grid(True, alpha=0.3)
#         ax.set_ylim(0.2, 1.01)
#         # Add a representative SLO line (median across available SLOs)
#         if slo_vals_overall:
#             slo_line = np.median(slo_vals_overall)
#             ax.axvline(
#                 slo_line,
#                 color="k",
#                 linestyle="--",
#                 linewidth=2,
#                 label="SLO",
#             )

#         ax.legend(loc="lower right", fontsize=18, frameon=True, fancybox=True, shadow=True)

#         plt.tight_layout(rect=[0, 0, 1, 0.94])
#         plt.savefig(
#             output_dir / f"overall_cdf_{shape_dist}_{slo_config}.png",
#             dpi=150,
#             bbox_inches="tight",
#         )
#         plt.savefig(
#             output_dir / f"overall_cdf_{shape_dist}_{slo_config}.pdf",
#             dpi=300,
#             bbox_inches="tight",
#         )
#         print(
#             f"Saved overall CDF: {output_dir / f'overall_cdf_{shape_dist}_{slo_config}.png'}"
#         )

print(f"\n{'='*140}")
print(f"All per-shape plots saved to: {output_dir}")
print(f"{'='*140}")
