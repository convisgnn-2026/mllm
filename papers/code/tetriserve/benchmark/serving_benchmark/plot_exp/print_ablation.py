import json
import matplotlib.pyplot as plt
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

json_file = "tetriserve_benchmark/serving_benchmark/ablation_study/benchmark_results.json"

with open(json_file, "r") as f:
    data = json.load(f)

schedule_logic_list = [
    "dyn_slo_schedule_global_dp_wo_placement",
    "dyn_slo_schedule_global_dp",
    "dyn_slo_schedule_global_dp_scale_up",
]
shape_dist_list = ["evenly", "long_seqlen_high"]
slo_config_list = [
    "custom_1_1.0x_slo_H100_flux",
    # "custom_1_1.1x_slo_H100_flux",
    # "custom_1_1.3x_slo_H100_flux",
    "custom_1_1.5x_slo_H100_flux",
]

# Preserve original printing for quick inspection
for shape_dist in shape_dist_list:
    for slo_config in slo_config_list:
        print(f"{shape_dist} {slo_config}")
        for schedule_logic in schedule_logic_list:
            print(f"  {schedule_logic}")
            for item in data:
                if (
                    item["config"]["schedule_logic"] == schedule_logic
                    and item["config"]["shape_dist"] == shape_dist
                    and item["config"]["slo_config"] == slo_config
                ):
                    print(f"SLO Meet Ratio: {item['stats']['slo_meet_ratio']:.2f}")
                    print(f"latency_mean: {item['stats']['latency_mean']:.2f}")

# Aggregate results for plotting
records = {}
for item in data:
    cfg = item["config"]
    key = (cfg["shape_dist"], cfg["slo_config"], cfg["schedule_logic"])
    if (
        cfg["shape_dist"] in shape_dist_list
        and cfg["slo_config"] in slo_config_list
        and cfg["schedule_logic"] in schedule_logic_list
    ):
        records[key] = {
            "slo_meet_ratio": item["stats"]["slo_meet_ratio"],
            "latency_mean": item["stats"]["latency_mean"],
        }


def get_metric_list(metric):
    vals = []
    for shape in shape_dist_list:
        for slo in slo_config_list:
            for sched in schedule_logic_list:
                vals.append(records[(shape, slo, sched)][metric])
    return vals


# Build grouped bar plot with two subplots (SAR and Latency)
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
bar_width = 0.22

schedule_labels = {
    "dyn_slo_schedule_global_dp_wo_placement": "naive scheduling",
    "dyn_slo_schedule_global_dp": "with placement",
    "dyn_slo_schedule_global_dp_scale_up": "scale up",
}
schedule_colors = {
    "dyn_slo_schedule_global_dp_wo_placement": "#4C78A8",
    "dyn_slo_schedule_global_dp": "#F58518",
    "dyn_slo_schedule_global_dp_scale_up": "#54A24B",
}

categories = []
for shape in shape_dist_list:
    for slo in slo_config_list:
        categories.append((shape, slo))

x_base = list(range(len(categories)))
for idx, sched in enumerate(schedule_logic_list):
    offsets = [x + (idx - (len(schedule_logic_list) - 1) / 2) * bar_width for x in x_base]
    sar_vals = [records[(shape, slo, sched)]["slo_meet_ratio"] for (shape, slo) in categories]
    lat_vals = [records[(shape, slo, sched)]["latency_mean"] for (shape, slo) in categories]
    axes[0].bar(
        offsets,
        sar_vals,
        width=bar_width,
        label=schedule_labels[sched],
        color=schedule_colors[sched],
    )
    axes[1].bar(
        offsets,
        lat_vals,
        width=bar_width,
        label=schedule_labels[sched],
        color=schedule_colors[sched],
    )

xtick_labels = []
for shape, slo in categories:
    shape_label = "Uniform" if shape == "evenly" else "Skewed"
    slo_label = slo.replace("custom_1_", "").replace("_slo_H100_flux", "")
    xtick_labels.append(f"{slo_label} / {shape_label}")

axes[0].set_ylabel("SLO Meet Ratio", fontweight="bold")
axes[1].set_ylabel("Latency Mean (s)", fontweight="bold")
axes[0].set_ylim(0, 1)
axes[0].set_title("SAR by scheduling logic")
axes[1].set_title("Latency by scheduling logic")

axes[0].set_xticks(x_base)
axes[1].set_xticks(x_base)
for ax in axes:
    ax.set_xticklabels(xtick_labels, rotation=20, ha="right", fontweight="bold")
    # Make y-tick labels bold
    for label in ax.get_yticklabels():
        label.set_fontweight("bold")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

fig.legend(
    loc="upper center",
    ncol=len(schedule_logic_list),
    bbox_to_anchor=(0.5, 1.1),
    frameon=False,
)
fig.tight_layout()
plt.savefig("benchmark/serving_benchmark/plot_exp/ablation_summary.png", dpi=200)
plt.close(fig)
print("Saved plot: benchmark/serving_benchmark/plot_exp/ablation_summary.png")
