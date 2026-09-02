import json
import matplotlib.pyplot as plt
import numpy as np

LOG_DIR = "log/benchmark/scaling_efficiency/sd3/stats"

with open(f"{LOG_DIR}/e2e_scaling_efficiency_stats.json", "r") as f:
    data = json.load(f)

# Example format for the comm/comp ratio dict you'll provide:
# comm_comp_ratios = {
#     (bs, seq_len, ulysses_degree): {
#         "comp_time": float,
#         "comp_std": float,
#         "comm_time": float,
#         "comm_std": float
#     }
# }


def get_grouped_data(data, shape_list=[(256, 256), (512, 512), (1024, 1024), (2048, 2048)]):
    # Group data by seq_len and ulysses_degree
    grouped_data = {}
    for entry in data:
        height = entry["height"]
        width = entry["width"]
        ulysses_degree = entry["ulysses_degree"]
        shape = (height, width)
        if shape not in shape_list:
            continue
        if shape not in grouped_data:
            grouped_data[shape] = {}
        grouped_data[shape][ulysses_degree] = {"avg_time": entry["mean_duration"], "std_time": entry["std_duration"]}
    return grouped_data


grouped_data = get_grouped_data(data)

# Prepare data for plotting
shapes = sorted(grouped_data.keys(), key=lambda x: x[0] * x[1])
ulysses_degrees = sorted(list(set(d["ulysses_degree"] for d in data)))
x = np.arange(len(shapes))
width = 0.2  # Width of the bars

# Create figure and axis with larger size
plt.figure(figsize=(15, 10))
ax = plt.gca()

# Academic color palette
colors = [
    "#2ca02c",  # forest green
    "#1f77b4",  # navy blue
    "#ff7f0e",  # orange
    "#AA3377",  # Purple
]

# Define patterns for computation and communication
comp_pattern = "xx"  # Diagonal lines for computation
comm_pattern = ""  # Different diagonal lines for communication

# Plot bars for each ulysses_degree
bars = []
comp_patches = []
comm_patches = []

for i, degree in enumerate(ulysses_degrees):
    for j, shape in enumerate(shapes):
        # Calculate bar position
        bar_position = x[j] + (i - len(ulysses_degrees) / 2 + 0.5) * width

        # Get the total time and std
        total_time = grouped_data[shape][degree]["avg_time"]
        total_std = grouped_data[shape][degree]["std_time"]

        # Here you would use your comm_comp_ratios dict to get the split
        # For now using dummy values (70% comp, 30% comm)
        # Replace these with actual values from your dictionary
        comp_ratio = 0.7  # Example value
        comm_ratio = 0.3  # Example value

        comp_time = total_time * comp_ratio
        comm_time = total_time * comm_ratio

        # Plot computation part (bottom)
        comp_bar = ax.bar(
            bar_position,
            comp_time,
            width,
            color=colors[i % len(colors)],
            alpha=0.7,
            hatch=comp_pattern,
            label=f"P={degree} (Comp)" if j == 0 else "",
        )

        # Plot communication part (top)
        if degree > 1:  # Only add communication for parallel sizes > 1
            comm_bar = ax.bar(
                bar_position,
                comm_time,
                width,
                bottom=comp_time,  # Stack on top of computation
                color=colors[i % len(colors)],
                alpha=0.7,
                hatch=comm_pattern,
                label=f"P={degree} (Comm)" if j == 0 else "",
            )

            # Add error bar at the total height
            total_height = comp_time + comm_time
        else:
            # For degree=1, only use computation height
            total_height = comp_time

        # Add single error bar at the top of the total bar
        ax.errorbar(
            bar_position,
            total_height,
            yerr=total_std,
            fmt="none",
            ecolor="black",
            capsize=5,
            capthick=2,
            elinewidth=2,
            alpha=0.7,
        )

# Create legend patches for computation and communication
comp_patch = plt.Rectangle((0, 0), 1, 1, fc="gray", alpha=0.7, hatch=comp_pattern, label="Computation")
comm_patch = plt.Rectangle((0, 0), 1, 1, fc="gray", alpha=0.7, hatch=comm_pattern, label="Communication")

# Create legend patches for parallel sizes
parallel_patches = [
    plt.Rectangle((0, 0), 1, 1, fc=colors[i], alpha=0.7, label=f"P={d}") for i, d in enumerate(ulysses_degrees)
]

# Customize the plot
ax.set_ylabel("End-to-End Time (s)", fontsize=48)
ax.set_xlabel("Height × Width", fontsize=32)
# ax.set_title('End-to-End Time vs Input Size for Different Parallel Sizes', fontsize=48)

# Set x-axis ticks
# Calculate height/width from sequence length
seq_len_labels = []
for shape in shapes:
    seq_len_labels.append(f"{shape[0]}×{shape[1]}")

ax.set_xticks(x)
ax.set_xticklabels(seq_len_labels, rotation=0)

# Increase tick label sizes
ax.tick_params(axis="both", which="major", labelsize=32)
ax.tick_params(width=2)
ax.tick_params(length=10)

# Add legend with both component types and parallel sizes
legend_handles = [comp_patch, comm_patch] + parallel_patches
ax.legend(handles=legend_handles, fontsize=32, frameon=True)

# Enhance the academic look
ax.spines["top"].set_visible(True)
ax.spines["right"].set_visible(True)
for spine in ax.spines.values():
    spine.set_linewidth(3.0)
    spine.set_color("#333333")

# Set background color
ax.set_facecolor("#F8F8F8")

# Adjust layout
plt.tight_layout()

# Save the plot
plt.savefig(f"{LOG_DIR}/all_to_all_4D_time_4.png", dpi=300, bbox_inches="tight")
plt.savefig(f"{LOG_DIR}/all_to_all_4D_time_4.pdf", dpi=300, bbox_inches="tight")
plt.close()
