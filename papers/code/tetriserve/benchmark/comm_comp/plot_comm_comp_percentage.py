import json
import matplotlib.pyplot as plt
import numpy as np
import csv
from pathlib import Path
from matplotlib import font_manager as fm

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

# Load ASPLOS camera-ready style (includes Arial font and bold ticks)
# Use absolute path relative to this script's location
script_dir = Path(__file__).parent.resolve()
style_file = script_dir.parent / "plot_style" / "asplos_camera_ready.mplstyle"
if not style_file.exists():
    raise FileNotFoundError(f"Style file not found at: {style_file}")
plt.style.use(str(style_file))
print(f"Using style file: {style_file}")

parallel_2_file = "TetriServe_H100_by_Shenggui/comm_comp/flux/detailed_comm_comp_time_2.json"
parallel_4_file = "TetriServe_H100_by_Shenggui/comm_comp/flux/detailed_comm_comp_time_4.json"
parallel_8_file = "TetriServe_H100_by_Shenggui/comm_comp/flux/detailed_comm_comp_time_8.json"


def read_data(file_path):
    compute_states = {}
    with open(file_path, "r") as f:
        data = json.load(f)
    for entry in data:
        height = entry["height"]
        width = entry["width"]
        parallel_degree = entry["ulysses_degree"]
        assert entry["ring_degree"] == 1
        compute_states[(height, width, parallel_degree)] = {
            "comp_time_avg": entry["comp_time_avg"],
            "comm_time_avg": entry["comm_time_avg"],
            "e2e_time_avg": entry["e2e_time_avg"],
        }
    return compute_states


parallel_2_compute_states = read_data(parallel_2_file)
parallel_4_compute_states = read_data(parallel_4_file)
parallel_8_compute_states = read_data(parallel_8_file)

parallel_states = parallel_2_compute_states | parallel_4_compute_states | parallel_8_compute_states

sizes = [256, 512, 1024, 2048]
parallel_degrees = [2, 4, 8]
# parallel_degrees = [1, 2, 4, 8]  # Uncomment to include parallel degree 1 (always 0.0%)

# Collect data for plotting
data = {}
for size in sizes:
    data[size] = {}
    for parallel_degree in parallel_degrees:
        # Uncomment below if using parallel_degree = 1
        # if parallel_degree == 1:
        #     data[size][parallel_degree] = 0.0
        # else:
        compute_states = parallel_states[(size, size, parallel_degree)]
        comm_comp_ratio = compute_states["comm_time_avg"] / (
            compute_states["comp_time_avg"] + compute_states["comm_time_avg"]
        )
        data[size][parallel_degree] = comm_comp_ratio
        print(
            f"shape: {size}x{size}, parallel_degree: {parallel_degree}, "
            f"the percentage of comm/comp time is {comm_comp_ratio:.2f}"
        )

# Create the plot
fig, ax = plt.subplots(figsize=(12, 6))

# Bar positions
x = np.arange(len(sizes))
width = 0.2
multiplier = 0

# Academic color palette
colors = ["#DD8452", "#55A868", "#C44E52"]
# colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]  # Uncomment if using parallel degree 1

# Hatch patterns (/// style)
hatches = ["///", "///", "///"]
# hatches = ["", "///", "///", "///"]  # Uncomment if using parallel degree 1

# Put grid behind bars
ax.set_axisbelow(True)
ax.grid(axis="y", alpha=0.3, linestyle="--", zorder=0)

# Create bars for each parallel degree
for idx, parallel_degree in enumerate(parallel_degrees):
    offset = width * multiplier
    values = [data[size][parallel_degree] * 100 for size in sizes]  # percentage

    bars = ax.bar(
        x + offset,
        values,
        width,
        label=f"Degree of Parallelism {parallel_degree}",
        color=colors[idx],
        edgecolor="black",
        linewidth=0.8,
        hatch=hatches[idx],
        zorder=3,
    )

    # Value labels (removed % from display)
    for i, v in enumerate(values):
        ax.text(
            x[i] + offset,
            v + 0.3,
            f"{v:.1f}",
            ha="center",
            va="bottom",
            fontsize=16,
        )

    multiplier += 1

# Customize plot (bold tick labels are handled by the style file)
ax.set_xlabel("Image Size", fontsize=28, fontweight="bold", fontfamily="Arial")
ax.set_ylabel("Comm Percentage(%)", fontsize=25, fontweight="bold", fontfamily="Arial")
ax.set_xticks(x + width)  # Centered for 3 bars
ax.set_xticklabels([f"{size}x{size}" for size in sizes], fontsize=24)
ax.tick_params(axis="y", labelsize=18)
ax.legend(loc="upper right", fontsize=22)

max_val = max(data[size][pd] * 100 for size in sizes for pd in parallel_degrees)
ax.set_ylim([0, max_val * 1.2])

plt.tight_layout()

# Apply tick label fontweight after tight_layout (matplotlib doesn't support labelweight in style files)
for label in ax.get_xticklabels():
    label.set_fontweight("bold")
for label in ax.get_yticklabels():
    label.set_fontweight("bold")

plt.savefig("benchmark/comm_comp/comm_comp_ratio_graph.png", dpi=300, bbox_inches="tight")
plt.savefig("benchmark/comm_comp/comm_comp_ratio_graph.pdf", dpi=300, bbox_inches="tight")
print("\nGraph saved to benchmark/comm_comp/comm_comp_ratio_graph.png")

# Export data to CSV
csv_file = "benchmark/comm_comp/comm_comp_ratio_data.csv"
with open(csv_file, "w", newline="") as f:
    writer = csv.writer(f)
    # Write header
    writer.writerow(["Image_Size", "Degree_of_Parallelism", "Comm_Percentage"])
    # Write data rows
    for size in sizes:
        for parallel_degree in parallel_degrees:
            comm_percentage = data[size][parallel_degree] * 100
            writer.writerow([f"{size}x{size}", parallel_degree, f"{comm_percentage:.2f}"])

print(f"Data exported to {csv_file}")

plt.show()

print(ax.get_xticklabels()[0].get_fontfamily())
print(ax.get_xticklabels()[0].get_fontweight())
lbl = ax.get_xticklabels()[0]
print(lbl.get_fontproperties().get_name())
