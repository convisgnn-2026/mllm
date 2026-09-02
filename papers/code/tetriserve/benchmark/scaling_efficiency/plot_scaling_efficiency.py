import matplotlib.pyplot as plt
import json
import numpy as np
from matplotlib.lines import Line2D
import os
import pandas as pd
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


def plot_e2e_efficiency(data, LOG_DIR, shape_list=[(256, 256), (512, 512), (1024, 1024), (2048, 2048)]):
    """Plot end-to-end efficiency and export data to CSV."""
    bs_seqlen_groups = {}

    # Data for CSV export
    csv_data = []

    for entry in data:
        if (entry["height"], entry["width"]) not in shape_list:
            continue
        if entry["bs"] > 4:
            continue
        key = (entry["bs"], entry["height"], entry["width"])
        if key not in bs_seqlen_groups:
            bs_seqlen_groups[key] = []

        if entry["ring_degree"] == 1:
            bs_seqlen_groups[key].append(entry)

    filtered_groups = {k: v for k, v in bs_seqlen_groups.items() if len(v) >= 3}
    sorted_groups = sorted(filtered_groups.items(), key=lambda item: item[0][0] * item[0][1] * item[0][2])

    markers = ["o", "s", "D", "P", "X", "H", "v", "d", "p", "h"]

    # Group data by shape
    shape_groups = {}
    for (bs, height, width), entries in sorted_groups:
        shape = (height, width)
        if shape not in shape_groups:
            shape_groups[shape] = []
        shape_groups[shape].append((bs, entries))

    # Ensure output directory exists
    output_dir = f"{LOG_DIR}/individual_plots"
    os.makedirs(output_dir, exist_ok=True)

    # Get the total number of shapes for determining the last graph
    total_shapes = len(shape_groups)

    # Create and save individual plots
    for idx, (shape, bs_entries) in enumerate(shape_groups.items()):
        # Create a new figure for each shape with massive size
        plt.figure(figsize=(18, 16))  # Massively increased figure size
        ax = plt.gca()

        # Define colors for specific batch sizes
        color_map = {
            1: "#ff7f0e",  # Orange for BS=1
            2: "#1f77b4",  # Blue for BS=2
            4: "#2ca02c",  # Green for BS=4
        }
        all_x_vals = set()

        for color_idx, (bs, entries) in enumerate(sorted(bs_entries, key=lambda x: x[0])):
            entries = sorted(entries, key=lambda x: x["ulysses_degree"])
            ulysses_degrees = [entry["ulysses_degree"] for entry in entries]
            # times = [entry["avg_e2e_time"] / entry["steps"] * 10 for entry in entries]
            times = [
                entry["avg_text_encoding_time"] + entry["avg_denoising_time"] / entry["steps"] * 20 for entry in entries
            ]
            # times = [entry["avg_denoising_time"] / entry["steps"] * 10 for entry in entries]

            all_x_vals.update(ulysses_degrees)

            if ulysses_degrees and times:
                base_ulysses_degree = ulysses_degrees[0]
                base_time = times[0]
                # Removed ideal times calculation and plotting
                # ideal_times = [base_time / (w_size / base_ulysses_degree) for w_size in ulysses_degrees]
                color = color_map.get(bs, "#808080")  # Use gray as fallback for other BS values
                marker = markers[color_idx % len(markers)]
                line_label = f"BS={bs}"

                # Store data for CSV export
                for ulysses_deg, time_val in zip(ulysses_degrees, times):
                    csv_data.append(
                        {
                            "Shape": f"{shape[0]}x{shape[1]}",
                            "Batch_Size": bs,
                            "Ulysses_Degree": ulysses_deg,
                            "End_to_End_Time_s": time_val,
                        }
                    )

                # Removed ideal line plotting
                # ax.plot(ulysses_degrees, ideal_times, "--", color=color, linewidth=7, dashes=(7, 7))

                # Plot only the actual line with extremely large marker and massive line width
                ax.plot(
                    ulysses_degrees,
                    times,
                    "-",
                    color=color,
                    marker=marker,
                    markersize=22,
                    linewidth=7,
                    label=line_label,
                )

        # Massive font size for axis labels - extremely large now
        ax.set_xlabel("Ulysses Degree", fontsize=80, fontweight="bold")  # Increased from 36 to 48
        ax.set_ylabel("End-to-End Time (s)", fontsize=80, fontweight="bold")  # Increased from 36 to 48

        # No title at all

        # Make grid lines extremely prominent
        ax.grid(True, linestyle="--", alpha=0.7, linewidth=2.0)

        # Use log scale on x-axis to make intervals equal
        ax.set_xscale("log", base=2)

        # Set the x-ticks to be exactly at 1, 2, 4, 8
        ax.set_xticks(sorted(all_x_vals))
        ax.set_xticklabels([str(int(x)) for x in sorted(all_x_vals)])

        # Make tick labels massively large and bold
        ax.tick_params(axis="both", which="major", labelsize=80)  # Increased from 32 to 44

        # Make tick labels bold (style file doesn't support this)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight("bold")

        # Add extremely prominent minor grid lines
        ax.grid(True, which="minor", linestyle=":", alpha=0.5, linewidth=1.5)

        # Set y-axis to start from 0
        ax.set_ylim(bottom=0)

        # Only add legend to the last plot
        if idx == total_shapes - 1:
            # Removed custom lines for ideal/actual
            # custom_lines = [
            #     Line2D([0], [0], linestyle="--", color="gray", linewidth=7, label="Ideal"),
            #     Line2D([0], [0], linestyle="-", color="gray", marker="o", markersize=22, linewidth=7, label="Actual"),
            # ]
            handles, labels = ax.get_legend_handles_labels()

            # Make legend absolutely massive - now just showing batch sizes
            legend = ax.legend(handles=handles, fontsize=80, loc="best", frameon=True)  # Simplified legend

            # Make legend frame extremely prominent
            legend.get_frame().set_linewidth(4.0)  # Increased from 3.0 to 4.0
            legend.get_frame().set_edgecolor("black")
            legend.get_frame().set_facecolor("white")
            legend.get_frame().set_alpha(0.95)

        plt.tight_layout()

        # Save the individual plot
        plt.savefig(
            f"{output_dir}/e2e_efficiency_{shape[0]}x{shape[1]}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.savefig(
            f"{output_dir}/e2e_efficiency_{shape[0]}x{shape[1]}.pdf",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    # Export data to CSV after all plots are generated
    if csv_data:
        df = pd.DataFrame(csv_data)
        csv_path = f"{LOG_DIR}/e2e_efficiency_data.csv"
        df.to_csv(csv_path, index=False)
        print(f"Data exported to {csv_path}")


if __name__ == "__main__":
    MODELS = ["flux", "sd3"]
    gpu_names = [
        # "L40",
        "H100_HBM3",
        # "A40",
    ]

    for model in MODELS:
        for gpu_name in gpu_names:
            LOG_DIR = f"log/benchmark/scaling_efficiency/{model}/{gpu_name}"
            try:
                with open(f"{LOG_DIR}/e2e_scaling_efficiency.json", "r") as f:
                    data = json.load(f)
            except FileNotFoundError as e:
                print(f"No data found for {LOG_DIR}: {e}")
                continue

            if model == "flux":
                shape_list = [(256, 256), (512, 512), (1024, 1024), (2048, 2048)]
            elif model == "sd3":
                shape_list = [(256, 256), (512, 512), (1024, 1024), (2048, 2048)]
            else:
                raise ValueError(f"Invalid model: {model}")

            plot_e2e_efficiency(data, LOG_DIR, shape_list=shape_list)
            print(f"Plotted data for {LOG_DIR}")
