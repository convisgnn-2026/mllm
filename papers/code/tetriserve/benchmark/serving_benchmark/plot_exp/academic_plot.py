"""
Academic plotting module for paper-ready figures.

Creates publication-quality plots with consistent styling, proper naming conventions, and unified colors.
"""

from typing import Optional, Dict, List
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from matplotlib import font_manager as fm

from .config import (
    PlotConfig,
    get_ordered_schedule_logics,
    PLOT_PARAMS,
)
from .data_processor import (
    get_experimental_conditions,
    filter_by_condition,
    generate_condition_string,
    generate_save_path_suffix,
)


# Arial font validation
def validate_arial_font():
    """Validate that Arial font is available and register it with matplotlib."""
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


# Academic naming conventions
ACADEMIC_SCHEDULE_NAME_MAPPING = {
    "data_parallel": "xDiT (SP=1)",
    "fixed_sp_2": "xDiT (SP=2)",
    "fixed_sp_4": "xDiT (SP=4)",
    "fixed_sp_8": "xDiT (SP=8)",
    # "per_resolution_schedule": "Static SP",
    "per_resolution_schedule_1_1_2_8": "RSSP",
    "per_resolution_schedule_1_1_2_4": "RSSP",
    "dyn_slo_schedule_global_scale_up": "TetriServe (ours) + Scale-up",
    "dyn_slo_schedule_global": "TetriServe (ours)",
    # "dyn_slo_schedule_global_dp": "TetriServe (ours) + DP",
    "dyn_slo_schedule_global_dp": "TetriServe (ours)",
}

# Academic color scheme
ACADEMIC_COLORS = {
    "TetriServe (ours)": "#FF8C00",  # Orange for TetriServe
    "TetriServe (ours) + DP": "#FF8C00",  # Orange for TetriServe
    "xDiT (SP=1)": "#87CEEB",  # Light blue for SP=1
    "xDiT (SP=2)": "#A9A9A9",  # Gray for SP=2
    "xDiT (SP=4)": "#696969",  # Dark gray for SP=4
    "xDiT (SP=8)": "#000000",  # Pure black for SP=8
    "Static SP": "#4B0082",  # Indigo for Static SP
    "RSSP": "#4B0082",  # Indigo for RSSP
}


def remap_academic_schedule_name(schedule_logic: str) -> str:
    """Remap schedule logic names for academic papers."""
    return ACADEMIC_SCHEDULE_NAME_MAPPING.get(schedule_logic, schedule_logic)


def get_academic_colors_for_schedules(schedule_logics: List[str]) -> List[str]:
    """Get academic colors for the given schedule logics."""
    colors = []
    for logic in schedule_logics:
        academic_name = remap_academic_schedule_name(logic)
        color = ACADEMIC_COLORS.get(academic_name, "#808080")  # Default gray
        colors.append(color)
    return colors


class AcademicPlotter:
    """Creates publication-ready plots for academic papers."""

    def __init__(self, config: PlotConfig = None):
        self.config = config or PlotConfig()
        # Force academic style
        self.config.use_academic_style = True
        self.config.color_scheme = "academic"

        # Apply academic matplotlib settings
        self._setup_academic_style()

    def _setup_academic_style(self):
        """Setup academic matplotlib style settings."""
        # Validate Arial font availability
        validate_arial_font()

        plt.rcParams.update(
            {
                # Fonts
                "font.size": 9,
                "font.family": "sans-serif",
                "font.sans-serif": ["Arial"],
                # Saving figures
                "path.simplify": True,
                # SVG export settings
                "svg.fonttype": "none",
                "svg.hashsalt": "42",
                # PDF/PS font settings
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
                # Additional academic settings
                "axes.linewidth": 1.0,
                "axes.edgecolor": "black",
                "axes.facecolor": "white",
                "figure.facecolor": "white",
                "savefig.facecolor": "white",
                "savefig.edgecolor": "none",
            }
        )

    def plot_all_academic_figures(self, df: pd.DataFrame, save_dir: str = None) -> None:
        """Generate all academic figures for the paper."""
        if df.empty:
            print("No data available, skipping academic plots")
            return

        # Create academic plot directory
        if save_dir:
            base_path = Path(save_dir)
            base_path.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print("GENERATING ACADEMIC PLOTS FOR PAPER")
        print(f"{'='*60}")

        # Generate academic plots (only essential ones for now)
        self.plot_sar_merged(df, save_dir)
        self.plot_sar_spider_per_rate(df, save_dir)

        print(f"\n{'='*60}")
        print("ACADEMIC PLOTS COMPLETE")
        print(f"{'='*60}")

    def plot_sar_merged(self, df: pd.DataFrame, save_dir: str = None) -> None:
        """Generate merged SAR plot for academic paper."""
        print("\nGenerating academic merged SAR plot...")

        if save_dir:
            sar_dir = Path(save_dir) / "sar_merged"
            sar_dir.mkdir(parents=True, exist_ok=True)

            # Use academic SAR plotter
            plotter = AcademicSARMergedPlotter(self.config)
            plotter.plot(df, str(sar_dir))

    def plot_sar_spider_per_rate(self, df: pd.DataFrame, save_dir: str = None) -> None:
        """Generate SAR spider plots per rate for academic paper."""
        print("\nGenerating academic SAR spider per rate plots...")

        if save_dir:
            spider_dir = Path(save_dir) / "sar_spider_per_rate"
            spider_dir.mkdir(parents=True, exist_ok=True)

            # Use the spider plotter with academic styling
            plotter = AcademicSARSpiderPlotter(self.config)
            plotter.plot(df, str(spider_dir))


class AcademicSARMergedPlotter:
    """Academic version of merged SLO meet ratio plotter with SAR naming."""

    def __init__(self, config: PlotConfig = None):
        self.config = config or PlotConfig()

    def plot(self, df: pd.DataFrame, save_dir: str = None) -> None:
        """Plot academic merged SAR (one plot comparing all SLO configs)."""
        if df.empty:
            print("No data available, skipping academic merged SAR plot")
            return

        # Check if we have SLO config data
        if "slo_config" not in df.columns or df["slo_config"].isna().all():
            print("No SLO config data available, skipping academic merged SAR plot")
            return

        # Get experimental conditions (excluding SLO config)
        conditions = get_experimental_conditions(df)

        for condition in conditions.itertuples(index=False):
            condition_data = filter_by_condition(df, condition)
            if not condition_data.empty:
                self._create_academic_merged_sar_plot(condition_data, condition, save_dir)

    def _create_academic_merged_sar_plot(self, condition_data: pd.DataFrame, condition, save_dir: Optional[str]):
        """Create academic merged SAR plot showing all SLO configs in one chart."""
        rates = sorted(condition_data["rate"].unique())
        slo_configs = sorted(condition_data["slo_config"].dropna().unique())

        if len(slo_configs) < 2:
            print(f"Not enough SLO configs ({len(slo_configs)}) for merged plot, skipping")
            return

        # Calculate average SAR for each scheduler and SLO config
        plot_data = {}
        schedule_logics = get_ordered_schedule_logics(condition_data["schedule_logic"].unique())

        for schedule_logic in schedule_logics:
            plot_data[schedule_logic] = []

            for slo_config in slo_configs:
                slo_data = condition_data[condition_data["slo_config"] == slo_config]
                sar_values = []

                # Calculate SAR for each rate
                for rate in rates:
                    rate_data = slo_data[
                        (slo_data["schedule_logic"] == schedule_logic) & (abs(slo_data["rate"] - rate) < 0.01)
                    ]

                    if not rate_data.empty:
                        sar = rate_data["slo_meet_ratio"].mean()
                        sar_values.append(sar)

                # Calculate average across all rates
                avg_sar = np.mean(sar_values) if sar_values else 0
                plot_data[schedule_logic].append(avg_sar)

        # Create the grouped bar chart with academic paper style
        fig, ax = plt.subplots(figsize=(12, 4))  # Wider but shorter for better aspect ratio

        # Setup bar positions
        x = np.arange(len(slo_configs))
        n_groups = len(schedule_logics)
        width = 0.15  # Fixed wider bar width

        # Use proper academic colors
        academic_colors = get_academic_colors_for_schedules(schedule_logics)

        # Plot bars for each scheduler
        for i, schedule_logic in enumerate(schedule_logics):
            offset = (i - (n_groups - 1) / 2) * width
            bars = ax.bar(
                x + offset,
                plot_data[schedule_logic],
                width,
                label=remap_academic_schedule_name(schedule_logic),
                color=academic_colors[i],
                edgecolor="black",
                linewidth=0.8,
            )

        # Academic paper styling with bigger axis fonts
        ax.set_xlabel("SLO Scale", fontsize=24, fontweight="bold")
        ax.set_ylabel("SLO Attainment Ratio", fontsize=20, fontweight="bold")

        # Extract SLO ratios for display
        slo_ratios = []
        for slo in slo_configs:
            parts = slo.split("_")
            ratio = None
            for i, part in enumerate(parts):
                if "." in part or part.replace(".", "").replace("x", "").isdigit():
                    if i + 1 < len(parts) and parts[i + 1].startswith("slo"):
                        ratio = part
                        break
            if ratio is None:
                ratio = "1.0x"
            slo_ratios.append(ratio)

        ax.set_xticks(x)
        ax.set_xticklabels(slo_ratios, fontsize=20, fontweight="bold")

        # Bigger tick labels for axis values - make ytick labels bold
        ax.tick_params(axis="both", which="major", labelsize=20)
        for label in ax.get_yticklabels():
            label.set_fontweight("bold")

        # Add subtle grid lines
        ax.grid(True, axis="y", linestyle="-", alpha=0.3, linewidth=0.8, color="gray")
        ax.set_axisbelow(True)

        # Set y-axis limits
        ax.set_ylim(0, 1.1)  # Slightly higher for labels
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

        # Add legend at the top
        ax.legend(
            bbox_to_anchor=(0.5, 1.02),
            loc="lower center",
            ncol=len(schedule_logics),  # Horizontal layout
            fontsize=14,
            frameon=True,
            fancybox=False,
            edgecolor="black",
            framealpha=1.0,
        )

        # # Add legend at the top (2 rows x 3 columns layout)
        # ax.legend(
        #     bbox_to_anchor=(0.5, 1.08),
        #     loc="lower center",
        #     ncol=3,  # 3 columns, automatically creates 2 rows for 6 items
        #     fontsize=14,
        #     frameon=True,
        #     fancybox=False,
        #     edgecolor="black",
        #     framealpha=1.0,
        # )

        # Remove top and right spines for cleaner look
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Make remaining spines thicker
        ax.spines["left"].set_linewidth(1.2)
        ax.spines["bottom"].set_linewidth(1.2)

        plt.tight_layout()

        # Save plot in multiple formats
        if save_dir:
            filename_base = f"academic_sar_merged_{generate_save_path_suffix(condition, '')}"

            # Save PNG (high quality)
            png_path = f"{save_dir}/{filename_base}.png"
            plt.savefig(png_path, dpi=self.config.dpi, bbox_inches="tight", facecolor="white", edgecolor="none")
            print(f"Saved academic merged SAR plot: {png_path}")

            # Save PDF (vector, Type 42 fonts for compatibility)
            pdf_path = f"{save_dir}/{filename_base}.pdf"
            plt.savefig(pdf_path, bbox_inches="tight", facecolor="white", edgecolor="none")
            print(f"Saved academic merged SAR plot (PDF): {pdf_path}")

            # Save SVG (for MS Office integration)
            svg_path = f"{save_dir}/{filename_base}.svg"
            plt.savefig(svg_path, bbox_inches="tight", facecolor="white", edgecolor="none")
            print(f"Saved academic merged SAR plot (SVG): {svg_path}")

            # Export data to CSV
            csv_data = []
            for schedule_logic in schedule_logics:
                for i, slo_config in enumerate(slo_configs):
                    csv_data.append(
                        {
                            "Schedule_Logic": remap_academic_schedule_name(schedule_logic),
                            "SLO_Config": slo_config,
                            "Average_SAR": plot_data[schedule_logic][i],
                        }
                    )

            if csv_data:
                export_df = pd.DataFrame(csv_data)
                csv_path = f"{save_dir}/{filename_base}_data.csv"
                export_df.to_csv(csv_path, index=False)
                print(f"Data exported to {csv_path}")

        plt.close()


class AcademicSARSpiderPlotter:
    """Academic version of spider plotter for SAR analysis - per rate per SLO."""

    def __init__(self, config: PlotConfig = None):
        self.config = config or PlotConfig()

    def plot(self, df: pd.DataFrame, save_dir: str = None) -> None:
        """Plot academic SAR spider plots per rate per SLO (one plot per rate-SLO combination)."""
        if df.empty:
            print("No data available, skipping academic SAR spider plots")
            return

        # Get experimental conditions
        conditions = get_experimental_conditions(df)
        print(f"Generating academic per-rate per-SLO spider plots for {len(conditions)} experimental conditions")

        total_plots = 0
        for idx, condition in enumerate(conditions.itertuples(index=False)):
            condition_data = filter_by_condition(df, condition)

            if condition_data.empty:
                continue

            # Get unique rates and SLO configs
            rates = sorted(condition_data["rate"].unique())
            slo_configs = (
                sorted(condition_data["slo_config"].dropna().unique())
                if "slo_config" in condition_data.columns
                else [None]
            )

            # Create individual plots for each rate-SLO combination
            for rate in rates:
                for slo_config in slo_configs:
                    rate_slo_data = condition_data[abs(condition_data["rate"] - rate) < 0.01]
                    if slo_config:
                        rate_slo_data = rate_slo_data[rate_slo_data["slo_config"] == slo_config]

                    if not rate_slo_data.empty:
                        self._create_academic_single_spider_plot(rate_slo_data, condition, save_dir, rate, slo_config)
                        total_plots += 1

            # Progress indicator
            if (idx + 1) % 5 == 0 or idx == 0:
                print(f"Processed {idx + 1}/{len(conditions)} experimental conditions")

        print(f"Generated {total_plots} academic per-rate per-SLO spider plots")

    def _create_academic_single_spider_plot(
        self,
        data: pd.DataFrame,
        condition,
        save_dir: Optional[str],
        rate: float,
        slo_config: Optional[str],
    ):
        """Create a single academic spider plot showing SAR across resolutions for a specific rate-SLO combination."""
        # Get unique shapes and scheduling algorithms
        shapes = ["256x256", "512x512", "1024x1024", "2048x2048"]
        schedule_logics = get_ordered_schedule_logics(data["schedule_logic"].unique())

        # Create single spider plot with academic styling
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection="polar"))

        # Prepare SAR data for each scheduler - calculate per-resolution from per_request_info
        spider_data = {}
        for schedule_logic in schedule_logics:
            logic_data = data[data["schedule_logic"] == schedule_logic]

            if logic_data.empty:
                continue

            # Calculate SAR for each resolution from per_request_info
            sar_ratios = []
            for shape in shapes:
                # Extract per-resolution SAR from per_request_info
                shape_sar = self._calculate_shape_sar_from_per_request_info(logic_data, shape)
                sar_ratios.append(shape_sar if shape_sar is not None else 0)

            if any(ratio > 0 for ratio in sar_ratios):
                spider_data[schedule_logic] = sar_ratios

        if not spider_data:
            ax.text(0.5, 0.5, f"No SAR data", ha="center", va="center", transform=ax.transAxes, fontsize=12)
            plt.close()
            return

        # Number of variables (shapes)
        num_vars = len(shapes)

        # Compute angle for each axis
        angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
        angles += angles[:1]  # Complete the circle

        # Use the proper academic colors matching other academic plots
        colors = get_academic_colors_for_schedules(schedule_logics)

        # Plot each scheduling algorithm with academic styling
        for i, schedule_logic in enumerate(schedule_logics):
            if schedule_logic in spider_data:
                values = spider_data[schedule_logic]
                values += values[:1]  # Complete the circle

                # Plot without circles, use lighter fill that stacks well, thicker lines
                ax.plot(
                    angles,
                    values,
                    "-",
                    linewidth=3,
                    label=remap_academic_schedule_name(schedule_logic),
                    color=colors[i],
                )
                ax.fill(angles, values, alpha=0.1, color=colors[i])  # Very light fill

        # Fix axis to go in the right order and start at 12 o'clock
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)

        # Draw axis lines for each angle and add labels with much bigger fonts
        ax.set_xticks(angles[:-1])

        # Create custom labels - empty strings for labels that will be rotated
        custom_labels = []
        for shape in shapes:
            if shape in ["2048x2048", "512x512"]:
                # These will be rotated vertically, so use empty string for tick label
                custom_labels.append("")
            else:
                custom_labels.append(shape)

        ax.set_xticklabels(custom_labels, size=28, fontweight="bold")  # Much bigger

        # Manually add vertical text for specific labels
        for i, (angle, shape) in enumerate(zip(angles[:-1], shapes)):
            if shape in ["2048x2048", "512x512"]:
                # Add vertical text closer to the spider plot circle
                ax.text(
                    angle, 1.10, shape, ha="center", va="center", fontsize=28, fontweight="bold", rotation=90
                )  # Much bigger, closer to circle

        # Set y-axis limits (SAR is between 0 and 1)
        ax.set_ylim(0, 1.0)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.tick_params(axis="y", labelsize=24)  # Much bigger radial tick labels
        for label in ax.get_yticklabels():
            label.set_fontweight("bold")

        # Hide the outer boundary circle (this was causing the dark 1.0 circle)
        ax.spines["polar"].set_visible(False)

        # Disable ALL default grid elements
        ax.grid(False)  # Turn off default grid completely
        ax.set_rgrids([])  # Remove default radial grid

        # Draw custom radial grid circles with specific alphas
        circle_angles = np.linspace(0, 2 * np.pi, 100)
        radial_values = [0.2, 0.4, 0.6, 0.8, 1.0]
        radial_alphas = [0.3, 0.3, 0.3, 0.3, 0.3]  # All circles same visibility

        for radius, alpha in zip(radial_values, radial_alphas):
            ax.plot(circle_angles, [radius] * len(circle_angles), color="gray", alpha=alpha, linewidth=0.8)

        # Add radial labels manually at specific angle
        label_angle = np.pi / 4  # 45 degrees
        for radius in radial_values:
            ax.text(
                label_angle,
                radius,
                f"{radius:.1f}",
                fontsize=24,
                ha="center",
                va="center",  # Much bigger
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.9, edgecolor="none"),
            )

        # Draw radial lines (spokes) with light alpha
        for angle in angles[:-1]:
            ax.plot([angle, angle], [0, 1.0], color="gray", alpha=0.15, linewidth=0.5)

        # Add radial labels with academic terminology
        ax.set_ylabel("SLO Attainment Ratio (SAR)", labelpad=55, fontsize=28, fontweight="bold")  # Much bigger

        # No title to keep plots clean for academic papers
        # No legend - keep spider plots clean for academic papers

        plt.tight_layout()

        # Save plot in multiple formats
        if save_dir:
            rate_str = f"rate-{rate*60:.0f}rpm"
            slo_str = f"slo-{slo_config}" if slo_config else ""
            filename = (
                f"academic_spider_sar_per_rate_per_slo_{rate_str}_{slo_str}_{generate_save_path_suffix(condition, '')}"
            )

            # Save PNG
            png_path = f"{save_dir}/{filename}.png"
            plt.savefig(png_path, dpi=self.config.dpi, bbox_inches="tight", facecolor="white", edgecolor="none")
            print(f"Saved academic per-rate per-SLO spider plot: {png_path}")

            # Save PDF
            pdf_path = f"{save_dir}/{filename}.pdf"
            plt.savefig(pdf_path, bbox_inches="tight", facecolor="white", edgecolor="none")
            print(f"Saved academic per-rate per-SLO spider plot (PDF): {pdf_path}")

            # Save SVG
            svg_path = f"{save_dir}/{filename}.svg"
            plt.savefig(svg_path, bbox_inches="tight", facecolor="white", edgecolor="none")
            print(f"Saved academic per-rate per-SLO spider plot (SVG): {svg_path}")

            # Export data to CSV
            csv_data = []
            for schedule_logic in schedule_logics:
                if schedule_logic in spider_data:
                    for i, shape in enumerate(shapes):
                        csv_data.append(
                            {
                                "Schedule_Logic": remap_academic_schedule_name(schedule_logic),
                                "Shape": shape,
                                "SAR": spider_data[schedule_logic][i],
                                "Rate_RPM": rate * 60,
                                "SLO_Config": slo_config if slo_config else "N/A",
                            }
                        )

            if csv_data:
                export_df = pd.DataFrame(csv_data)
                csv_path = f"{save_dir}/{filename}_data.csv"
                export_df.to_csv(csv_path, index=False)
                print(f"Data exported to {csv_path}")

        plt.close()

    def _calculate_shape_sar_from_per_request_info(self, data: pd.DataFrame, shape: str) -> Optional[float]:
        """Calculate SAR for a specific shape/resolution from per_request_info."""
        import json

        total_requests = 0
        slo_met_requests = 0

        # Parse shape string (e.g., "1024x1024" -> width=1024, height=1024)
        try:
            width_str, height_str = shape.split("x")
            target_width = int(width_str)
            target_height = int(height_str)
        except ValueError:
            return None

        for _, row in data.iterrows():
            per_request_info = row.get("per_request_info")

            if per_request_info is not None:
                try:
                    if isinstance(per_request_info, str):
                        per_request_info = json.loads(per_request_info)

                    if isinstance(per_request_info, list):
                        for req_info in per_request_info:
                            if isinstance(req_info, dict):
                                # Check if this request matches the target resolution
                                req_width = req_info.get("width")
                                req_height = req_info.get("height")

                                if req_width == target_width and req_height == target_height:
                                    total_requests += 1
                                    # Check if SLO was met
                                    slo_met = req_info.get("slo_met")
                                    if slo_met:
                                        slo_met_requests += 1

                except (json.JSONDecodeError, TypeError, ValueError):
                    continue

        # Return SAR for this resolution
        if total_requests > 0:
            return slo_met_requests / total_requests
        else:
            # Fallback to aggregate if no per-request data for this resolution
            return data["slo_meet_ratio"].mean() if "slo_meet_ratio" in data.columns else None
