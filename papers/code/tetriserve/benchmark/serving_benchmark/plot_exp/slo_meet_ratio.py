"""
SLO meet ratio vs SLO configuration plotting module.
Creates bar charts showing per-rate and averaged SLO meet ratio performance.
"""

from typing import Optional, Dict, List
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np

from .config import (
    PlotConfig,
    get_ordered_schedule_logics,
    remap_schedule_logic_name,
    get_colors_for_schedules,
    PLOT_PARAMS,
)
from .data_processor import (
    get_experimental_conditions,
    filter_by_condition,
    generate_condition_string,
    generate_save_path_suffix,
)


class SLOMeetRatioPlotter:
    """Creates SLO meet ratio vs SLO configuration bar charts."""

    def __init__(self, config: PlotConfig = None):
        self.config = config or PlotConfig()

    def plot(self, df: pd.DataFrame, save_dir: str = None) -> None:
        """Plot SLO meet ratio vs SLO configurations (uses aggregate statistics)."""
        if df.empty:
            print("No data available, skipping SLO meet ratio plot")
            return

        # Check if we have SLO config data
        if "slo_config" not in df.columns or df["slo_config"].isna().all():
            print("No SLO config data available, skipping SLO meet ratio plot")
            return

        # Create output directory
        if save_dir:
            from pathlib import Path

            Path(save_dir).mkdir(parents=True, exist_ok=True)

        # Get experimental conditions (excluding SLO config)
        conditions = get_experimental_conditions(df)
        print(f"Generating SLO meet ratio plots for {len(conditions)} experimental conditions")

        for idx, condition in enumerate(conditions.itertuples(index=False)):
            condition_data = filter_by_condition(df, condition)

            if condition_data.empty:
                continue

            # Create plots using aggregate approach (verified to match raw data)
            self._create_per_rate_plots(condition_data, condition, save_dir)
            self._create_merged_avg_slo_meet_ratio_plot(condition_data, condition, save_dir)
            self._create_merged_avg_slo_meet_ratio_line_plot(condition_data, condition, save_dir)

            # Progress indicator
            if (idx + 1) % 5 == 0 or idx == 0:
                print(f"Processed {idx + 1}/{len(conditions)} SLO meet ratio conditions")

    def _calculate_slo_meet_ratio(self, data: pd.DataFrame) -> float:
        """Calculate SLO meet ratio from aggregate statistics.

        Args:
            data: DataFrame containing the data for calculation

        Returns:
            Average SLO meet ratio value
        """
        return data["slo_meet_ratio"].mean()

    def _create_per_rate_plots(self, condition_data: pd.DataFrame, condition, save_dir: Optional[str]):
        """Create SLO meet ratio plots for each SLO configuration."""
        rates = sorted(condition_data["rate"].unique())
        slo_configs = sorted(condition_data["slo_config"].dropna().unique())

        if len(slo_configs) < 2:
            print(f"Not enough SLO configs ({len(slo_configs)}) for SLO meet ratio plot, skipping")
            return

        # Create subplots for all SLO configs
        n_slos = len(slo_configs)
        if n_slos == 1:
            fig, axes = plt.subplots(1, 1, figsize=self.config.figsize)
            axes = [axes]
        elif n_slos == 2:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        elif n_slos <= 4:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            axes = axes.flatten()
        else:
            rows = int(np.ceil(n_slos / 3))
            fig, axes = plt.subplots(rows, 3, figsize=(18, 6 * rows))
            axes = axes.flatten()

        # Plot for each SLO config
        for i, slo_config in enumerate(slo_configs):
            ax = axes[i]
            slo_data = condition_data[condition_data["slo_config"] == slo_config]

            if not slo_data.empty:
                self._plot_slo_meet_ratio_for_slo(ax, slo_data, slo_config, rates)

        # Hide unused subplots
        for i in range(n_slos, len(axes)):
            axes[i].set_visible(False)

        # Add overall title
        condition_str = generate_condition_string(condition)
        fig.suptitle(
            f"SLO Meet Ratio vs Arrival Rate (Per SLO Configuration)\n{condition_str}",
            fontsize=16,
            fontweight="bold",
            y=0.98,
        )

        plt.tight_layout()

        # Save plot
        if save_dir:
            filename = f"slo_meet_ratio_per_rate_{generate_save_path_suffix(condition, '')}.png"
            save_path = f"{save_dir}/{filename}"
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches="tight")
            print(f"Saved per-rate SLO meet ratio plot: {save_path}")

        plt.close()

    def _plot_slo_meet_ratio_for_slo(self, ax, slo_data: pd.DataFrame, slo_config: str, rates: List[float]):
        """Plot SLO meet ratio vs rate for a specific SLO configuration using bar charts."""
        schedule_logics = get_ordered_schedule_logics(slo_data["schedule_logic"].unique())
        colors = get_colors_for_schedules(schedule_logics, self.config.color_scheme)

        # Calculate SLO meet ratio for each rate and scheduler
        slo_meet_data = {}
        for schedule_logic in schedule_logics:
            slo_meet_values = []
            for rate in rates:
                logic_rate_data = slo_data[
                    (slo_data["schedule_logic"] == schedule_logic) & (abs(slo_data["rate"] - rate) < 0.01)
                ]

                if not logic_rate_data.empty:
                    slo_meet_ratio = self._calculate_slo_meet_ratio(logic_rate_data)
                    slo_meet_values.append(slo_meet_ratio)
                else:
                    slo_meet_values.append(0)

            slo_meet_data[schedule_logic] = slo_meet_values

        # Create bar chart
        x = np.arange(len(rates))
        width = 0.8 / len(schedule_logics)

        # Plot bars for each scheduling algorithm
        for i, (schedule_logic, color) in enumerate(zip(schedule_logics, colors)):
            values = slo_meet_data[schedule_logic]
            offset = (i - len(schedule_logics) / 2 + 0.5) * width

            bars = ax.bar(
                x + offset,
                values,
                width,
                label=remap_schedule_logic_name(schedule_logic),
                color=color,
                alpha=PLOT_PARAMS["alpha"],
            )

            # Add value labels on bars
            for bar, val in zip(bars, values):
                if val > 0.01:  # Only show if value is significant
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01,
                        f"{val:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )

        # Styling
        ax.set_xlabel("Arrival Rate (req/min)", fontsize=10)
        ax.set_ylabel("SLO Meet Ratio", fontsize=10, fontweight="bold")
        ax.set_title(f"SLO Configuration: {slo_config}", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{r*60:.0f}" for r in rates], fontweight="bold")  # Convert to req/min
        # Make y-tick labels bold
        for label in ax.get_yticklabels():
            label.set_fontweight("bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=PLOT_PARAMS["grid_alpha"], axis="y")
        ax.set_ylim(0, 1.05)  # SLO meet ratio is between 0 and 1

        # Add reference line at 0.95 if configured
        if self.config.show_reference_line:
            ax.axhline(
                y=self.config.reference_value,
                color="red",
                linestyle="--",
                alpha=0.5,
                label=f"{self.config.reference_value:.0%} SLO",
            )

    def _create_merged_avg_slo_meet_ratio_line_plot(
        self, condition_data: pd.DataFrame, condition, save_dir: Optional[str]
    ):
        """Create merged line plot showing average SLO meet ratio for all SLO configs in one chart."""
        rates = sorted(condition_data["rate"].unique())
        slo_configs = sorted(condition_data["slo_config"].dropna().unique())

        if len(slo_configs) < 2:
            print(f"Not enough SLO configs ({len(slo_configs)}) for merged line plot, skipping")
            return

        # Calculate average SLO meet ratio for each scheduler and SLO config
        plot_data = {}
        schedule_logics = get_ordered_schedule_logics(condition_data["schedule_logic"].unique())

        for schedule_logic in schedule_logics:
            plot_data[schedule_logic] = []

            for slo_config in slo_configs:
                slo_data = condition_data[condition_data["slo_config"] == slo_config]
                slo_meet_values = []

                # Calculate SLO meet ratio for each rate
                for rate in rates:
                    rate_data = slo_data[
                        (slo_data["schedule_logic"] == schedule_logic) & (abs(slo_data["rate"] - rate) < 0.01)
                    ]

                    if not rate_data.empty:
                        slo_meet_ratio = self._calculate_slo_meet_ratio(rate_data)
                        slo_meet_values.append(slo_meet_ratio)

                # Calculate average across all rates
                avg_slo_meet = np.mean(slo_meet_values) if slo_meet_values else 0
                plot_data[schedule_logic].append(avg_slo_meet)

        # Create the line plot with enhanced academic paper style
        fig, ax = plt.subplots(figsize=(10, 6))

        # Extract SLO ratios for display and sort from high to low (e.g., "custom_1_1.3x_slo" -> "1.3x")
        slo_ratio_pairs = []
        for slo in slo_configs:
            # Extract ratio from name (e.g., "custom_1_1.3x_slo_H100_flux" -> "1.3x")
            # Split by underscore and find the part that looks like a ratio (ends with 'x' or is numeric)
            parts = slo.split("_")
            ratio = None
            for i, part in enumerate(parts):
                if "." in part or part.replace(".", "").replace("x", "").isdigit():
                    # Found the numeric part - might be "1.3x" or just "1.3"
                    if i + 1 < len(parts) and parts[i + 1].startswith("slo"):
                        # This is the ratio before "_slo"
                        ratio = part
                        break

            if ratio is None:
                ratio = "1.0x"  # Default fallback

            # Convert to float for sorting (e.g., "1.3x" -> 1.3)
            ratio_value = float(ratio.rstrip("x"))
            slo_ratio_pairs.append((ratio_value, ratio, slo))

        # Sort from high to low (1.5x, 1.4x, 1.3x, 1.2x, 1.1x, 1.0x)
        slo_ratio_pairs.sort(reverse=True)
        slo_ratios = [pair[1] for pair in slo_ratio_pairs]
        slo_configs_sorted = [pair[2] for pair in slo_ratio_pairs]

        # Define academic colors with FluidSP highlighted and xDiT variants in attractive dark colors
        academic_colors = []
        for schedule_logic in schedule_logics:
            if "dyn_slo_schedule_global" in schedule_logic:  # FluidSP(ours)
                academic_colors.append("#FF6600")  # Bright orange for our method
            elif "data_parallel" in schedule_logic:  # xDiT(SP=1)
                academic_colors.append("#1f4e79")  # Dark blue
            elif "fixed_sp_2" in schedule_logic:  # xDiT(SP=2)
                academic_colors.append("#2d5016")  # Dark green
            elif "fixed_sp_4" in schedule_logic:  # xDiT(SP=4)
                academic_colors.append("#c55a5a")  # Light red
            elif "fixed_sp_8" in schedule_logic:  # xDiT(SP=8)
                academic_colors.append("#833c0c")  # Dark brown
            else:
                # Fallback colors for any other schedulers
                academic_colors.append("#0066CC")  # Blue

        # Reorder plot data according to sorted SLO configs
        plot_data_sorted = {}
        for schedule_logic in schedule_logics:
            plot_data_sorted[schedule_logic] = []
            for slo_config in slo_configs_sorted:
                # Find the index of this slo_config in the original order
                orig_index = slo_configs.index(slo_config)
                plot_data_sorted[schedule_logic].append(plot_data[schedule_logic][orig_index])

        # Plot lines for each scheduler with clean academic styling
        x = np.arange(len(slo_configs_sorted))
        for i, schedule_logic in enumerate(schedule_logics):
            ax.plot(
                x,
                plot_data_sorted[schedule_logic],
                marker="^",
                markersize=7,
                linewidth=2.5,
                linestyle="-",  # Use solid lines only for clean academic look
                label=remap_schedule_logic_name(schedule_logic),
                color=academic_colors[i],
                markerfacecolor=academic_colors[i],
                markeredgecolor="white",
                markeredgewidth=1,
                alpha=0.9,
            )

        # Enhanced academic paper styling
        ax.set_xlabel("SLO Ratio", fontsize=14, fontweight="bold")
        ax.set_ylabel("Average SLO Meet Ratio", fontsize=14, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(slo_ratios, fontsize=12, fontweight="bold")

        # Improve tick styling
        ax.tick_params(axis="both", which="major", labelsize=12, width=1.5, length=6)
        ax.tick_params(axis="both", which="minor", width=1, length=3)
        # Make y-tick labels bold
        for label in ax.get_yticklabels():
            label.set_fontweight("bold")

        # Add subtle grid lines
        ax.grid(True, axis="y", linestyle="-", alpha=0.3, linewidth=0.8, color="gray")
        ax.grid(True, axis="x", linestyle="-", alpha=0.2, linewidth=0.5, color="gray")
        ax.set_axisbelow(True)

        # Set y-axis limits with better spacing
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))

        # Enhanced legend styling
        legend = ax.legend(
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            fontsize=11,
            frameon=True,
            fancybox=False,
            edgecolor="black",
            framealpha=1.0,
            shadow=False,
        )
        legend.get_frame().set_linewidth(1.5)

        # Professional spine styling
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
            spine.set_color("black")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Add reference line at 0.95 if configured with better styling
        if self.config.show_reference_line:
            ax.axhline(y=self.config.reference_value, color="red", linestyle="--", alpha=0.7, linewidth=2, zorder=0)
            # Add reference line label
            ax.text(
                len(slo_configs) - 0.1,
                self.config.reference_value + 0.02,
                f"{self.config.reference_value:.0%} SLO Target",
                ha="right",
                va="bottom",
                fontsize=10,
                color="red",
                fontweight="bold",
            )

        # Set background color to pure white
        ax.set_facecolor("white")
        fig.patch.set_facecolor("white")

        # No title for academic paper style - information goes in caption

        # Adjust layout to accommodate legend outside the plot
        plt.tight_layout()

        # Save plot
        if save_dir:
            filename = f"slo_meet_ratio_merged_avg_line_{generate_save_path_suffix(condition, '')}.png"
            save_path = f"{save_dir}/{filename}"
            # bbox_inches="tight" ensures the legend outside the plot is included
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches="tight")
            print(f"Saved merged average SLO meet ratio line plot: {save_path}")

        plt.close()

    def _create_merged_avg_slo_meet_ratio_plot(self, condition_data: pd.DataFrame, condition, save_dir: Optional[str]):
        """Create merged plot showing average SLO meet ratio for all SLO configs in one chart."""
        rates = sorted(condition_data["rate"].unique())
        slo_configs = sorted(condition_data["slo_config"].dropna().unique())

        if len(slo_configs) < 2:
            print(f"Not enough SLO configs ({len(slo_configs)}) for merged plot, skipping")
            return

        # Calculate average SLO meet ratio for each scheduler and SLO config
        plot_data = {}
        schedule_logics = get_ordered_schedule_logics(condition_data["schedule_logic"].unique())

        for schedule_logic in schedule_logics:
            plot_data[schedule_logic] = []

            for slo_config in slo_configs:
                slo_data = condition_data[condition_data["slo_config"] == slo_config]
                slo_meet_values = []

                # Calculate SLO meet ratio for each rate
                for rate in rates:
                    rate_data = slo_data[
                        (slo_data["schedule_logic"] == schedule_logic) & (abs(slo_data["rate"] - rate) < 0.01)
                    ]

                    if not rate_data.empty:
                        slo_meet_ratio = self._calculate_slo_meet_ratio(rate_data)
                        slo_meet_values.append(slo_meet_ratio)

                # Calculate average across all rates
                avg_slo_meet = np.mean(slo_meet_values) if slo_meet_values else 0
                plot_data[schedule_logic].append(avg_slo_meet)

        # Create the grouped bar chart with academic paper style
        fig, ax = plt.subplots(figsize=(8, 6))

        # Setup bar positions
        x = np.arange(len(slo_configs))
        n_groups = len(schedule_logics)
        width = 0.8 / n_groups  # Total width divided by number of bars
        multiplier = 0

        # Define academic paper style colors
        # Orange for first, light blue for second, grays for others
        academic_colors = ["#FF8C00", "#87CEEB", "#A9A9A9", "#696969", "#2F4F4F"]
        if len(schedule_logics) > len(academic_colors):
            # Add more gray shades if needed
            academic_colors.extend(["#808080"] * (len(schedule_logics) - len(academic_colors)))

        # Plot bars for each scheduler (NO value labels)
        for i, schedule_logic in enumerate(schedule_logics):
            offset = width * multiplier - width * n_groups / 2 + width / 2
            bars = ax.bar(
                x + offset,
                plot_data[schedule_logic],
                width,
                label=remap_schedule_logic_name(schedule_logic),
                color=academic_colors[i],
                edgecolor="black",
                linewidth=0.5,
            )

            # Add value labels on bars
            for bar, val in zip(bars, plot_data[schedule_logic]):
                if val > 0.01:  # Show label if value is significant
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01,
                        f"{val:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )

            multiplier += 1

        # Academic paper styling
        ax.set_xlabel("SLO Scale", fontsize=14, fontweight="bold")
        ax.set_ylabel("SLO Attainment Ratio", fontsize=14, fontweight="bold")

        # Extract SLO ratios for display (e.g., "custom_1_1.3x_slo_H100_flux" -> "1.3x")
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
        ax.set_xticklabels(slo_ratios, fontsize=12, fontweight="bold")

        # Improve tick styling
        ax.tick_params(axis="both", which="major", labelsize=12, width=1.5, length=6)
        # Make y-tick labels bold
        for label in ax.get_yticklabels():
            label.set_fontweight("bold")

        # Add dotted grid lines like in the paper
        ax.grid(True, axis="y", linestyle=":", alpha=0.7, linewidth=1)
        ax.set_axisbelow(True)

        # Set y-axis limits
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

        # Add legend on top (centered above the plot)
        ax.legend(
            bbox_to_anchor=(0.5, 1.18),
            loc="upper center",
            ncol=len(schedule_logics),  # Horizontal layout
            fontsize=12,
            frameon=True,
            fancybox=False,
            edgecolor="black",
            framealpha=1.0,
            shadow=False,
        )

        # Remove top and right spines for cleaner look
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Add reference line at 0.95 if configured
        if self.config.show_reference_line:
            ax.axhline(y=self.config.reference_value, color="red", linestyle="--", alpha=0.5, linewidth=1.5)

        # No title for academic paper style - information goes in caption

        # Adjust layout to accommodate legend on top
        plt.tight_layout()

        # Save plot
        if save_dir:
            filename = f"slo_meet_ratio_merged_avg_{generate_save_path_suffix(condition, '')}.png"
            save_path = f"{save_dir}/{filename}"
            # bbox_inches="tight" ensures the legend above the plot is included
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches="tight")
            print(f"Saved merged average SLO meet ratio plot: {save_path}")

        plt.close()
