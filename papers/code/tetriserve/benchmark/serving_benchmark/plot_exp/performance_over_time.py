"""
Performance over time plotting module.
Creates time-series plots showing how latency and goodput change over time.
Similar to Clockwork Figure 6.
"""

from typing import Optional, Dict, List
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np
import json

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


class PerformanceOverTimePlotter:
    """Creates performance over time plots showing latency and goodput trends."""

    def __init__(self, config: PlotConfig = None):
        self.config = config or PlotConfig()

    def plot(self, df: pd.DataFrame, save_dir: str = None) -> None:
        """Plot performance over time for each experimental condition."""
        if df.empty:
            print("No data available, skipping performance over time plots")
            return

        # Check if we have time-series data
        if not self._has_time_series_data(df):
            print("No time-series data available, skipping performance over time plots")
            return

        # Get experimental conditions
        conditions = get_experimental_conditions(df)
        print(f"Generating performance over time plots for {len(conditions)} experimental conditions")

        for idx, condition in enumerate(conditions.itertuples(index=False)):
            condition_data = filter_by_condition(df, condition)

            if condition_data.empty:
                continue

            # Create plots for different SLO configurations
            self._create_performance_plots_by_slo(condition_data, condition, save_dir)

            # Progress indicator
            if (idx + 1) % 5 == 0 or idx == 0:
                print(f"Processed {idx + 1}/{len(conditions)} performance over time conditions")

    def _has_time_series_data(self, df: pd.DataFrame) -> bool:
        """Check if the dataframe contains time-series information."""
        # Look for time-related columns from benchmark results
        time_columns = ["request_starts", "request_finishes", "latency"]
        has_time_cols = any(col in df.columns for col in time_columns)

        # Check if per_request_info contains timestamps
        has_request_times = False
        if "per_request_info" in df.columns:
            for _, row in df.head().iterrows():
                per_request_info = row.get("per_request_info")
                if per_request_info is not None:
                    try:
                        if isinstance(per_request_info, str):
                            per_request_info = json.loads(per_request_info)
                        if isinstance(per_request_info, list) and len(per_request_info) > 0:
                            # Check if requests have timestamp information
                            first_req = per_request_info[0]
                            if isinstance(first_req, dict) and any(
                                key in first_req for key in ["timestamp", "start_time", "arrival_time"]
                            ):
                                has_request_times = True
                                break
                    except (json.JSONDecodeError, TypeError):
                        continue

        return has_time_cols or has_request_times

    def _create_performance_plots_by_slo(self, condition_data: pd.DataFrame, condition, save_dir: Optional[str]):
        """Create performance over time plots for each SLO configuration and arrival rate."""
        # Check if we have SLO config data
        if "slo_config" not in condition_data.columns or condition_data["slo_config"].isna().all():
            # Create plots without SLO grouping, but separate by rate
            rates = sorted(condition_data["rate"].unique())
            for rate in rates:
                rate_data = condition_data[condition_data["rate"] == rate]
                if not rate_data.empty:
                    self._create_performance_plots(rate_data, condition, save_dir, None, rate)
            return

        slo_configs = sorted(condition_data["slo_config"].dropna().unique())
        rates = sorted(condition_data["rate"].unique())

        # Create separate plots for each SLO config and arrival rate combination
        for slo_config in slo_configs:
            for rate in rates:
                slo_rate_data = condition_data[
                    (condition_data["slo_config"] == slo_config) & (condition_data["rate"] == rate)
                ]
                if not slo_rate_data.empty:
                    self._create_performance_plots(slo_rate_data, condition, save_dir, slo_config, rate)

    def _create_performance_plots(
        self, data: pd.DataFrame, condition, save_dir: Optional[str], slo_config: Optional[str], rate: float = None
    ):
        """Create the actual performance over time plots with both raw data and aggregate slo_meet_ratio approaches."""
        # Extract time-series data
        time_series_data = self._extract_time_series_data(data)

        if not time_series_data:
            print(f"No time-series data could be extracted for {slo_config or 'no SLO'}, rate {rate}")
            return

        # Get all scheduling algorithms for this condition
        schedule_logics = get_ordered_schedule_logics(data["schedule_logic"].unique())

        # Create subdirectories for different calculation approaches
        if save_dir:
            from pathlib import Path

            base_path = Path(save_dir)

            # Create main approach directories
            raw_data_dir = base_path / "raw_data_calculation"
            aggregate_dir = base_path / "aggregate_slo_meet_ratio"

            # Create subdirectories for raw data approach
            raw_latency_dir = raw_data_dir / "latency_over_time"
            raw_sliding_window_dir = raw_data_dir / "goodput_sliding_window"
            raw_cumulative_time_dir = raw_data_dir / "goodput_cumulative_time"
            raw_cumulative_requests_dir = raw_data_dir / "goodput_cumulative_requests"

            # Create subdirectories for aggregate approach
            agg_sliding_window_dir = aggregate_dir / "goodput_sliding_window"
            agg_cumulative_time_dir = aggregate_dir / "goodput_cumulative_time"
            agg_cumulative_requests_dir = aggregate_dir / "goodput_cumulative_requests"

            # Create all directories
            for dir_path in [
                raw_latency_dir,
                raw_sliding_window_dir,
                raw_cumulative_time_dir,
                raw_cumulative_requests_dir,
                agg_sliding_window_dir,
                agg_cumulative_time_dir,
                agg_cumulative_requests_dir,
            ]:
                dir_path.mkdir(parents=True, exist_ok=True)

            # Create plots using raw data calculation (per-request SLOs)
            self._create_combined_latency_over_time_plot(
                time_series_data, schedule_logics, condition, str(raw_latency_dir), slo_config, rate
            )

            self._create_combined_goodput_over_time_plot(
                time_series_data,
                schedule_logics,
                condition,
                str(raw_sliding_window_dir),
                slo_config,
                rate,
                method="sliding_window",
                use_raw_data=True,
            )

            self._create_combined_goodput_over_time_plot(
                time_series_data,
                schedule_logics,
                condition,
                str(raw_cumulative_time_dir),
                slo_config,
                rate,
                method="cumulative_time",
                use_raw_data=True,
            )

            self._create_combined_goodput_over_time_plot(
                time_series_data,
                schedule_logics,
                condition,
                str(raw_cumulative_requests_dir),
                slo_config,
                rate,
                method="cumulative_requests",
                use_raw_data=True,
            )

            # Create plots using aggregate slo_meet_ratio
            self._create_combined_goodput_over_time_plot(
                time_series_data,
                schedule_logics,
                condition,
                str(agg_sliding_window_dir),
                slo_config,
                rate,
                method="sliding_window",
                use_raw_data=False,
                data_df=data,
            )

            self._create_combined_goodput_over_time_plot(
                time_series_data,
                schedule_logics,
                condition,
                str(agg_cumulative_time_dir),
                slo_config,
                rate,
                method="cumulative_time",
                use_raw_data=False,
                data_df=data,
            )

            self._create_combined_goodput_over_time_plot(
                time_series_data,
                schedule_logics,
                condition,
                str(agg_cumulative_requests_dir),
                slo_config,
                rate,
                method="cumulative_requests",
                use_raw_data=False,
                data_df=data,
            )
        else:
            # Fallback to original behavior if no save_dir (use raw data approach)
            self._create_combined_latency_over_time_plot(
                time_series_data, schedule_logics, condition, save_dir, slo_config, rate
            )

            self._create_combined_goodput_over_time_plot(
                time_series_data, schedule_logics, condition, save_dir, slo_config, rate, use_raw_data=True
            )

    def _extract_time_series_data(self, data: pd.DataFrame) -> Dict:
        """Extract time-series data from benchmark results."""
        time_series = {}

        for _, row in data.iterrows():
            schedule_logic = row["schedule_logic"]
            if schedule_logic not in time_series:
                time_series[schedule_logic] = {
                    "timestamps": [],
                    "latencies": [],
                    "arrivals": [],
                    "completions": [],
                    "slos": [],  # Add per-request SLO data
                }

            # First try to extract from request_starts, request_finishes, latency, and per_request_info
            request_starts = row.get("request_starts")
            request_finishes = row.get("request_finishes")
            latency_data = row.get("latency")
            per_request_info = row.get("per_request_info")

            if request_starts is not None and latency_data is not None and per_request_info is not None:
                try:
                    # Handle different data formats
                    if isinstance(request_starts, str):
                        request_starts = json.loads(request_starts)
                    if isinstance(latency_data, str):
                        latency_data = json.loads(latency_data)
                    if isinstance(per_request_info, str):
                        per_request_info = json.loads(per_request_info)

                    # Convert to lists if they're numpy arrays or other formats
                    if hasattr(request_starts, "tolist"):
                        request_starts = request_starts.tolist()
                    if hasattr(latency_data, "tolist"):
                        latency_data = latency_data.tolist()

                    if (
                        isinstance(request_starts, list)
                        and isinstance(latency_data, list)
                        and isinstance(per_request_info, list)
                        and len(request_starts) == len(latency_data) == len(per_request_info)
                    ):

                        for start_time, latency, req_info in zip(request_starts, latency_data, per_request_info):
                            completion_time = start_time + latency
                            req_slo = req_info.get("slo", 3.0) if isinstance(req_info, dict) else 3.0

                            time_series[schedule_logic]["timestamps"].append(start_time)
                            time_series[schedule_logic]["latencies"].append(latency)
                            time_series[schedule_logic]["arrivals"].append(start_time)
                            time_series[schedule_logic]["completions"].append(completion_time)
                            time_series[schedule_logic]["slos"].append(req_slo)

                except (json.JSONDecodeError, TypeError, ValueError) as e:
                    print(f"Warning: Could not parse request_starts/latency/per_request_info data: {e}")
                    # Continue to try per_request_info method

            # Fallback: try to extract per-request information
            if not time_series[schedule_logic]["timestamps"]:
                per_request_info = row.get("per_request_info")

                if per_request_info is not None and latency_data is not None:
                    try:
                        if isinstance(per_request_info, str):
                            per_request_info = json.loads(per_request_info)
                        if isinstance(latency_data, str):
                            latency_data = json.loads(latency_data)

                        if isinstance(per_request_info, list) and isinstance(latency_data, list):
                            for req_info, latency in zip(per_request_info, latency_data):
                                if isinstance(req_info, dict):
                                    # Extract timestamp information
                                    arrival_time = req_info.get(
                                        "arrival_time", req_info.get("timestamp", req_info.get("start_time"))
                                    )
                                    req_slo = req_info.get("slo", 3.0)
                                    if arrival_time is not None:
                                        completion_time = arrival_time + latency

                                        time_series[schedule_logic]["timestamps"].append(arrival_time)
                                        time_series[schedule_logic]["latencies"].append(latency)
                                        time_series[schedule_logic]["arrivals"].append(arrival_time)
                                        time_series[schedule_logic]["completions"].append(completion_time)
                                        time_series[schedule_logic]["slos"].append(req_slo)

                    except (json.JSONDecodeError, TypeError) as e:
                        print(f"Warning: Could not parse per-request time-series data: {e}")

            # Skip if no real time-series data found
            if not time_series[schedule_logic]["timestamps"]:
                print(
                    f"No real time-series data found for {schedule_logic}, skipping performance over time plots for this scheduler"
                )

        return time_series

    def _create_combined_latency_over_time_plot(
        self,
        time_series_data: Dict,
        schedule_logics: List[str],
        condition,
        save_dir: Optional[str],
        slo_config: Optional[str],
        rate: float = None,
    ):
        """Create combined latency over time plot for all scheduling algorithms."""
        # Create the plot
        fig, ax = plt.subplots(figsize=self.config.figsize)

        # Get colors for all scheduling algorithms
        colors = get_colors_for_schedules(schedule_logics, self.config.color_scheme)

        # Plot each scheduling algorithm
        for i, schedule_logic in enumerate(schedule_logics):
            if schedule_logic not in time_series_data or not time_series_data[schedule_logic]["timestamps"]:
                continue

            data = time_series_data[schedule_logic]
            timestamps = np.array(data["timestamps"])
            latencies = np.array(data["latencies"])

            # Calculate latency over time (sliding window)
            window_duration = 30.0  # 30-second windows
            time_windows, latency_values = self._calculate_latency_over_time(timestamps, latencies, window_duration)

            if not time_windows:
                continue

            # Plot latency over time
            color = colors[i]
            ax.plot(
                time_windows,
                latency_values,
                linewidth=2,
                marker="o",
                markersize=4,
                color=color,
                label=f"{remap_schedule_logic_name(schedule_logic)}",
            )

        # Styling
        ax.set_xlabel("Time (minutes)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Latency (seconds)", fontsize=12, fontweight="bold")

        # Make tick labels bold
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight("bold")

        # Create title
        title_parts = ["Latency Over Time - All Scheduling Algorithms"]
        if slo_config:
            slo_ratio = slo_config.replace("custom_1_", "").replace("_slo", "")
            title_parts.append(f"SLO: {slo_ratio}")

        # Add arrival rate to title
        if rate is not None:
            title_parts.append(f"Arrival Rate: {rate*60:.1f} req/min")

        title_parts.append(generate_condition_string(condition))

        ax.set_title("\n".join(title_parts), fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        if save_dir:
            slo_str = f"_slo-{slo_config}" if slo_config else ""
            rate_str = f"_rate-{rate*60:.0f}rpm" if rate is not None else ""
            filename = f"latency_over_time_combined{slo_str}{rate_str}_{generate_save_path_suffix(condition, '')}.png"
            save_path = f"{save_dir}/{filename}"
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches="tight")
            print(f"Saved combined latency over time plot: {save_path}")

        plt.close()

    def _create_combined_goodput_over_time_plot(
        self,
        time_series_data: Dict,
        schedule_logics: List[str],
        condition,
        save_dir: Optional[str],
        slo_config: Optional[str],
        rate: float = None,
        method: str = "sliding_window",
        use_raw_data: bool = True,
        data_df: pd.DataFrame = None,
    ):
        """Create combined goodput over time plot for all scheduling algorithms.

        Args:
            use_raw_data: If True, calculate goodput from raw timing data. If False, use aggregate slo_meet_ratio.
            data_df: DataFrame containing aggregate statistics (required when use_raw_data=False)
        """
        # Create the plot
        fig, ax = plt.subplots(figsize=self.config.figsize)

        # Get colors for all scheduling algorithms
        colors = get_colors_for_schedules(schedule_logics, self.config.color_scheme)

        # Plot each scheduling algorithm
        for i, schedule_logic in enumerate(schedule_logics):
            if schedule_logic not in time_series_data or not time_series_data[schedule_logic]["timestamps"]:
                continue

            data = time_series_data[schedule_logic]
            timestamps = np.array(data["timestamps"])
            latencies = np.array(data["latencies"])
            completions = np.array(data["completions"])

            if use_raw_data:
                # Use raw data calculation with per-request SLOs
                slos = np.array(data["slos"])

                # Calculate goodput based on method
                if method == "sliding_window":
                    window_duration = 30.0  # 30-second windows
                    time_windows, goodput_values = self._calculate_goodput_over_time(
                        timestamps, latencies, completions, slos, window_duration, slo_config
                    )
                elif method == "cumulative_time":
                    time_windows, goodput_values = self._calculate_cumulative_goodput_by_time(
                        timestamps, latencies, completions, slos, slo_config
                    )
                elif method == "cumulative_requests":
                    time_windows, goodput_values = self._calculate_cumulative_goodput_by_requests(
                        timestamps, latencies, completions, slos, slo_config
                    )
                else:
                    print(f"Unknown method: {method}, falling back to sliding_window")
                    window_duration = 30.0
                    time_windows, goodput_values = self._calculate_goodput_over_time(
                        timestamps, latencies, completions, slos, window_duration, slo_config
                    )
            else:
                # Use aggregate slo_meet_ratio approach with real timing data
                # Get aggregate statistics for this algorithm
                alg_data = data_df[data_df["schedule_logic"] == schedule_logic]
                if alg_data.empty:
                    print(f"Warning: No aggregate stats for {schedule_logic}, skipping")
                    continue

                aggregate_slo_meet_ratio = alg_data.iloc[0].get("slo_meet_ratio", 0)
                aggregate_throughput = alg_data.iloc[0].get("throughput", 0)  # req/s

                # Calculate constant goodput using aggregate values
                aggregate_goodput = aggregate_throughput * aggregate_slo_meet_ratio * 60  # req/min

                # Use real timing data for x-axis
                time_windows, goodput_values = self._create_constant_goodput_series(
                    timestamps, completions, aggregate_goodput, method
                )

            if not time_windows:
                continue

            # Plot goodput over time
            color = colors[i]
            ax.plot(
                time_windows,
                goodput_values,
                linewidth=2,
                marker="o",
                markersize=4,
                color=color,
                label=f"{remap_schedule_logic_name(schedule_logic)}",
            )

        # Styling
        ax.set_xlabel("Time (minutes)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Goodput (req/min)", fontsize=12, fontweight="bold")

        # Make tick labels bold
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight("bold")

        # Create title based on method
        method_titles = {
            "sliding_window": "Goodput Over Time (30s Sliding Window)",
            "cumulative_time": "Cumulative Goodput Over Time",
            "cumulative_requests": "Cumulative Goodput by Request Count",
        }

        # Add calculation approach to title
        approach_suffix = " (Raw Data)" if use_raw_data else " (Aggregate SLO%)"
        title_parts = [
            method_titles.get(method, "Goodput Over Time") + approach_suffix + " - All Scheduling Algorithms"
        ]
        if slo_config:
            slo_ratio = slo_config.replace("custom_1_", "").replace("_slo", "")
            title_parts.append(f"SLO: {slo_ratio}")

        # Add arrival rate to title
        if rate is not None:
            title_parts.append(f"Arrival Rate: {rate*60:.1f} req/min")

        condition_str = generate_condition_string(condition)
        title_parts.append(condition_str)

        ax.set_title("\n".join(title_parts), fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot with method-specific filename
        if save_dir:
            slo_str = f"_slo-{slo_config}" if slo_config else ""
            rate_str = f"_rate-{rate*60:.0f}rpm" if rate is not None else ""
            filename_base = f"goodput_{method}_combined{slo_str}{rate_str}_{generate_save_path_suffix(condition, '')}"

            # Save PNG
            png_path = f"{save_dir}/{filename_base}.png"
            plt.savefig(png_path, dpi=self.config.dpi, bbox_inches="tight")
            print(f"Saved {method} goodput plot: {png_path}")

            # Save PDF
            pdf_path = f"{save_dir}/{filename_base}.pdf"
            plt.savefig(pdf_path, bbox_inches="tight")
            print(f"Saved {method} goodput plot (PDF): {pdf_path}")

        plt.close()

    def _create_latency_over_time_plot(
        self,
        time_series_data: Dict,
        schedule_logic: str,
        condition,
        save_dir: Optional[str],
        slo_config: Optional[str],
        rate: float = None,
    ):
        """Create latency over time plot for a specific scheduling algorithm."""
        if schedule_logic not in time_series_data or not time_series_data[schedule_logic]["timestamps"]:
            return

        data = time_series_data[schedule_logic]
        timestamps = np.array(data["timestamps"])
        latencies = np.array(data["latencies"])

        # Convert timestamps to minutes
        time_minutes = timestamps / 60.0

        # Create rolling average for smoother visualization
        window_size = max(10, len(timestamps) // 50)  # Adaptive window size
        rolling_latency = self._rolling_average(latencies, window_size)
        rolling_time = self._rolling_average(time_minutes, window_size)

        # Create the plot
        fig, ax = plt.subplots(figsize=self.config.figsize)

        # Plot raw data as scatter with low alpha
        ax.scatter(time_minutes, latencies, alpha=0.1, s=1, color="lightgray", label="Individual Requests")

        # Plot rolling average as main line
        color = get_colors_for_schedules([schedule_logic], self.config.color_scheme)[0]
        ax.plot(
            rolling_time,
            rolling_latency,
            linewidth=2,
            color=color,
            label=f"{remap_schedule_logic_name(schedule_logic)} (Avg)",
        )

        # Styling
        ax.set_xlabel("Time (minutes)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Latency (seconds)", fontsize=12, fontweight="bold")

        # Make tick labels bold
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight("bold")

        # Create title
        title_parts = [f"Latency Over Time - {remap_schedule_logic_name(schedule_logic)}"]
        if slo_config:
            slo_ratio = slo_config.replace("custom_1_", "").replace("_slo", "")
            title_parts.append(f"SLO: {slo_ratio}")

        # Add arrival rate to title
        if rate is not None:
            title_parts.append(f"Arrival Rate: {rate*60:.1f} req/min")

        title_parts.append(generate_condition_string(condition))

        ax.set_title("\n".join(title_parts), fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        if save_dir:
            slo_str = f"_slo-{slo_config}" if slo_config else ""
            rate_str = f"_rate-{rate*60:.0f}rpm" if rate is not None else ""
            filename = (
                f"latency_over_time_{schedule_logic}{slo_str}{rate_str}_{generate_save_path_suffix(condition, '')}.png"
            )
            save_path = f"{save_dir}/{filename}"
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches="tight")
            print(f"Saved latency over time plot: {save_path}")

        plt.close()

    def _create_goodput_over_time_plot(
        self,
        time_series_data: Dict,
        schedule_logic: str,
        condition,
        save_dir: Optional[str],
        slo_config: Optional[str],
        rate: float = None,
    ):
        """Create goodput over time plot for a specific scheduling algorithm."""
        if schedule_logic not in time_series_data or not time_series_data[schedule_logic]["timestamps"]:
            return

        data = time_series_data[schedule_logic]
        timestamps = np.array(data["timestamps"])
        latencies = np.array(data["latencies"])
        completions = np.array(data["completions"])
        slos = np.array(data["slos"])

        # Calculate goodput over time (sliding window)
        window_duration = 30.0  # 30-second windows
        time_windows, goodput_values = self._calculate_goodput_over_time(
            timestamps, latencies, completions, slos, window_duration, slo_config
        )

        if not time_windows:
            return

        # Create the plot
        fig, ax = plt.subplots(figsize=self.config.figsize)

        # Plot goodput over time
        color = get_colors_for_schedules([schedule_logic], self.config.color_scheme)[0]
        ax.plot(
            time_windows,
            goodput_values,
            linewidth=2,
            marker="o",
            markersize=4,
            color=color,
            label=f"{remap_schedule_logic_name(schedule_logic)}",
        )

        # Add average goodput line for reference
        avg_goodput = np.mean(goodput_values) if goodput_values else 0
        ax.axhline(y=avg_goodput, color=color, linestyle="--", alpha=0.5, label=f"Average: {avg_goodput:.1f} req/min")

        # Styling
        ax.set_xlabel("Time (minutes)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Goodput (req/min)", fontsize=12, fontweight="bold")

        # Make tick labels bold
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight("bold")

        # Create title with arrival rate information
        title_parts = [f"Goodput Over Time - {remap_schedule_logic_name(schedule_logic)}"]
        if slo_config:
            slo_ratio = slo_config.replace("custom_1_", "").replace("_slo", "")
            title_parts.append(f"SLO: {slo_ratio}")

        # Add arrival rate to title
        if rate is not None:
            title_parts.append(f"Arrival Rate: {rate*60:.1f} req/min")

        condition_str = generate_condition_string(condition)
        title_parts.append(condition_str)

        ax.set_title("\n".join(title_parts), fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        if save_dir:
            slo_str = f"_slo-{slo_config}" if slo_config else ""
            rate_str = f"_rate-{rate*60:.0f}rpm" if rate is not None else ""
            filename = (
                f"goodput_over_time_{schedule_logic}{slo_str}{rate_str}_{generate_save_path_suffix(condition, '')}.png"
            )
            save_path = f"{save_dir}/{filename}"
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches="tight")
            print(f"Saved goodput over time plot: {save_path}")

        plt.close()

    def _calculate_goodput_over_time(
        self,
        timestamps: np.ndarray,
        latencies: np.ndarray,
        completions: np.ndarray,
        slos: np.ndarray,
        window_duration: float,
        slo_config: Optional[str],
    ) -> tuple:
        """Calculate goodput over time using sliding windows with per-request SLOs."""
        if len(timestamps) == 0:
            return [], []

        # Create time windows
        max_time = max(np.max(timestamps), np.max(completions))
        time_windows = (
            np.arange(window_duration, max_time + window_duration, window_duration) / 60.0
        )  # Convert to minutes
        goodput_values = []

        for window_end in time_windows:
            window_start = window_end - window_duration / 60.0  # Convert to minutes

            # Find completions in this window
            window_mask = (completions >= window_start * 60) & (completions < window_end * 60)
            window_latencies = latencies[window_mask]
            window_slos = slos[window_mask]

            if len(window_latencies) == 0:
                goodput_values.append(0)
                continue

            # Calculate SLO meet ratio for this window using per-request SLOs (with 7% tolerance like workload.py)
            slo_meets_mask = (window_latencies <= window_slos) | (
                np.abs(window_latencies - window_slos) / window_slos <= 0.07
            )
            slo_meets = np.sum(slo_meets_mask)
            slo_meet_ratio = slo_meets / len(window_latencies)

            # Calculate throughput (completions per minute)
            throughput = len(window_latencies)  # Already per minute due to window_duration

            # Goodput = throughput * SLO meet ratio
            goodput = throughput * slo_meet_ratio
            goodput_values.append(goodput)

        return time_windows.tolist(), goodput_values

    def _calculate_latency_over_time(
        self, timestamps: np.ndarray, latencies: np.ndarray, window_duration: float
    ) -> tuple:
        """Calculate average latency over time using sliding windows."""
        if len(timestamps) == 0:
            return [], []

        # Create time windows
        max_time = np.max(timestamps)
        time_windows = (
            np.arange(window_duration, max_time + window_duration, window_duration) / 60.0
        )  # Convert to minutes
        latency_values = []

        for window_end in time_windows:
            window_start = window_end - window_duration / 60.0  # Convert to minutes

            # Find requests that started in this window
            window_mask = (timestamps >= window_start * 60) & (timestamps < window_end * 60)
            window_latencies = latencies[window_mask]

            if len(window_latencies) == 0:
                latency_values.append(0)
                continue

            # Calculate average latency for this window
            avg_latency = np.mean(window_latencies)
            latency_values.append(avg_latency)

        return time_windows.tolist(), latency_values

    def _calculate_cumulative_goodput_by_time(
        self,
        timestamps: np.ndarray,
        latencies: np.ndarray,
        completions: np.ndarray,
        slos: np.ndarray,
        slo_config: Optional[str],
    ) -> tuple:
        """Calculate cumulative goodput over time using fixed time intervals with per-request SLOs."""
        if len(timestamps) == 0:
            return [], []

        # Create time windows (every minute)
        max_time = np.max(completions)
        time_windows = np.arange(60, max_time + 60, 60) / 60.0  # Convert to minutes
        goodput_values = []

        for window_end_minutes in time_windows:
            window_end_seconds = window_end_minutes * 60

            # Find all requests completed by this time
            completed_mask = completions <= window_end_seconds
            completed_latencies = latencies[completed_mask]
            completed_slos = slos[completed_mask]

            if len(completed_latencies) == 0:
                goodput_values.append(0)
                continue

            # Count total SLO meets up to this time using per-request SLOs (with 7% tolerance like workload.py)
            slo_meets_mask = (completed_latencies <= completed_slos) | (
                np.abs(completed_latencies - completed_slos) / completed_slos <= 0.07
            )
            total_slo_meets = np.sum(slo_meets_mask)

            # Cumulative goodput = total SLO meets / elapsed time
            cumulative_goodput = total_slo_meets / window_end_minutes
            goodput_values.append(cumulative_goodput)

        return time_windows.tolist(), goodput_values

    def _calculate_cumulative_goodput_by_requests(
        self,
        timestamps: np.ndarray,
        latencies: np.ndarray,
        completions: np.ndarray,
        slos: np.ndarray,
        slo_config: Optional[str],
        request_interval: int = 50,
    ) -> tuple:
        """Calculate cumulative goodput using fixed request intervals with per-request SLOs."""
        if len(timestamps) == 0:
            return [], []

        # Sort requests by completion time
        sorted_indices = np.argsort(completions)
        sorted_completions = completions[sorted_indices]
        sorted_latencies = latencies[sorted_indices]
        sorted_slos = slos[sorted_indices]

        time_points = []
        goodput_values = []

        # Calculate goodput at fixed request intervals
        for i in range(request_interval, len(sorted_latencies) + 1, request_interval):
            # Consider first i completed requests
            first_i_latencies = sorted_latencies[:i]
            first_i_completions = sorted_completions[:i]
            first_i_slos = sorted_slos[:i]

            # Count requests meeting SLO using per-request SLOs (with 7% tolerance like workload.py)
            slo_meets_mask = (first_i_latencies <= first_i_slos) | (
                np.abs(first_i_latencies - first_i_slos) / first_i_slos <= 0.07
            )
            slo_meets = np.sum(slo_meets_mask)

            # Time elapsed = completion time of i-th request
            elapsed_time_minutes = first_i_completions[i - 1] / 60.0

            # Cumulative goodput = total SLO meets / elapsed time
            cumulative_goodput = slo_meets / elapsed_time_minutes

            time_points.append(elapsed_time_minutes)
            goodput_values.append(cumulative_goodput)

        return time_points, goodput_values

    def _create_constant_goodput_series(
        self, timestamps: np.ndarray, completions: np.ndarray, aggregate_goodput: float, method: str
    ) -> tuple:
        """Create goodput time series using constant aggregate goodput with real timing data.

        Args:
            timestamps: Request start times (real data)
            completions: Request completion times (real data)
            aggregate_goodput: Aggregate goodput value (req/min) - constant across time
            method: Calculation method ("sliding_window", "cumulative_time", "cumulative_requests")

        Returns:
            Tuple of (time_points, goodput_values) where goodput_values are constant
        """
        if len(timestamps) == 0:
            return [], []

        if method == "sliding_window":
            # Create sliding window time series with constant aggregate goodput
            window_duration = 30.0  # 30-second windows
            max_time = max(np.max(timestamps), np.max(completions))
            time_windows = (
                np.arange(window_duration, max_time + window_duration, window_duration) / 60.0
            )  # Convert to minutes
            goodput_values = [aggregate_goodput] * len(time_windows)  # Constant goodput
            return time_windows.tolist(), goodput_values

        elif method == "cumulative_time":
            # Create cumulative time series - goodput stays constant over time
            max_time = np.max(completions)
            time_windows = np.arange(60, max_time + 60, 60) / 60.0  # Convert to minutes
            goodput_values = [aggregate_goodput] * len(time_windows)  # Constant cumulative goodput
            return time_windows.tolist(), goodput_values

        elif method == "cumulative_requests":
            # Create request-based cumulative series
            sorted_completions = np.sort(completions)
            time_points = []
            goodput_values = []

            request_interval = 50
            for i in range(request_interval, len(sorted_completions) + 1, request_interval):
                elapsed_time_minutes = sorted_completions[i - 1] / 60.0
                time_points.append(elapsed_time_minutes)
                goodput_values.append(aggregate_goodput)  # Constant goodput

            return time_points, goodput_values

        else:
            # Default to sliding window
            return self._create_constant_goodput_series(timestamps, completions, aggregate_goodput, "sliding_window")

    def _get_slo_threshold(self, slo_config: Optional[str]) -> float:
        """Get SLO threshold from config name."""
        if not slo_config:
            return 3.0  # Default SLO threshold

        try:
            # Extract multiplier from SLO config name (e.g., "custom_1_1.3x_slo" -> 1.3)
            slo_parts = slo_config.replace("custom_1_", "").replace("_slo", "").replace("x", "")
            slo_multiplier = float(slo_parts)

            # Assume base SLO is 2.0 seconds (adjust based on your system)
            base_slo = 2.0
            return base_slo * slo_multiplier
        except (ValueError, AttributeError):
            return 3.0  # Default fallback

    def _rolling_average(self, data: np.ndarray, window_size: int) -> np.ndarray:
        """Calculate rolling average of data."""
        if len(data) < window_size:
            return data

        return np.convolve(data, np.ones(window_size) / window_size, mode="valid")
