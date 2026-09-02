"""
Main analyzer module for FluidServe experiment plotting.
Simplified version focusing on SLO analysis and performance over time.
"""

from pathlib import Path
from typing import List, Optional
import pandas as pd

from .config import PlotConfig
from .data_processor import DataProcessor
from .slo_meet_ratio import SLOMeetRatioPlotter
from .spider_slo_ratio import SpiderSLORatioPlotter
from .performance_over_time import PerformanceOverTimePlotter
from .academic_plot import AcademicPlotter


class ExperimentAnalyzer:
    """Main analyzer class that orchestrates data loading and plotting."""

    def __init__(self, results_file: str, exclude_schedules: List[str] = None):
        """
        Initialize the analyzer with benchmark results.

        Args:
            results_file: Path to benchmark results JSON file
            exclude_schedules: List of scheduling algorithms to exclude from analysis
        """
        # Load and process data
        self.processor = DataProcessor(results_file)
        self.df = self.processor.to_dataframe()
        self.df = self.processor.filter_schedules(self.df, exclude_schedules or [])

        # Validate data
        if self.df.empty:
            raise ValueError("No data remaining after filtering. Please check your exclude list.")

        # Print summary
        summary = self.processor.get_data_summary(self.df)
        print(f"\nAvailable models in data: {', '.join(summary['models'])}")
        print(f"Available scheduling algorithms in data: {', '.join(summary['schedules'])}")
        if summary["slo_configs"]:
            print(f"Available SLO configs in data: {', '.join(summary['slo_configs'])}")
        print(f"Available rates in data (req/s): {summary['rates']}")
        print(f"Total data points: {len(self.df)}")

    def plot_slo_meet_ratio_analysis(self, save_dir: str = None, plot_config: PlotConfig = None):
        """
        Plot SLO meet ratio vs SLO configuration bar charts.
        Creates both per-rate and averaged plots.

        Args:
            save_dir: Directory to save plots
            plot_config: Plot configuration
        """
        print("\nGenerating SLO meet ratio analysis...")
        plotter = SLOMeetRatioPlotter(plot_config)
        plotter.plot(self.df, save_dir)

    def plot_spider_slo_ratio_analysis(self, save_dir: str = None, plot_config: PlotConfig = None):
        """
        Plot spider/radar charts showing SLO meet ratio by resolution for different rates.

        Args:
            save_dir: Directory to save plots
            plot_config: Plot configuration
        """
        print("\nGenerating spider SLO ratio analysis...")
        plotter = SpiderSLORatioPlotter(plot_config)
        plotter.plot(self.df, save_dir)

    def plot_performance_over_time_analysis(self, save_dir: str = None, plot_config: PlotConfig = None):
        """
        Plot performance over time showing how latency and goodput change over time.
        Creates 8 figures (4 different methods * 2 metrics) similar to Clockwork Fig. 6.

        Args:
            save_dir: Directory to save plots
            plot_config: Plot configuration
        """
        print("\nGenerating performance over time analysis...")
        plotter = PerformanceOverTimePlotter(plot_config)
        plotter.plot(self.df, save_dir)

    def plot_academic_figures(self, save_dir: str = None, plot_config: PlotConfig = None):
        """
        Generate publication-ready academic plots with consistent styling.

        Args:
            save_dir: Directory to save plots
            plot_config: Plot configuration
        """
        print("\nGenerating academic publication-ready plots...")
        plotter = AcademicPlotter(plot_config)
        plotter.plot_all_academic_figures(self.df, save_dir)

    def run_analysis(self, output_dir: str, plot_config: PlotConfig = None):
        """
        Run the complete analysis pipeline (simplified version).

        Args:
            output_dir: Base directory for saving all plots
            plot_config: Plot configuration
        """
        print(f"\n{'='*60}")
        print("RUNNING FLUIDSERVE EXPERIMENT ANALYSIS (SIMPLIFIED)")
        print(f"{'='*60}")

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Create subdirectories for remaining plot types
        slo_meet_dir = output_path / "slo_meet_ratio_analysis"
        spider_slo_dir = output_path / "spider_slo_ratio_analysis"
        performance_time_dir = output_path / "performance_over_time"

        slo_meet_dir.mkdir(exist_ok=True)
        spider_slo_dir.mkdir(exist_ok=True)
        performance_time_dir.mkdir(exist_ok=True)

        # Run analyses
        self.plot_slo_meet_ratio_analysis(str(slo_meet_dir), plot_config)
        self.plot_spider_slo_ratio_analysis(str(spider_slo_dir), plot_config)
        self.plot_performance_over_time_analysis(str(performance_time_dir), plot_config)

        print(f"\n{'='*60}")
        print("ANALYSIS COMPLETE")
        print(f"{'='*60}")
        print(f"\nResults saved to: {output_path}")
        print("Directory structure:")
        print(f"├── slo_meet_ratio_analysis/   - SLO meet ratio analysis (uses aggregate statistics)")
        print(f"│   ├── *_per_rate_*.png                  - Bar charts of SLO meet ratio vs rate for each SLO")
        print(f"│   ├── *_merged_avg_*.png                - Grouped bar chart comparing average SLO meet ratio")
        print(f"│   └── *_merged_avg_line_*.png           - Line chart comparing average SLO meet ratio")
        print(f"├── spider_slo_ratio_analysis/ - Spider/radar plots (uses aggregate statistics)")
        print(f"│   ├── *_spider_slo_ratio_by_resolution_*.png - SLO meet ratio by resolution")
        print(f"│   └── *_spider_slo_ratio_summary_*.png       - Summary plots (2x2 layout)")
        print(f"└── performance_over_time/     - Temporal analysis (shows variations over time)")
        print(f"    ├── raw_data_calculation/      - Per-window SLO calculation (temporal variations)")
        print(f"    └── aggregate_slo_meet_ratio/  - Constant aggregate goodput (baseline comparison)")
        print(f"\nTip: Use --plot-types academic to generate publication-ready plots with enhanced styling!")

    def get_dataframe(self) -> pd.DataFrame:
        """Get the processed dataframe for custom analysis."""
        return self.df.copy()
