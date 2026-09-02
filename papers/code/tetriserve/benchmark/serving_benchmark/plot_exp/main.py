#!/usr/bin/env python3
"""
Main entry point for FluidServe experiment plotting.

Usage:
    python -m benchmark.serving.plot_exp.main results.json --output-dir analysis_output
"""

import argparse
from pathlib import Path

from .analyzer import ExperimentAnalyzer
from .config import PlotConfig


def main():
    parser = argparse.ArgumentParser(description="Analyze FluidServe serving benchmark results (simplified version)")
    parser.add_argument("results_file", type=str, help="Path to benchmark results JSON file")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark/serving/plot_exp_results",
        help="Directory to save analysis outputs",
    )
    parser.add_argument(
        "--exclude-schedules", nargs="+", default=None, help="List of scheduling algorithms to exclude from analysis"
    )
    parser.add_argument(
        "--no-reference-line", action="store_true", help="Do not show the 95%% reference line in SLO plots"
    )
    parser.add_argument("--figsize", nargs=2, type=int, default=[10, 6], help="Figure size (width height)")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for saved figures")
    parser.add_argument(
        "--color-scheme",
        type=str,
        default="academic",
        help="Color scheme: 'academic' for paper style or matplotlib colormap name",
    )
    parser.add_argument(
        "--plot-types",
        nargs="+",
        default=["all"],
        choices=["all", "slo_meet", "spider_slo", "performance_time", "academic"],
        help="Types of plots to generate (default: all). Use 'academic' for publication-ready plots",
    )

    args = parser.parse_args()

    # Configure plotting
    plot_config = PlotConfig(
        figsize=tuple(args.figsize),
        show_reference_line=not args.no_reference_line,
        dpi=args.dpi,
        color_scheme=args.color_scheme,
        use_academic_style=True,  # Always use academic style
    )

    try:
        # Initialize analyzer
        analyzer = ExperimentAnalyzer(args.results_file, exclude_schedules=args.exclude_schedules)

        # Run analysis based on selected plot types
        if "all" in args.plot_types:
            analyzer.run_analysis(args.output_dir, plot_config)
        else:
            # Run individual analyses based on selected types
            from pathlib import Path

            output_path = Path(args.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            if "slo_meet" in args.plot_types:
                slo_meet_dir = output_path / "slo_meet_ratio_analysis"
                slo_meet_dir.mkdir(exist_ok=True)
                analyzer.plot_slo_meet_ratio_analysis(str(slo_meet_dir), plot_config)

            if "spider_slo" in args.plot_types:
                spider_slo_dir = output_path / "spider_slo_ratio_analysis"
                spider_slo_dir.mkdir(exist_ok=True)
                analyzer.plot_spider_slo_ratio_analysis(str(spider_slo_dir), plot_config)

            if "performance_time" in args.plot_types:
                performance_time_dir = output_path / "performance_over_time"
                performance_time_dir.mkdir(exist_ok=True)
                analyzer.plot_performance_over_time_analysis(str(performance_time_dir), plot_config)

            if "academic" in args.plot_types:
                academic_dir = output_path / "academic_plots"
                academic_dir.mkdir(exist_ok=True)
                analyzer.plot_academic_figures(str(academic_dir), plot_config)

    except Exception as e:
        print(f"\nError: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
