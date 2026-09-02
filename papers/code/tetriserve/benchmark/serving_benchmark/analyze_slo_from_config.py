#!/usr/bin/env python3
"""
SLO Analysis Using Configuration Files

This script calculates SLO attendance for benchmark results using specific SLO configuration files
(like std_0.7x_slo.json) instead of uniform multipliers. This is more realistic since different
request shapes often have different SLO requirements.

Usage:
    python analyze_slo_from_config.py results.json --slo-configs std_0.7x_slo std_1x_slo std_1.2x_slo
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
from datetime import datetime
import copy
import sys
import os

# Add the parent directory to path to import from tetriserve
sys.path.append(str(Path(__file__).parent.parent.parent))

from tetriserve.utils import get_gpu_name


def load_slo_config(slo_config_name: str) -> Dict[str, Union[str, float, int]]:
    """Load SLO configuration from JSON file.

    Returns:
        Dict mapping "(bs, height, width)" to SLO values (either "2.0x" or 15.0)
    """
    script_dir = Path(__file__).parent
    slo_config_file = script_dir / "slo_configs" / f"{slo_config_name}.json"

    if not slo_config_file.exists():
        raise FileNotFoundError(f"SLO config file not found: {slo_config_file}")

    with open(slo_config_file, "r") as f:
        slo_mapping = json.load(f)

    # Remove comments and validate format
    clean_mapping = {}
    for key, value in slo_mapping.items():
        if key.startswith("_"):  # Skip comments
            continue
        clean_mapping[key] = value

    print(f"Loaded SLO config '{slo_config_name}' with {len(clean_mapping)} entries")
    return clean_mapping


def load_baseline_slo_dict(model_name: str = "sd3", steps: int = 20) -> Dict[Tuple[int, int, int], float]:
    """Load baseline SLO values from scaling efficiency data.

    Returns:
        Dict mapping (bs, height, width) to baseline SLO values in seconds
    """
    base_path = Path("log/benchmark/scaling_efficiency")

    # Include GPU name in path
    try:
        gpu_name = get_gpu_name()
    except Exception as e:
        print(f"Warning: Could not get GPU name, using default: {e}")
        gpu_name = "H100_HBM3"  # Reasonable default

    model_path = base_path / model_name / gpu_name / f"e2e_scaling_efficiency_1.json"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Baseline scaling efficiency data not found: {model_path}\n"
            f"Please run scaling efficiency benchmark first for model {model_name}"
        )

    with open(model_path, "r") as f:
        data = json.load(f)

    baselines = {}
    for entry in data:
        bs = entry["bs"]
        height = entry["height"]
        width = entry["width"]
        entry_steps = entry["steps"]
        avg_e2e_time = entry["avg_e2e_time"]

        # Scale to target steps (same as workload.py)
        total_time = avg_e2e_time * (steps / entry_steps)

        key = (bs, height, width)
        baselines[key] = total_time

    print(f"Loaded {len(baselines)} baseline SLO values for {model_name} (scaled to {steps} steps)")
    return baselines


def parse_slo_value(slo_config_value: Union[str, float, int], baseline_slo: float) -> float:
    """Parse SLO configuration value into final SLO.

    Returns:
        Final SLO value in seconds
    """
    if isinstance(slo_config_value, str) and slo_config_value.endswith("x"):
        # Multiplier format: "2.0x"
        try:
            multiplier = float(slo_config_value[:-1])
            return baseline_slo * multiplier
        except ValueError:
            raise ValueError(f"Invalid multiplier format: '{slo_config_value}'. Use format like '2.0x'")

    elif isinstance(slo_config_value, (int, float)):
        # Fixed value format: 15.0
        return float(slo_config_value)

    else:
        raise ValueError(f"Invalid SLO format: '{slo_config_value}'. Use either '2.0x' (multiplier) or 15.0 (fixed)")


def calculate_slo_attendance_from_config(
    result: Dict,
    slo_config: Dict[str, Union[str, float, int]],
    baseline_slo_dict: Dict[Tuple[int, int, int], float],
) -> Tuple[float, List[Dict]]:
    """Calculate SLO attendance using a specific SLO configuration file.

    Args:
        result: Benchmark result containing per_request_info with latencies
        slo_config: SLO configuration mapping (bs, h, w) -> SLO requirement
        baseline_slo_dict: Baseline latencies for scaling multiplier-based SLOs (optional for fixed values)

    Returns:
        Tuple of (slo_meet_ratio, updated_per_request_info) where per_request_info has new slo and slo_met values
    """
    stats = result.get("stats", {})
    per_request_info = stats.get("per_request_info", [])

    if not per_request_info:
        # Fallback to old format with separate arrays
        latencies = stats.get("latency", [])
        if not latencies:
            return 1.0
        # Will reconstruct from latency array below (old code path)
    else:
        # Use per_request_info (new format) - already has latency
        latencies = [req["latency"] for req in per_request_info]

    if not latencies:
        return 1.0

    slo_met_count = 0
    detailed_info = []  # For debugging
    updated_per_request_info = []  # Updated with new SLO values

    # Process each request
    for i, req_info in enumerate(per_request_info):
        # Extract request details
        latency = req_info.get("latency")
        if latency is None:
            continue

        bs = req_info["batch_size"]
        height = req_info["height"]
        width = req_info["width"]

        # Look up the SLO config for this request shape
        slo_key = f"({bs}, {height}, {width})"

        if slo_key not in slo_config:
            # Try fallback - look for any config with matching height/width (different batch size)
            fallback_key = None
            for key in slo_config.keys():
                if f", {height}, {width})" in key:
                    fallback_key = key
                    break

            if fallback_key:
                slo_config_value = slo_config[fallback_key]
            else:
                raise ValueError(f"No SLO config found for {slo_key}. Available keys: {list(slo_config.keys())}")
        else:
            slo_config_value = slo_config[slo_key]

        # Get baseline SLO for multiplier-based configs (if needed)
        baseline_key = (bs, height, width)
        baseline_slo = baseline_slo_dict.get(baseline_key, 10.0)  # Default if not found

        # Parse the SLO config to get the target SLO
        target_slo = parse_slo_value(slo_config_value, baseline_slo)

        # Check if request meets SLO using the same logic as Request.slo_met property
        # Allow 7% tolerance: latency <= slo * (1 + 0.07)
        tolerance = 0.07
        meets_slo = latency <= target_slo * (1 + tolerance)

        if meets_slo:
            slo_met_count += 1

        # Update per_request_info with new SLO values
        updated_req_info = req_info.copy()
        updated_req_info["slo"] = target_slo
        updated_req_info["slo_met"] = meets_slo
        updated_per_request_info.append(updated_req_info)

        # Store detailed info for debugging
        detailed_info.append(
            {
                "req_idx": req_info.get("req_idx", i),
                "latency": latency,
                "target_slo": target_slo,
                "baseline_slo": baseline_slo,
                "slo_config": slo_config_value,
                "shape": (bs, height, width),
                "meets_slo": meets_slo,
            }
        )

    slo_meet_ratio = slo_met_count / len(latencies)

    # Print summary for the first few requests (debugging)
    if len(detailed_info) > 0:
        print(f"\nSample SLO calculations (first 5 requests):")
        print("Req | Shape         | Latency | Baseline | Config | Target | Meets")
        print("-" * 70)
        for info in detailed_info[:5]:
            bs, h, w = info["shape"]
            print(
                f"{info['req_idx']:3d} | ({bs},{h:4d},{w:4d}) | {info['latency']:7.2f} | {info['baseline_slo']:8.2f} | {info['slo_config']:>6} | {info['target_slo']:6.2f} | {'✓' if info['meets_slo'] else '✗'}"
            )

        if len(detailed_info) > 5:
            print(f"... and {len(detailed_info) - 5} more requests")

        print(f"\nOverall SLO meet ratio: {slo_meet_ratio:.3f} ({slo_met_count}/{len(latencies)})")

    return slo_meet_ratio, updated_per_request_info


def process_results_with_slo_configs(
    input_file: str, output_file: str, slo_config_names: List[str], schedulers_to_process: Optional[List[str]] = None
) -> None:
    """Process benchmark results with different SLO configuration files.

    Args:
        input_file: Input benchmark results JSON
        output_file: Output file for processed results
        slo_config_names: List of SLO config names to apply (without .json extension)
        schedulers_to_process: List of schedulers to process (None = all)
    """
    # Load results
    with open(input_file, "r") as f:
        results = json.load(f)

    print(f"Loaded {len(results)} results from {input_file}")

    # Group results by model to load baselines efficiently
    models_in_results = set(r.get("config", {}).get("model_name", "sd3") for r in results if "config" in r)
    baseline_slo_dicts = {}

    for model_name in models_in_results:
        # Use steps from first result for this model (they should all be the same)
        steps = 20  # default
        for r in results:
            if r.get("config", {}).get("model_name") == model_name and "config" in r:
                steps = r["config"].get("steps", 20)
                break

        baseline_slo_dicts[model_name] = load_baseline_slo_dict(model_name, steps)
        print(f"Loaded baseline SLOs for {model_name} with {steps} steps")

    # Load all SLO configs
    slo_configs = {}
    for slo_config_name in slo_config_names:
        slo_configs[slo_config_name] = load_slo_config(slo_config_name)

    # Process results
    new_results = []

    for i, result in enumerate(results):
        # Skip results with errors
        if result.get("error"):
            new_results.append(result)
            continue

        config = result.get("config", {})
        schedule_logic = config.get("schedule_logic", "")

        # Filter by scheduler if specified
        if schedulers_to_process and schedule_logic not in schedulers_to_process:
            new_results.append(result)  # Keep unchanged
            continue

        model_name = config.get("model_name", "sd3")

        # Get the baseline for this result
        baseline_slo_dict = baseline_slo_dicts[model_name]

        # Create new results with different SLO configs
        for slo_config_name in slo_config_names:
            slo_config = slo_configs[slo_config_name]

            # Deep copy result
            new_result = copy.deepcopy(result)

            # Update config with new SLO config name
            new_result["config"]["slo_config"] = slo_config_name

            # Recalculate SLO attendance
            try:
                print(
                    f"\nProcessing result {i+1}/{len(results)}: {schedule_logic} "
                    f"(model={model_name}, gpu={config.get('gpu_num')}, rate={config.get('rate')}) "
                    f"with SLO config '{slo_config_name}'"
                )

                new_slo_meet_ratio, updated_per_request_info = calculate_slo_attendance_from_config(
                    result, slo_config, baseline_slo_dict
                )

                # Update stats with new values
                new_result["stats"]["slo_meet_ratio"] = new_slo_meet_ratio
                new_result["stats"]["slo_violations"] = len(updated_per_request_info) - int(
                    new_slo_meet_ratio * len(updated_per_request_info)
                )
                # Update per_request_info with new SLO values
                new_result["stats"]["per_request_info"] = updated_per_request_info

                # Add metadata
                if "metadata" not in new_result:
                    new_result["metadata"] = {}
                new_result["metadata"]["reprocessed"] = True
                new_result["metadata"]["reprocess_timestamp"] = datetime.now().isoformat()
                new_result["metadata"]["original_slo_config"] = config.get("slo_config", "unknown")
                new_result["metadata"]["processor_version"] = "config_based_v2.0"  # Updated for new data format

                new_results.append(new_result)

                print(f"✓ SLO meet ratio: {new_slo_meet_ratio:.3f}")

            except Exception as e:
                print(f"✗ Error processing result {i+1} with SLO config '{slo_config_name}': {e}")
                # Add error result
                error_result = copy.deepcopy(result)
                error_result["error"] = str(e)
                error_result["config"]["slo_config"] = slo_config_name
                new_results.append(error_result)

    # Save results
    with open(output_file, "w") as f:
        json.dump(new_results, f, indent=2)

    print(f"\nSaved {len(new_results)} results to {output_file}")

    # Summary statistics
    successful = sum(1 for r in new_results if not r.get("error"))
    failed = len(new_results) - successful

    print(f"Successfully processed: {successful}")
    if failed > 0:
        print(f"Failed: {failed}")

    # Summary by scheduler and SLO config
    print(f"\nSummary by scheduler and SLO config:")
    scheduler_stats = {}

    for result in new_results:
        if result.get("error"):
            continue

        scheduler = result.get("config", {}).get("schedule_logic", "unknown")
        slo_config = result.get("config", {}).get("slo_config", "unknown")

        key = f"{scheduler}__{slo_config}"
        if key not in scheduler_stats:
            scheduler_stats[key] = {"count": 0, "slo_sum": 0}

        scheduler_stats[key]["count"] += 1
        scheduler_stats[key]["slo_sum"] += result.get("stats", {}).get("slo_meet_ratio", 0)

    for key, stats in scheduler_stats.items():
        scheduler, slo_config = key.split("__")
        avg_slo = stats["slo_sum"] / stats["count"] if stats["count"] > 0 else 0
        print(
            f"  {scheduler} (SLO config {slo_config}): {stats['count']} results, " f"avg SLO meet ratio = {avg_slo:.3f}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="SLO Analysis Using Configuration Files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Apply different SLO configs to baseline results (recommended workflow)
    python benchmark/serving_benchmark/analyze_slo_from_config.py \
      benchmark/serving_benchmark/results/benchmark_flux_H100_baseline_medium/benchmark_results.json \
      --slo-configs custom_1_1.0x_slo_H100_flux custom_1_1.1x_slo_H100_flux custom_1_1.2x_slo_H100_flux \
                    custom_1_1.3x_slo_H100_flux custom_1_1.5x_slo_H100_flux

    # Process only specific schedulers
    python benchmark/serving_benchmark/analyze_slo_from_config.py results.json \
      --schedulers data_parallel fixed_sp_2 fixed_sp_4 \
      --slo-configs custom_1_1.0x_slo_H100_flux custom_1_1.3x_slo_H100_flux

    # Generate output for plotting with multiple SLO configs
    python benchmark/serving_benchmark/analyze_slo_from_config.py \
      benchmark/serving_benchmark/results/benchmark_flux_H100_baseline_medium/benchmark_results.json \
      -o benchmark/serving_benchmark/results/benchmark_flux_H100_baseline_medium/benchmark_results_multi_slo.json \
      --slo-configs custom_1_1.0x_slo_H100_flux custom_1_1.1x_slo_H100_flux custom_1_1.2x_slo_H100_flux
        """,
    )

    parser.add_argument("input_file", help="Input benchmark results JSON file")
    parser.add_argument(
        "-o", "--output", help="Output file (default: input_file with _config_analyzed suffix)", default=None
    )
    parser.add_argument(
        "--slo-configs",
        nargs="+",
        required=True,
        help="SLO configuration files to apply (without .json extension). "
        "Files should be in benchmark/serving_benchmark/slo_configs/ directory.",
    )
    parser.add_argument("--schedulers", nargs="+", help="Only process these schedulers (default: all schedulers)")

    args = parser.parse_args()

    # Determine output file
    if args.output is None:
        input_path = Path(args.input_file)
        args.output = str(input_path.parent / f"{input_path.stem}_config_analyzed{input_path.suffix}")

    # Validate SLO config files exist
    missing_configs = []
    for slo_config in args.slo_configs:
        config_path = Path(__file__).parent / "slo_configs" / f"{slo_config}.json"
        if not config_path.exists():
            missing_configs.append(slo_config)

    if missing_configs:
        print(f"❌ Missing SLO config files: {missing_configs}")
        print(f"Available configs in benchmark/serving_benchmark/slo_configs/:")
        slo_config_dir = Path(__file__).parent / "slo_configs"
        if slo_config_dir.exists():
            for config_file in sorted(slo_config_dir.glob("*.json")):
                print(f"  - {config_file.stem}")
        print("\nNote: Config files are loaded from benchmark/serving_benchmark/slo_configs/")
        sys.exit(1)

    try:
        process_results_with_slo_configs(args.input_file, args.output, args.slo_configs, args.schedulers)
        print(f"\n✅ Successfully completed SLO analysis with configuration files!")
        print(f"📊 Results saved to: {args.output}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
