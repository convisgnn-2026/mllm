"""
Data processing module for benchmark results.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np


class DataProcessor:
    """Handles loading and processing of benchmark results."""

    def __init__(self, results_file: str):
        self.results_file = Path(results_file)
        self.raw_results = self._load_results()

    def _load_results(self) -> List[Dict[str, Any]]:
        """Load benchmark results from JSON file."""
        with open(self.results_file, "r") as f:
            return json.load(f)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to a pandas DataFrame for easier analysis."""
        data = []
        for result in self.raw_results:
            if "error" in result:
                continue  # Skip failed experiments

            config = result["config"]
            stats = result["stats"]

            row = {
                # Configuration
                "model_name": config["model_name"],
                "schedule_logic": config["schedule_logic"],
                "gpu_num": config["gpu_num"],
                "distribution": config["distribution"],
                "rate": config["rate"],
                "cv": config["cv"],
                "steps": config["steps"],
                "shape_dist": config["shape_dist"],
                "output_type": config["output_type"],
                "duration": config.get("duration", None),
                "request_num": config.get("request_num", None),
                "slo_config": config.get("slo_config", None),
                # Metrics
                "num_requests": stats["num_requests"],
                "throughput": stats["throughput"],
                "latency_mean": stats["latency_mean"],
                "latency_std": stats["latency_std"],
                "latency_p90": stats["latency_p90"],
                "latency_p99": stats["latency_p99"],
                "request_rate": stats["request_rate"],
                "slo_meet_ratio": stats["slo_meet_ratio"],
                # Per-request data for spider plots and time-series analysis
                "per_request_info": stats.get("per_request_info", None),
                "latency": stats.get("latency", None),
                "request_starts": stats.get("request_starts", None),
                "request_finishes": stats.get("request_finishes", None),
            }
            data.append(row)

        return pd.DataFrame(data)

    def filter_schedules(self, df: pd.DataFrame, exclude_schedules: List[str]) -> pd.DataFrame:
        """Filter out specified scheduling algorithms."""
        if not exclude_schedules:
            return df

        filtered_df = df[~df["schedule_logic"].isin(exclude_schedules)]
        print(f"Filtered out scheduling algorithms: {', '.join(exclude_schedules)}")
        print(f"Remaining data points: {len(filtered_df)}")
        return filtered_df

    def get_data_summary(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Get summary of available data."""
        return {
            "models": sorted(df["model_name"].unique()),
            "schedules": sorted(df["schedule_logic"].unique()),
            "slo_configs": sorted(df["slo_config"].dropna().unique()) if "slo_config" in df.columns else [],
            "rates": sorted(df["rate"].unique()),
        }


def get_experimental_conditions(df: pd.DataFrame, extra_columns: List[str] = None) -> pd.DataFrame:
    """
    Get all unique experimental condition combinations.

    Args:
        df: DataFrame with benchmark results
        extra_columns: Additional columns to include in conditions

    Returns:
        DataFrame with unique condition combinations
    """
    condition_columns = ["shape_dist", "cv", "distribution", "model_name"]

    # Add workload specification columns
    if "duration" in df.columns and df["duration"].notna().any():
        condition_columns.append("duration")
    if "request_num" in df.columns and df["request_num"].notna().any():
        condition_columns.append("request_num")

    # Add any extra columns
    if extra_columns:
        for col in extra_columns:
            if col in df.columns and col not in condition_columns:
                condition_columns.append(col)

    conditions = df[condition_columns].drop_duplicates()
    return conditions.sort_values(condition_columns)


def filter_by_condition(df: pd.DataFrame, condition) -> pd.DataFrame:
    """
    Filter dataframe by specific experimental condition.

    Args:
        df: DataFrame to filter
        condition: Condition object from itertuples

    Returns:
        Filtered DataFrame
    """
    condition_filter = (
        (df["shape_dist"] == condition.shape_dist)
        & (df["cv"] == condition.cv)
        & (df["distribution"] == condition.distribution)
        & (df["model_name"] == condition.model_name)
    )

    # Add duration filter if present
    if hasattr(condition, "duration") and condition.duration is not None:
        condition_filter = condition_filter & (df["duration"] == condition.duration)

    # Add request_num filter if present
    if hasattr(condition, "request_num") and condition.request_num is not None:
        condition_filter = condition_filter & (df["request_num"] == condition.request_num)

    # Add slo_config filter if present
    if hasattr(condition, "slo_config"):
        if pd.isna(condition.slo_config):
            condition_filter = condition_filter & df["slo_config"].isna()
        else:
            condition_filter = condition_filter & (df["slo_config"] == condition.slo_config)

    return df[condition_filter]


def generate_condition_string(condition) -> str:
    """Generate human-readable condition string for plot titles."""
    parts = [
        f"Model: {condition.model_name}",
        f"Shape: {condition.shape_dist}",
        f"CV: {condition.cv}",
        f"Distribution: {condition.distribution}",
    ]

    # Add workload specification
    if hasattr(condition, "duration") and condition.duration is not None:
        parts.append(f"Duration: {condition.duration}s")
    # Note: Removed request_num from title as requested

    # Add SLO config if present
    if hasattr(condition, "slo_config") and not pd.isna(condition.slo_config):
        parts.append(f"SLO Config: {condition.slo_config}")

    return ", ".join(parts)


def generate_save_path_suffix(condition, plot_type: str) -> str:
    """Generate condition-specific filename suffix."""
    condition_parts = [
        f"model-{condition.model_name.replace('/', '_')}",
        f"shape-{condition.shape_dist}",
        f"cv-{condition.cv}",
        f"dist-{condition.distribution}",
    ]

    # Add workload specification
    if hasattr(condition, "duration") and condition.duration is not None:
        condition_parts.append(f"dur-{condition.duration}")
    if hasattr(condition, "request_num") and condition.request_num is not None:
        condition_parts.append(f"req-{condition.request_num}")

    # Add SLO config if present
    if hasattr(condition, "slo_config") and not pd.isna(condition.slo_config):
        condition_parts.append(f"slo-{condition.slo_config}")

    condition_str = "_".join(condition_parts).replace(".", "p")
    return f"{plot_type}_{condition_str}"
