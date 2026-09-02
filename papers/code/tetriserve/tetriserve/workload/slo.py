"""
SLO (Service Level Objective) management - clean and simple.

No caching, no global state, just pure functions and simple classes.
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional
from pathlib import Path
import json

from .core import Workload, Request


class SLOPolicy(ABC):
    """
    Abstract SLO policy.

    An SLO policy calculates the maximum allowed latency for a request.
    """

    @abstractmethod
    def calculate_slo(self, request: Request) -> float:
        """
        Calculate SLO for a request.

        Args:
            request: Request to calculate SLO for

        Returns:
            SLO in seconds
        """
        pass


class FixedSLOPolicy(SLOPolicy):
    """
    Fixed SLO policy - same SLO for all requests.

    Simple and predictable.
    """

    def __init__(self, slo: float):
        """
        Initialize with fixed SLO.

        Args:
            slo: SLO in seconds
        """
        if slo <= 0:
            raise ValueError(f"SLO must be positive, got {slo}")
        self._slo = slo

    def calculate_slo(self, request: Request) -> float:
        return self._slo


class BaselineSLOPolicy(SLOPolicy):
    """
    SLO based on baseline latencies from performance model.

    Load baseline data once at initialization (no caching needed).
    """

    def __init__(self, baseline_data: Dict[Tuple[int, int, int], float], multiplier: float = 1.0):
        """
        Initialize with baseline data.

        Args:
            baseline_data: Dict mapping (batch_size, height, width) to baseline latency
            multiplier: Multiplier to apply to baseline (e.g., 1.5 for 50% slack)
        """
        self.baseline_data = baseline_data
        self.multiplier = multiplier

    def calculate_slo(self, request: Request) -> float:
        """Calculate SLO as baseline * multiplier."""
        key = (request.batch_size, request.height, request.width)
        baseline = self.baseline_data.get(key, 10.0)  # Default 10s
        return baseline * self.multiplier

    @classmethod
    def from_file(
        cls,
        model_name: str,
        steps: int,
        gpu_count: int = 1,
        base_path: str = "log/benchmark/scaling_efficiency",
        multiplier: float = 1.0,
    ) -> "BaselineSLOPolicy":
        """
        Load baseline data from scaling efficiency file.

        Args:
            model_name: Model name (e.g., 'sd3', 'flux')
            steps: Number of inference steps
            gpu_count: Number of GPUs used for baseline
            base_path: Path to scaling efficiency data
            multiplier: Multiplier for baseline

        Returns:
            BaselineSLOPolicy instance

        Raises:
            FileNotFoundError: If baseline data not found
        """
        # Import here to avoid circular dependency
        from tetriserve.utils import get_gpu_name

        path = Path(base_path) / model_name / get_gpu_name()
        file_path = path / f"e2e_scaling_efficiency_{gpu_count}.json"

        if not file_path.exists():
            raise FileNotFoundError(
                f"Baseline data not found: {file_path}\n" f"Please run scaling efficiency benchmark first."
            )

        with open(file_path, "r") as f:
            data = json.load(f)

        # Convert to baseline dict
        baseline_data = {}
        for entry in data:
            bs = entry["bs"]
            height = entry["height"]
            width = entry["width"]
            entry_steps = entry["steps"]
            avg_time = entry["avg_e2e_time"]

            # Scale to target steps
            scaled_time = avg_time * (steps / entry_steps)
            baseline_data[(bs, height, width)] = scaled_time

        return cls(baseline_data, multiplier)


class CustomSLOPolicy(SLOPolicy):
    """
    Custom SLO policy from JSON configuration.

    Supports both multiplier format ("1.5x") and fixed values (15.0).
    """

    def __init__(self, slo_mapping: Dict[str, any], baseline_policy: Optional[BaselineSLOPolicy] = None):
        """
        Initialize with custom SLO mapping.

        Args:
            slo_mapping: Dict mapping "(bs, height, width)" to SLO value
                        Values can be "1.5x" (multiplier) or 15.0 (fixed)
            baseline_policy: Baseline policy for multiplier calculations
        """
        self.slo_mapping = slo_mapping
        self.baseline_policy = baseline_policy

    def calculate_slo(self, request: Request) -> float:
        """Calculate SLO from custom mapping."""
        key = f"({request.batch_size}, {request.height}, {request.width})"

        if key not in self.slo_mapping:
            # Try to find any mapping for this shape (ignoring batch)
            for mapping_key in self.slo_mapping:
                if f", {request.height}, {request.width})" in mapping_key:
                    return self._parse_slo_value(self.slo_mapping[mapping_key], request)

            raise ValueError(f"No SLO mapping found for {key}. " f"Available keys: {list(self.slo_mapping.keys())}")

        return self._parse_slo_value(self.slo_mapping[key], request)

    def _parse_slo_value(self, slo_config: any, request: Request) -> float:
        """
        Parse SLO configuration value.

        Args:
            slo_config: Either "1.5x" (multiplier) or 15.0 (fixed)
            request: Request to calculate SLO for

        Returns:
            SLO in seconds
        """
        # Multiplier format: "1.5x"
        if isinstance(slo_config, str) and slo_config.endswith("x"):
            try:
                multiplier = float(slo_config[:-1])
            except ValueError:
                raise ValueError(f"Invalid multiplier format: '{slo_config}'. " f"Use format like '1.5x'")

            # Need baseline policy for multipliers
            if self.baseline_policy is None:
                raise ValueError(f"Multiplier '{slo_config}' requires baseline_policy")

            baseline = self.baseline_policy.calculate_slo(request)
            return baseline * multiplier

        # Fixed value format: 15.0
        elif isinstance(slo_config, (int, float)):
            return float(slo_config)

        else:
            raise ValueError(
                f"Invalid SLO format: '{slo_config}'. " f"Use either '1.5x' (multiplier) or 15.0 (fixed value)"
            )

    @classmethod
    def from_file(cls, config_file: str, baseline_policy: Optional[BaselineSLOPolicy] = None) -> "CustomSLOPolicy":
        """
        Load custom SLO policy from JSON file.

        Args:
            config_file: Path to SLO configuration JSON file
            baseline_policy: Baseline policy for multiplier calculations

        Returns:
            CustomSLOPolicy instance
        """
        config_path = Path(config_file)
        if not config_path.exists():
            raise FileNotFoundError(f"SLO config not found: {config_file}")

        with open(config_path, "r") as f:
            slo_mapping = json.load(f)

        return cls(slo_mapping, baseline_policy)


def apply_slo_policy(workload: Workload, policy: SLOPolicy) -> Workload:
    """
    Apply SLO policy to all requests in workload.

    Args:
        workload: Workload to apply policy to
        policy: SLO policy to use

    Returns:
        Workload with SLOs assigned (modifies in-place for efficiency)
    """
    for request in workload.requests:
        request.slo = policy.calculate_slo(request)

    return workload


# Convenience functions for common use cases


def load_slo_policy(
    config_name: str,
    model_name: str,
    steps: int,
    config_dir: str = "benchmark/serving/slo_configs",
    baseline_dir: str = "log/benchmark/scaling_efficiency",
) -> SLOPolicy:
    """
    Load SLO policy from configuration.

    This is a convenience function that:
    1. Loads baseline data if needed (for multiplier support)
    2. Loads custom configuration
    3. Returns appropriate policy

    Args:
        config_name: Name of SLO config (without .json)
        model_name: Model name for baseline
        steps: Number of inference steps
        config_dir: Directory containing SLO configs
        baseline_dir: Directory containing baseline data

    Returns:
        SLO policy ready to use

    Raises:
        FileNotFoundError: If the specified SLO config file is not found
    """
    # Try to load baseline for multipliers
    baseline_policy = None
    try:
        baseline_policy = BaselineSLOPolicy.from_file(model_name, steps, gpu_count=1, base_path=baseline_dir)
    except FileNotFoundError:
        pass  # No baseline available, can still use fixed values

    # Load custom config - MUST exist, no silent fallback
    config_file = Path(config_dir) / f"{config_name}.json"

    if not config_file.exists():
        raise FileNotFoundError(
            f"SLO config not found: {config_file}\n"
            f"Specified: '{config_name}'\n"
            f"Looking in: {config_dir}\n"
            f"Available files: {list(Path(config_dir).glob('*.json'))}"
        )

    return CustomSLOPolicy.from_file(str(config_file), baseline_policy)
