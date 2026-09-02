"""
Arrival processes for workload generation.

Simple implementations of Poisson and Gamma processes.
"""

from abc import ABC, abstractmethod
from typing import List
import numpy as np

from .core import Request, Workload


class ArrivalProcess(ABC):
    """
    Abstract base class for arrival processes.

    An arrival process generates timestamps for when requests arrive.
    """

    @abstractmethod
    def generate_arrival_times(self, start: float, duration: float, seed: int = 42) -> List[float]:
        """
        Generate arrival times.

        Args:
            start: Start time
            duration: Duration to generate arrivals for
            seed: Random seed for reproducibility

        Returns:
            List of arrival timestamps
        """
        pass

    @abstractmethod
    def get_rate(self) -> float:
        """Get mean arrival rate (requests per second)."""
        pass

    @abstractmethod
    def get_cv(self) -> float:
        """Get coefficient of variation of inter-arrival times."""
        pass

    def generate_workload(self, model_name: str, duration: float, seed: int = 42) -> Workload:
        """
        Generate a workload with empty requests.

        Args:
            model_name: Model to use
            duration: Duration in seconds
            seed: Random seed

        Returns:
            Workload with requests (not yet populated)
        """
        arrival_times = self.generate_arrival_times(0.0, duration, seed)

        # Create empty requests
        requests = [
            Request(req_idx=i, model_name=model_name, arrival_time=arrival_times[i]) for i in range(len(arrival_times))
        ]

        return Workload(requests, arrival_times)


class PoissonProcess(ArrivalProcess):
    """
    Poisson arrival process (exponentially distributed inter-arrival times).

    This is a memoryless process where arrivals are independent.
    CV = 1.0 (by definition)
    """

    def __init__(self, rate: float):
        """
        Initialize Poisson process.

        Args:
            rate: Mean arrival rate (requests per second)

        Raises:
            ValueError: If rate is not positive
        """
        if rate <= 0:
            raise ValueError(f"Rate must be positive, got {rate}")

        self._rate = rate

    def get_rate(self) -> float:
        return self._rate

    def get_cv(self) -> float:
        return 1.0  # Poisson always has CV = 1

    def generate_arrival_times(self, start: float, duration: float, seed: int = 42) -> List[float]:
        """Generate Poisson arrivals."""
        np.random.seed(seed)

        # Exponential inter-arrival times
        mean_interval = 1.0 / self._rate

        # Generate enough intervals (with buffer)
        max_arrivals = int(duration * self._rate * 1.5) + 10
        intervals = np.random.exponential(mean_interval, size=max_arrivals)

        # Compute cumulative arrival times
        arrival_times = []
        current_time = start

        for interval in intervals:
            current_time += interval
            if current_time >= start + duration:
                break
            arrival_times.append(current_time)

        return arrival_times


class GammaProcess(ArrivalProcess):
    """
    Gamma arrival process (Gamma distributed inter-arrival times).

    More flexible than Poisson:
    - CV < 1: More regular than Poisson (less bursty)
    - CV = 1: Same as Poisson
    - CV > 1: More bursty than Poisson
    """

    def __init__(self, rate: float, cv: float):
        """
        Initialize Gamma process.

        Args:
            rate: Mean arrival rate (requests per second)
            cv: Coefficient of variation (CV = std_dev / mean)

        Raises:
            ValueError: If rate or cv is not positive
        """
        if rate <= 0:
            raise ValueError(f"Rate must be positive, got {rate}")
        if cv <= 0:
            raise ValueError(f"CV must be positive, got {cv}")

        self._rate = rate
        self._cv = cv

        # Gamma distribution parameters
        # For Gamma: mean = shape * scale, cv = 1 / sqrt(shape)
        self._shape = 1.0 / (cv * cv)
        self._scale = cv * cv / rate

    def get_rate(self) -> float:
        return self._rate

    def get_cv(self) -> float:
        return self._cv

    def generate_arrival_times(self, start: float, duration: float, seed: int = 42) -> List[float]:
        """Generate Gamma arrivals."""
        np.random.seed(seed)

        # Generate enough intervals (with buffer)
        max_arrivals = int(self._rate * duration * 1.5) + 10
        intervals = np.random.gamma(self._shape, self._scale, size=max_arrivals)

        # Compute cumulative arrival times
        arrival_times = []
        current_time = start

        for interval in intervals:
            current_time += interval
            if current_time >= start + duration:
                break
            arrival_times.append(current_time)

        return arrival_times


class DeterministicProcess(ArrivalProcess):
    """
    Deterministic arrival process (fixed inter-arrival times).

    Useful for testing and debugging.
    CV = 0 (perfectly regular)
    """

    def __init__(self, rate: float):
        """
        Initialize deterministic process.

        Args:
            rate: Arrival rate (requests per second)
        """
        if rate <= 0:
            raise ValueError(f"Rate must be positive, got {rate}")

        self._rate = rate

    def get_rate(self) -> float:
        return self._rate

    def get_cv(self) -> float:
        return 0.0  # Deterministic has CV = 0

    def generate_arrival_times(self, start: float, duration: float, seed: int = 42) -> List[float]:
        """Generate deterministic arrivals."""
        interval = 1.0 / self._rate
        num_requests = int(duration * self._rate)

        return [start + i * interval for i in range(num_requests)]
