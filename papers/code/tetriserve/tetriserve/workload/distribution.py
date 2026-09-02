"""
Shape and batch size distributions - clean and simple.

No global state, just pure data structures and functions.
"""

from dataclasses import dataclass
from typing import List, Tuple
import math
import numpy as np


@dataclass(frozen=True)
class ShapeDistribution:
    """
    Immutable shape distribution configuration.

    Attributes:
        shapes: List of (height, width) tuples
        probabilities: Probability for each shape (must sum to 1.0)
    """

    shapes: Tuple[Tuple[int, int], ...]
    probabilities: Tuple[float, ...]

    def __post_init__(self):
        """Validate distribution."""
        if len(self.shapes) != len(self.probabilities):
            raise ValueError(
                f"Number of shapes ({len(self.shapes)}) must match "
                f"number of probabilities ({len(self.probabilities)})"
            )

        prob_sum = sum(self.probabilities)
        if not (0.99 <= prob_sum <= 1.01):  # Allow small floating point error
            raise ValueError(f"Probabilities must sum to 1.0, got {prob_sum}")

    def sample(self, rng: np.random.Generator) -> Tuple[int, int]:
        """
        Sample a shape from the distribution.

        Args:
            rng: Numpy random number generator

        Returns:
            (height, width) tuple
        """
        idx = rng.choice(len(self.shapes), p=self.probabilities)
        return self.shapes[idx]

    @classmethod
    def evenly(cls) -> "ShapeDistribution":
        """
        Evenly distributed across all resolutions.

        Returns:
            ShapeDistribution with equal probabilities
        """
        shapes = ((256, 256), (512, 512), (1024, 1024), (2048, 2048))
        probs = tuple([1.0 / len(shapes)] * len(shapes))
        return cls(shapes, probs)

    @classmethod
    def short_seqlen_high(cls) -> "ShapeDistribution":
        """
        Distribution favoring shorter sequence lengths (smaller images).

        Uses sigmoid function to favor smaller resolutions.
        """
        shapes = ((256, 256), (512, 512), (1024, 1024), (2048, 2048))

        # Calculate probabilities using sigmoid (favors small)
        raw_probs = [1.0 / (1.0 + math.exp((h * w // 16 // 16) / 4096)) for h, w in shapes]

        # Normalize to sum to 1.0
        total = sum(raw_probs)
        probs = tuple([p / total for p in raw_probs])

        return cls(shapes, probs)

    @classmethod
    def long_seqlen_high(cls) -> "ShapeDistribution":
        """
        Distribution favoring longer sequence lengths (larger images).
        Uses exponential function to favor larger resolutions.
        """
        shapes = ((256, 256), (512, 512), (1024, 1024), (2048, 2048))

        # Calculate probabilities using exponential (favors large)
        raw_probs = [math.exp((h * w // 16 // 16) / 16384) for h, w in shapes]

        # Normalize to sum to 1.0
        total = sum(raw_probs)
        probs = tuple([p / total for p in raw_probs])

        return cls(shapes, probs)

    @classmethod
    def long_seqlen_high_power(cls) -> "ShapeDistribution":
        """
        Distribution favoring longer sequence lengths (larger images).
        Uses exponential function to favor larger resolutions.
        """
        shapes = ((256, 256), (512, 512), (1024, 1024), (2048, 2048))

        # Calculate probabilities using exponential (favors large)
        areas = [h * w for h, w in shapes]
        raw_probs = [a**0.4 for a in areas]
        # raw_probs = [math.exp((h * w // 16 // 16) / 16384) for h, w in shapes]

        # Normalize to sum to 1.0
        total = sum(raw_probs)
        probs = tuple([p / total for p in raw_probs])

        return cls(shapes, probs)

    @classmethod
    def poisson(cls) -> "ShapeDistribution":
        """
        Poisson-like distribution normalized by latent length.
        """
        shapes = ((256, 256), (512, 512), (1024, 1024), (2048, 2048))

        # Poisson distribution
        raw_probs = [math.exp(-1.0 * (h * w // 16 // 16) / 16384) for h, w in shapes]

        # Normalize
        total = sum(raw_probs)
        probs = tuple([p / total for p in raw_probs])

        return cls(shapes, probs)

    @classmethod
    def custom(cls, shapes: List[Tuple[int, int]], probabilities: List[float]) -> "ShapeDistribution":
        """
        Create custom distribution.

        Args:
            shapes: List of (height, width) tuples
            probabilities: List of probabilities (must sum to 1.0)

        Returns:
            ShapeDistribution instance
        """
        return cls(tuple(shapes), tuple(probabilities))


@dataclass(frozen=True)
class BatchSizeDistribution:
    """
    Immutable batch size distribution.

    Attributes:
        batch_sizes: List of batch sizes
        probabilities: Probability for each batch size (must sum to 1.0)
    """

    batch_sizes: Tuple[int, ...]
    probabilities: Tuple[float, ...]

    def __post_init__(self):
        """Validate distribution."""
        if len(self.batch_sizes) != len(self.probabilities):
            raise ValueError(
                f"Number of batch sizes ({len(self.batch_sizes)}) must match "
                f"number of probabilities ({len(self.probabilities)})"
            )

        prob_sum = sum(self.probabilities)
        if not (0.99 <= prob_sum <= 1.01):
            raise ValueError(f"Probabilities must sum to 1.0, got {prob_sum}")

        for bs in self.batch_sizes:
            if bs <= 0:
                raise ValueError(f"Batch size must be positive, got {bs}")

    def sample(self, rng: np.random.Generator) -> int:
        """
        Sample a batch size from the distribution.

        Args:
            rng: Numpy random number generator

        Returns:
            Batch size
        """
        idx = rng.choice(len(self.batch_sizes), p=self.probabilities)
        return self.batch_sizes[idx]

    @classmethod
    def uniform(cls, batch_sizes: List[int]) -> "BatchSizeDistribution":
        """
        Create uniform distribution over batch sizes.

        Args:
            batch_sizes: List of batch sizes

        Returns:
            BatchSizeDistribution with equal probabilities
        """
        probs = tuple([1.0 / len(batch_sizes)] * len(batch_sizes))
        return cls(tuple(batch_sizes), probs)

    @classmethod
    def single_batch(cls, batch_size: int = 1) -> "BatchSizeDistribution":
        """
        Create distribution with single batch size.

        Args:
            batch_size: Batch size to use

        Returns:
            BatchSizeDistribution with probability 1.0 for given batch size
        """
        return cls((batch_size,), (1.0,))
