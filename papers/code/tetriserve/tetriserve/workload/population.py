"""
Request population - assign shapes and parameters to requests.

Simple, pure functions. No side effects, no global state.
"""

from typing import List
import numpy as np

from .core import Workload, Request
from .distribution import ShapeDistribution, BatchSizeDistribution


def populate_requests(
    workload: Workload,
    shape_dist: ShapeDistribution,
    batch_dist: BatchSizeDistribution,
    num_inference_steps: int = 20,
    output_type: str = "latent",
    seed: int = 42,
) -> Workload:
    """
    Populate requests with shapes and parameters.

    Args:
        workload: Workload with empty requests
        shape_dist: Distribution for image shapes
        batch_dist: Distribution for batch sizes
        num_inference_steps: Number of inference steps
        output_type: Output type ('pil', 'latent', 'pt')
        seed: Random seed for reproducibility

    Returns:
        Workload with populated requests

    Note:
        This function modifies requests in-place for efficiency,
        but conceptually creates a new populated workload.
    """
    rng = np.random.default_rng(seed)

    for req in workload.requests:
        # Sample shape
        height, width = shape_dist.sample(rng)

        # Sample batch size
        batch_size = batch_dist.sample(rng)

        # Create prompts for batch
        prompt = ["a photo of a cat"] * batch_size

        # Update request (in-place for efficiency)
        req.batch_size = batch_size
        req.height = height
        req.width = width
        req.num_inference_steps = num_inference_steps
        req.prompt = prompt
        req.output_type = output_type

    return workload


def populate_requests_with_prompts(
    workload: Workload,
    prompts: List[str],
    shape_dist: ShapeDistribution,
    batch_dist: BatchSizeDistribution,
    num_inference_steps: int = 20,
    output_type: str = "latent",
    seed: int = 42,
) -> Workload:
    """
    Populate requests with custom prompts.

    Args:
        workload: Workload with empty requests
        prompts: List of prompts to sample from
        shape_dist: Distribution for image shapes
        batch_dist: Distribution for batch sizes
        num_inference_steps: Number of inference steps
        output_type: Output type
        seed: Random seed

    Returns:
        Workload with populated requests
    """
    rng = np.random.default_rng(seed)

    for req in workload.requests:
        # Sample shape
        height, width = shape_dist.sample(rng)

        # Sample batch size
        batch_size = batch_dist.sample(rng)

        # Sample prompts for batch
        prompt = [rng.choice(prompts) for _ in range(batch_size)]

        # Update request
        req.batch_size = batch_size
        req.height = height
        req.width = width
        req.num_inference_steps = num_inference_steps
        req.prompt = prompt
        req.output_type = output_type

    return workload


def populate_fixed_requests(
    workload: Workload,
    height: int = 1024,
    width: int = 1024,
    batch_size: int = 1,
    num_inference_steps: int = 20,
    output_type: str = "latent",
) -> Workload:
    """
    Populate all requests with same fixed parameters.

    Useful for benchmarking a specific configuration.

    Args:
        workload: Workload with empty requests
        height: Image height
        width: Image width
        batch_size: Batch size
        num_inference_steps: Number of steps
        output_type: Output type

    Returns:
        Workload with fixed populated requests
    """
    prompt = ["a photo of a cat"] * batch_size

    for req in workload.requests:
        req.batch_size = batch_size
        req.height = height
        req.width = width
        req.num_inference_steps = num_inference_steps
        req.prompt = prompt
        req.output_type = output_type

    return workload
