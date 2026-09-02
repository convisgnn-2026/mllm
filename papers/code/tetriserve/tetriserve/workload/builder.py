"""
Workload builder for fluent API.

Makes it easy to create configured workloads with a clean, readable syntax.
"""

from typing import Optional
from .core import Workload
from .arrival import ArrivalProcess, PoissonProcess
from .distribution import ShapeDistribution, BatchSizeDistribution
from .population import populate_requests
from .slo import SLOPolicy, apply_slo_policy


class WorkloadBuilder:
    """
    Fluent builder for creating workloads.

    Example with duration:
        workload = (
            WorkloadBuilder()
            .with_model("sd3")
            .with_arrival_process(PoissonProcess(rate=1.0))
            .with_duration(60)  # Generate for 60 seconds
            .with_shape_distribution(ShapeDistribution.evenly())
            .with_batch_distribution(BatchSizeDistribution.single_batch(1))
            .with_slo_policy(FixedSLOPolicy(10.0))
            .with_num_steps(20)
            .build()
        )

    Example with num_requests:
        workload = (
            WorkloadBuilder()
            .with_model("flux")
            .with_arrival_process(PoissonProcess(rate=2.0))
            .with_num_requests(100)  # Generate exactly 100 requests
            .with_shape_distribution(ShapeDistribution.evenly())
            .with_num_steps(20)
            .build()
        )
    """

    def __init__(self):
        """Initialize builder with defaults."""
        self._model_name: Optional[str] = None
        self._arrival_process: Optional[ArrivalProcess] = None
        self._duration: Optional[float] = None
        self._num_requests: Optional[int] = None
        self._seed: int = 42

        # Population parameters
        self._shape_dist: Optional[ShapeDistribution] = None
        self._batch_dist: Optional[BatchSizeDistribution] = None
        self._num_steps: int = 20
        self._output_type: str = "latent"

        # SLO policy
        self._slo_policy: Optional[SLOPolicy] = None

    def with_model(self, model_name: str) -> "WorkloadBuilder":
        """Set model name."""
        self._model_name = model_name
        return self

    def with_arrival_process(self, process: ArrivalProcess) -> "WorkloadBuilder":
        """Set arrival process."""
        self._arrival_process = process
        return self

    def with_duration(self, duration: float) -> "WorkloadBuilder":
        """Set workload duration in seconds."""
        if duration <= 0:
            raise ValueError(f"Duration must be positive, got {duration}")
        if self._num_requests is not None:
            raise ValueError("Cannot set both duration and num_requests. Choose one.")
        self._duration = duration
        return self

    def with_num_requests(self, num_requests: int) -> "WorkloadBuilder":
        """
        Set number of requests to generate.

        Note: This is an alternative to with_duration(). Use one or the other, not both.
        The builder will calculate the appropriate duration based on the arrival rate.

        Args:
            num_requests: Number of requests to generate

        Returns:
            Self for chaining
        """
        if num_requests <= 0:
            raise ValueError(f"num_requests must be positive, got {num_requests}")
        if self._duration is not None:
            raise ValueError("Cannot set both num_requests and duration. Choose one.")
        self._num_requests = num_requests
        return self

    def with_seed(self, seed: int) -> "WorkloadBuilder":
        """Set random seed."""
        self._seed = seed
        return self

    def with_shape_distribution(self, dist: ShapeDistribution) -> "WorkloadBuilder":
        """Set shape distribution."""
        self._shape_dist = dist
        return self

    def with_batch_distribution(self, dist: BatchSizeDistribution) -> "WorkloadBuilder":
        """Set batch size distribution."""
        self._batch_dist = dist
        return self

    def with_num_steps(self, steps: int) -> "WorkloadBuilder":
        """Set number of inference steps."""
        if steps <= 0:
            raise ValueError(f"Steps must be positive, got {steps}")
        self._num_steps = steps
        return self

    def with_output_type(self, output_type: str) -> "WorkloadBuilder":
        """Set output type."""
        valid_types = ["pil", "latent", "pt"]
        if output_type not in valid_types:
            raise ValueError(f"Invalid output type: {output_type}. Must be one of {valid_types}")
        self._output_type = output_type
        return self

    def with_slo_policy(self, policy: SLOPolicy) -> "WorkloadBuilder":
        """Set SLO policy."""
        self._slo_policy = policy
        return self

    def build(self) -> Workload:
        """
        Build the workload.

        Returns:
            Configured workload ready to use

        Raises:
            ValueError: If required parameters are missing
        """
        # Validate required parameters
        if self._model_name is None:
            raise ValueError("model_name is required (use with_model())")

        if self._arrival_process is None:
            raise ValueError("arrival_process is required (use with_arrival_process())")

        # Validate that exactly one of duration or num_requests is set
        if self._duration is None and self._num_requests is None:
            raise ValueError("Either duration or num_requests is required")

        if self._duration is not None and self._num_requests is not None:
            raise ValueError("Cannot specify both duration and num_requests. Choose one.")

        # Use defaults for optional parameters
        shape_dist = self._shape_dist or ShapeDistribution.evenly()
        batch_dist = self._batch_dist or BatchSizeDistribution.single_batch(1)

        # Determine duration for arrival generation
        if self._duration is not None:
            # Use specified duration
            generation_duration = self._duration
        else:
            # Calculate duration from num_requests and rate
            # Use 1.2x to ensure we get enough requests, then trim
            rate = self._arrival_process.get_rate()
            generation_duration = (self._num_requests / rate) * 1.5  # 50% buffer

        # Step 1: Generate workload with arrivals
        workload = self._arrival_process.generate_workload(self._model_name, generation_duration, self._seed)

        # If num_requests was specified, trim to exact number
        if self._num_requests is not None:
            if len(workload.requests) > self._num_requests:
                # Trim to exact number
                workload.requests = workload.requests[: self._num_requests]
                workload.arrival_times = workload.arrival_times[: self._num_requests]
            elif len(workload.requests) < self._num_requests:
                # Not enough requests generated (rare edge case)
                raise ValueError(
                    f"Could not generate {self._num_requests} requests. "
                    f"Only generated {len(workload.requests)}. "
                    f"Try increasing the rate or using duration instead."
                )

        # Step 2: Populate requests with shapes and parameters
        workload = populate_requests(
            workload,
            shape_dist,
            batch_dist,
            self._num_steps,
            self._output_type,
            self._seed,
        )

        # Step 3: Apply SLO policy if provided
        if self._slo_policy is not None:
            workload = apply_slo_policy(workload, self._slo_policy)

        return workload


def create_simple_workload(
    model_name: str,
    rate: float,
    duration: float = None,
    num_requests: int = None,
    distribution_type: str = "poisson",
    cv: float = 1.0,
    num_steps: int = 20,
    seed: int = 42,
) -> Workload:
    """
    Convenience function for creating simple workloads.

    Args:
        model_name: Model to use
        rate: Arrival rate (requests/second)
        duration: Duration in seconds (use this OR num_requests)
        num_requests: Number of requests (use this OR duration)
        distribution_type: 'poisson' or 'gamma'
        cv: Coefficient of variation (for gamma)
        num_steps: Number of inference steps
        seed: Random seed

    Returns:
        Workload with default settings
    """
    from .arrival import PoissonProcess, GammaProcess

    # Validate
    if duration is None and num_requests is None:
        raise ValueError("Must specify either duration or num_requests")
    if duration is not None and num_requests is not None:
        raise ValueError("Cannot specify both duration and num_requests")

    # Create arrival process
    if distribution_type == "poisson":
        arrival_process = PoissonProcess(rate)
    elif distribution_type == "gamma":
        arrival_process = GammaProcess(rate, cv)
    else:
        raise ValueError(f"Unknown distribution: {distribution_type}")

    # Build workload with defaults
    builder = (
        WorkloadBuilder()
        .with_model(model_name)
        .with_arrival_process(arrival_process)
        .with_num_steps(num_steps)
        .with_seed(seed)
    )

    # Add duration or num_requests
    if duration is not None:
        builder.with_duration(duration)
    else:
        builder.with_num_requests(num_requests)

    return builder.build()
