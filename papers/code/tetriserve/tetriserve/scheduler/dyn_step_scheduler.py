from copy import deepcopy
import math
import time
from typing import List, Tuple, Dict
import networkx as nx

from tetriserve.xfuser.config import InputConfig
from tetriserve.xfuser.logger import init_logger
from .scheduler import Scheduler
from tetriserve.utils import get_current_time
from tetriserve.config.request import (
    ScheduledRequestsBlock,
    ScheduledRequests,
    UnScheduledRequest,
    ScheduledRequestsBlockCache,
)
from tetriserve.scheduler.perf_model import E2ECostEstimator
from tetriserve.utils import get_gpu_name

# Try to import C++ optimizer
try:
    from .cpp_optimizer import cpp_optimizer

    CPP_OPTIMIZER_AVAILABLE = True
except ImportError:
    CPP_OPTIMIZER_AVAILABLE = False

logger = init_logger(__name__)

# 7% tolerance for SLO compliance: latency <= target_slo * 1.07
SLO_TOLERANCE = 0.07

# Minimum steps a GPU configuration must run to avoid frequent switching overhead
MIN_STEPS_PER_CONFIG = 2


class DynStepScheduler(Scheduler):
    """
    Dynamic Step Scheduler for FluidServe.

    This scheduler implements various scheduling strategies for diffusion model inference,
    including fixed sequence parallelism, dynamic SLO-aware scheduling, and optimal shape scheduling.
    It supports both C++ and Python implementations for performance optimization.

    The scheduler manages GPU allocation, request batching, and step-level granular scheduling
    to maximize throughput while meeting latency SLOs.

    Attributes:
        schedule_cache (ScheduledRequestsBlockCache): Cache for scheduled request blocks
        schedule_logic (str): The scheduling strategy to use
        e2e_cost_estimator (E2ECostEstimator): Cost estimator for GPU configurations
        scheduled_requests (Dict[int, UnScheduledRequest]): Tracking of scheduled requests
        use_cpp_optimizer (bool): Whether to use C++ optimizer for performance
    """

    def __init__(
        self,
        ranks: List[int],
        schedule_logic: str,
        e2e_scaling_efficiency_path: str = "log/benchmark/scaling_efficiency/sd3/e2e_scaling_efficiency.json",
        use_cpp_optimizer: bool = True,
    ):
        """
        Initialize the DynStepScheduler.

        Args:
            ranks (List[int]): List of available GPU ranks for scheduling
            schedule_logic (str): Scheduling strategy to use. Options include:
                - "data_parallel": Use single GPU per request
                - "fixed_sp_X": Use fixed sequence parallelism with X GPUs
                - "fcfs_optimal_shape": First-come-first-serve with optimal GPU allocation
                - "dyn_slo_schedule_global": Dynamic SLO-aware global scheduling (exhaustive search)
                - "dyn_slo_schedule_global_dp": Dynamic SLO-aware global scheduling (dynamic programming)
                - "dyn_sp_schedule": Dynamic sequence parallelism scheduling
            e2e_scaling_efficiency_path (str): Path to scaling efficiency data file
            use_cpp_optimizer (bool): Whether to use C++ optimizer for better performance
        """
        super().__init__(ranks)

        self.schedule_cache: ScheduledRequestsBlockCache = ScheduledRequestsBlockCache()
        self.schedule_logic = schedule_logic
        self.e2e_cost_estimator = E2ECostEstimator(len(self.ranks), e2e_scaling_efficiency_path)

        self.scheduled_requests: Dict[int, UnScheduledRequest] = {}  # (req_id: UnScheduledRequest)

        # Cache for get_split_batch_time results: key=(bs, seq_len, running_steps, gpu_num), value=(ulysses_degree, ring_degree, estimated_time)
        self._split_batch_time_cache: Dict[Tuple[int, int, int, int], Tuple[int, int, float]] = {}

        # Use C++ optimizer if available and requested
        self.use_cpp_optimizer = use_cpp_optimizer and CPP_OPTIMIZER_AVAILABLE

    def schedule(self) -> List[ScheduledRequestsBlock]:
        """
        Main scheduling method that dispatches to the appropriate scheduling strategy.

        Returns:
            List[ScheduledRequestsBlock]: List of scheduled request blocks ready for execution
        """
        ret_req_list: List[ScheduledRequestsBlock] = []

        if self.schedule_logic.startswith("fixed_sp_"):
            sp_degree = int(self.schedule_logic.split("_")[-1])
            ret_req_list = self._fixed_sp_schedule_internal(sp_degree)
        elif self.schedule_logic == "data_parallel":
            ret_req_list = self._fixed_sp_schedule_internal(1)
        elif self.schedule_logic == "fcfs_optimal_shape":
            ret_req_list = self.fcfs_optimal_shape_schedule()
        elif self.schedule_logic == "per_resolution_schedule":
            ret_req_list = self.per_resolution_schedule()
        elif self.schedule_logic.startswith("dyn_slo_schedule_global"):
            ret_req_list = self.dyn_slo_schedule_global()
        elif self.schedule_logic == "dyn_sp_schedule":
            ret_req_list = self.dyn_sp_schedule()
        elif self.schedule_logic == "per_resolution_schedule_1_1_4_8":
            ret_req_list = self.per_resolution_schedule("1_1_4_8")
        elif self.schedule_logic == "per_resolution_schedule_1_1_2_8":
            ret_req_list = self.per_resolution_schedule("1_1_2_8")
        elif self.schedule_logic == "per_resolution_schedule_1_1_2_4":
            ret_req_list = self.per_resolution_schedule("1_1_2_4")
        elif self.schedule_logic == "dyn_slo_schedule_exhaustive_optimal":
            ret_req_list = self.dyn_slo_schedule_exhaustive_optimal()
        else:
            raise ValueError(f"Unsupported schedule_logic or internal method not implemented: {self.schedule_logic}")

        self.schedule_idx_counter += 1

        self._is_new_request_coming = False

        return ret_req_list

    def per_resolution_schedule(self, gpu_config: str = "1_1_2_8") -> List[ScheduledRequestsBlock]:
        """
        Per-resolution scheduling.
        This method schedules requests based on their resolution.
        for 256x256, 512x512, 1024x1024, 2048x2048, we will use the following GPU configurations:
        - 256x256: 1 GPU
        - 512x512: 2 GPUs
        - 1024x1024: 4 GPUs
        - 2048x2048: 8 GPUs
        Returns:
            List[ScheduledRequestsBlock]: List of scheduled request blocks
        """
        if not self.request_tracker.get_requests():
            return []

        ranks = deepcopy(self.ranks)
        ret_blocks = []
        unscheduled_reqs = self.request_tracker.get_merged_requests()

        # Remove requests that are already running
        to_remove_index = []
        for req_index, unscheduled_req in enumerate(unscheduled_reqs):
            if set(unscheduled_req.sub_req_ids) & set(self.running_requests):
                to_remove_index.append(req_index)

        unscheduled_reqs = [
            unscheduled_req
            for req_index, unscheduled_req in enumerate(unscheduled_reqs)
            if req_index not in to_remove_index
        ]

        # Process requests in FCFS order
        while len(ranks) > 0 and unscheduled_reqs:
            unscheduled_req: UnScheduledRequest = unscheduled_reqs.pop(0)

            # Calculate the resolution from input config
            height = unscheduled_req.input_config.height
            width = unscheduled_req.input_config.width
            resolution = height * width

            # Map resolution to GPU count based on the resolution-to-GPU mapping
            if gpu_config == "1_1_4_8":
                if resolution <= 256 * 256:
                    gpu_num = 1
                elif resolution <= 512 * 512:
                    gpu_num = 1
                elif resolution <= 1024 * 1024:
                    gpu_num = 4
                else:  # resolution > 1024 * 1024 (including 2048x2048 and larger)
                    gpu_num = 8
            elif gpu_config == "1_1_2_8":
                if resolution <= 256 * 256:
                    gpu_num = 1
                elif resolution <= 512 * 512:
                    gpu_num = 1
                elif resolution <= 1024 * 1024:
                    gpu_num = 2
                else:  # resolution > 1024 * 1024 (including 2048x2048 and larger)
                    gpu_num = 8
            elif gpu_config == "1_1_2_4":
                if resolution <= 256 * 256:
                    gpu_num = 1
                elif resolution <= 512 * 512:
                    gpu_num = 1
                elif resolution <= 1024 * 1024:
                    gpu_num = 2
                else:  # resolution > 1024 * 1024 (including 2048x2048 and larger)
                    gpu_num = 4
            else:
                raise ValueError(f"Unsupported GPU configuration: {gpu_config}")

            # Skip if not enough GPUs available
            if gpu_num > len(ranks):
                continue

            # Estimate cost for the chosen GPU configuration
            u, r, est_time = self.e2e_cost_estimator.estimate_cost(
                bs=unscheduled_req.input_config.batch_size,
                image_size=resolution,
                running_steps=unscheduled_req.input_config.num_inference_steps,
                gpu_num=gpu_num,
                fixed_gpu_num=gpu_num,
            )

            # Create scheduled request
            sched_req = ScheduledRequests(
                req_ids=[unscheduled_req.req_id],
                sub_req_ids=unscheduled_req.sub_req_ids,
                parallel_ranks=ranks[:gpu_num],
                attn_ulysses_degree=u,
                attn_ring_degree=r,
                sp_degree=gpu_num,
                origin_steps=unscheduled_req.origin_steps,
                start_step=0,
                end_step=unscheduled_req.input_config.num_inference_steps,
                latent_length=unscheduled_req.latent_length,
                input_config=unscheduled_req.input_config,
                is_last_steps=[True] * unscheduled_req.input_config.batch_size,
                arrival_time=unscheduled_req.arrival_time,
                latency_threshold=unscheduled_req.latency_threshold,
                estimated_time=est_time,
            )

            block = ScheduledRequestsBlock(
                requests=[sched_req],
                begin_condition_ranks=ranks[:gpu_num],
                end_condition_req_ids=[unscheduled_req.req_id],
            )
            ret_blocks.append(block)

            # Remove used ranks
            ranks = ranks[gpu_num:]

        return ret_blocks

    def dyn_sp_schedule(self) -> List[ScheduledRequestsBlock]:
        """
        Dynamic Sequence Parallelism scheduling.

        This method dynamically selects the optimal number of GPUs for each request based on:
        1. SLO compliance: Whether the configuration can meet the latency threshold,
        2. If we could meet the latency threshold, we will use the minimum number of GPUs that can meet the SLO
        3. If we could not meet the latency threshold, we will use the single GPU configuration

        Returns:
            List[ScheduledRequestsBlock]: List of scheduled request blocks with optimal GPU allocation
        """
        if not self.request_tracker.get_requests():
            return []

        ranks = deepcopy(self.ranks)
        ret_blocks = []
        unscheduled_reqs = self.request_tracker.get_merged_requests()

        # Remove requests that are already running
        to_remove_index = []
        for req_index, unscheduled_req in enumerate(unscheduled_reqs):
            if set(unscheduled_req.sub_req_ids) & set(self.running_requests):
                to_remove_index.append(req_index)

        unscheduled_reqs = [
            unscheduled_req
            for req_index, unscheduled_req in enumerate(unscheduled_reqs)
            if req_index not in to_remove_index
        ]

        # Process requests in FCFS order
        while len(ranks) > 0 and unscheduled_reqs:
            unscheduled_req: UnScheduledRequest = unscheduled_reqs.pop(0)

            # Get current time remaining for SLO
            current_time = get_current_time()
            time_remaining = unscheduled_req.latency_threshold + unscheduled_req.arrival_time - current_time

            # Test different GPU configurations in order of preference
            gpu_configs = [1, 2, 4, 8]
            selected_gpu_num = 1  # Default to single GPU

            # Get execution parameters
            batch_size = unscheduled_req.input_config.batch_size
            image_size = unscheduled_req.input_config.height * unscheduled_req.input_config.width
            num_steps = unscheduled_req.input_config.num_inference_steps

            # Find configurations that can meet SLO
            slo_meeting_configs = []
            for gpu_num in gpu_configs:
                # Skip if not enough GPUs available
                if gpu_num > len(ranks):
                    continue

                # Get execution time for this GPU configuration
                time_per_step = self._get_time_per_step(batch_size, image_size, gpu_num)
                if time_per_step < 0:  # Configuration not available
                    continue

                total_time = time_per_step * num_steps

                # Check if configuration meets SLO
                if total_time <= time_remaining:
                    slo_meeting_configs.append((gpu_num, total_time))

            # Decision logic based on SLO compliance
            if slo_meeting_configs:
                # Rule 2: If we can meet SLO, use minimum number of GPUs that can meet it
                selected_gpu_num = min(slo_meeting_configs, key=lambda x: x[0])[0]
            else:
                # Rule 3: If we cannot meet SLO, use single GPU configuration
                selected_gpu_num = 1

            # Skip if not enough GPUs available for the chosen configuration
            if selected_gpu_num > len(ranks):
                continue

            # Estimate cost for the chosen configuration
            u, r, est_time = self.e2e_cost_estimator.estimate_cost(
                bs=batch_size,
                image_size=image_size,
                running_steps=num_steps,
                gpu_num=selected_gpu_num,
                fixed_gpu_num=selected_gpu_num,
            )

            # Create scheduled request
            sched_req = ScheduledRequests(
                req_ids=[unscheduled_req.req_id],
                sub_req_ids=unscheduled_req.sub_req_ids,
                parallel_ranks=ranks[:selected_gpu_num],
                attn_ulysses_degree=u,
                attn_ring_degree=r,
                sp_degree=selected_gpu_num,
                origin_steps=unscheduled_req.origin_steps,
                start_step=0,
                end_step=unscheduled_req.input_config.num_inference_steps,
                latent_length=unscheduled_req.latent_length,
                input_config=unscheduled_req.input_config,
                is_last_steps=[True] * unscheduled_req.input_config.batch_size,
                arrival_time=unscheduled_req.arrival_time,
                latency_threshold=unscheduled_req.latency_threshold,
                estimated_time=est_time,
            )

            block = ScheduledRequestsBlock(
                requests=[sched_req],
                begin_condition_ranks=ranks[:selected_gpu_num],
                end_condition_req_ids=[unscheduled_req.req_id],
            )
            ret_blocks.append(block)

            # Remove used ranks
            ranks = ranks[selected_gpu_num:]

        return ret_blocks

    def _fixed_sp_schedule_internal(self, gpu_num: int = None) -> List[ScheduledRequestsBlock]:
        """
        Internal method for fixed sequence parallelism scheduling.

        Schedules requests using a fixed number of GPUs per request in FCFS order.

        Args:
            gpu_num (int, optional): Number of GPUs to use per request.
                                   If None, uses all available GPUs.

        Returns:
            List[ScheduledRequestsBlock]: List of scheduled request blocks
        """
        if not self.request_tracker.get_requests():
            return []

        ranks = deepcopy(self.ranks)
        if gpu_num is None:
            gpu_num = len(ranks)
        if gpu_num > len(ranks):
            return []

        ret_blocks = []
        unscheduled_reqs = self.request_tracker.get_merged_requests()

        to_remove_index = []
        for req_index, unscheduled_req in enumerate(unscheduled_reqs):
            if set(unscheduled_req.sub_req_ids) & set(self.running_requests):
                to_remove_index.append(req_index)

        unscheduled_reqs = [
            unscheduled_req
            for req_index, unscheduled_req in enumerate(unscheduled_reqs)
            if req_index not in to_remove_index
        ]

        while len(ranks) >= gpu_num and unscheduled_reqs:
            unscheduled_req: UnScheduledRequest = unscheduled_reqs.pop(0)

            u, r, est_time = self.e2e_cost_estimator.estimate_cost(
                bs=unscheduled_req.input_config.batch_size,
                image_size=unscheduled_req.input_config.height * unscheduled_req.input_config.width,
                running_steps=unscheduled_req.input_config.num_inference_steps,
                gpu_num=gpu_num,
                fixed_gpu_num=gpu_num,
            )

            sched_req = ScheduledRequests(
                req_ids=[unscheduled_req.req_id],
                sub_req_ids=unscheduled_req.sub_req_ids,
                parallel_ranks=ranks[:gpu_num],
                attn_ulysses_degree=u,
                attn_ring_degree=r,
                sp_degree=gpu_num,
                origin_steps=unscheduled_req.origin_steps,
                start_step=0,
                end_step=unscheduled_req.input_config.num_inference_steps,
                latent_length=unscheduled_req.latent_length,
                input_config=unscheduled_req.input_config,
                is_last_steps=[True] * unscheduled_req.input_config.batch_size,
                arrival_time=unscheduled_req.arrival_time,
                latency_threshold=unscheduled_req.latency_threshold,
                estimated_time=est_time,
            )

            block = ScheduledRequestsBlock(
                requests=[sched_req],
                begin_condition_ranks=ranks[:gpu_num],
                end_condition_req_ids=[unscheduled_req.req_id],
            )
            ret_blocks.append(block)

            ranks = ranks[gpu_num:]

        return ret_blocks

    def get_optimal_shape(self, current_world_size: int, unscheduled_req: UnScheduledRequest, num_inference_steps: int = None) -> Tuple[float, List[Tuple[int, int]], float]:
        """
        Find optimal GPU allocation shape for a request.

        Find the optimal allocation of inference steps across
        different GPU configurations (1, 2, 4, 8 GPUs) to minimize total GPU-hours while
        meeting latency constraints.

        Args:
            current_world_size (int): Number of available GPUs
            unscheduled_req (UnScheduledRequest): Request to schedule
            num_inference_steps (int, optional): Number of inference steps.
                                               If None, uses request's num_inference_steps

        Returns:
            Tuple[float, List[Tuple[int, int]], float]:
                - Total GPU hours required
                - List of (gpu_count, steps) tuples for optimal allocation
                - Total execution time
        """
        if num_inference_steps is None:
            num_inference_steps = unscheduled_req.input_config.num_inference_steps

        batch_size = unscheduled_req.input_config.batch_size
        image_size = unscheduled_req.input_config.height * unscheduled_req.input_config.width
        latency_threshold = unscheduled_req.latency_threshold + unscheduled_req.arrival_time - get_current_time()

        # Get execution times for each GPU configuration (reused by both C++ and Python)
        time_1gpu = self._get_time_per_step(batch_size, image_size, 1)
        time_2gpu = self._get_time_per_step(batch_size, image_size, 2)
        time_4gpu = self._get_time_per_step(batch_size, image_size, 4)
        time_8gpu = self._get_time_per_step(batch_size, image_size, 8)

        return self.get_optimal_shape_python(
            unscheduled_req.req_id,
            current_world_size,
            num_inference_steps,
            latency_threshold,
            time_1gpu,
            time_2gpu,
            time_4gpu,
            time_8gpu,
        )

    def get_optimal_shape_python(
        self,
        req_id: int,
        current_world_size: int,
        num_inference_steps: int,
        latency_threshold: float,
        time_1gpu: float,
        time_2gpu: float,
        time_4gpu: float,
        time_8gpu: float,
    ) -> Tuple[float, List[Tuple[int, int]], float]:
        """
        Python implementation of optimal shape finding algorithm.

        Uses dynamic programming to find the optimal allocation of inference steps across
        different GPU configurations to minimize total GPU-hours while meeting latency constraints.

        Now includes a minimum step constraint to ensure each GPU configuration gets enough
        work to justify the communication and synchronization overhead.

        Args:
            current_world_size (int): Number of available GPUs
            num_inference_steps (int): Total number of inference steps to schedule
            latency_threshold (float): Maximum allowed latency in seconds
            time_1gpu (float): Time per step for single GPU configuration
            time_2gpu (float): Time per step for 2-GPU configuration
            time_4gpu (float): Time per step for 4-GPU configuration
            time_8gpu (float): Time per step for 8-GPU configuration

        Returns:
            Tuple[float, List[Tuple[int, int]], float]:
                - Total GPU hours required
                - List of (gpu_count, steps) tuples for optimal allocation
                - Total execution time
        """
        # Minimum steps required for each GPU configuration to justify overhead
        if get_gpu_name() == "H100_HBM3":
            MIN_STEPS_PER_CONFIG = 4
        else:
            MIN_STEPS_PER_CONFIG = 1

        best_gpu_hour = float("inf")
        best_shape = [0, 0, 0, 0]
        best_total_time = float("inf")

        failed_gpu_hour = float("inf")
        failed_shape = [0, 0, 0, 0]
        failed_total_time = float("inf")

        # Execution times per step for each GPU configuration
        times_per_step = [time_1gpu, time_2gpu, time_4gpu, time_8gpu]
        gpu_counts = [1, 2, 4, 8]

        for r1 in range(0, num_inference_steps + 1):
            for r2 in range(0, num_inference_steps - r1 + 1):
                for r4 in range(0, num_inference_steps - r1 - r2 + 1):
                    r8 = num_inference_steps - r1 - r2 - r4
                    rank_shape = [r1, r2, r4, r8]

                    # Check basic constraints
                    if r8 < 0:
                        continue
                    if r2 + r4 + r8 > 0 and current_world_size < 2:
                        continue
                    if r4 + r8 > 0 and current_world_size < 4:
                        continue
                    if r8 > 0 and current_world_size < 8:
                        continue

                    # Apply minimum step constraint: if a configuration is used, it needs minimum steps
                    if r1 > 0 and r1 < MIN_STEPS_PER_CONFIG:
                        continue
                    if r2 > 0 and r2 < MIN_STEPS_PER_CONFIG:
                        continue
                    if r4 > 0 and r4 < MIN_STEPS_PER_CONFIG:
                        continue
                    if r8 > 0 and r8 < MIN_STEPS_PER_CONFIG:
                        continue

                    # Check if configurations are available (use -1 to indicate unavailable)
                    skip = False
                    for i, (steps, time_per_step) in enumerate(zip(rank_shape, times_per_step)):
                        if steps > 0 and time_per_step < 0:
                            skip = True
                            break
                    if skip:
                        continue

                    # Calculate times using precomputed values
                    r1_time = 0 if r1 == 0 else time_1gpu * r1
                    r2_time = 0 if r2 == 0 else time_2gpu * r2
                    r4_time = 0 if r4 == 0 else time_4gpu * r4
                    r8_time = 0 if r8 == 0 else time_8gpu * r8

                    total_time = r1_time + r2_time + r4_time + r8_time
                    total_gpu_hour = r1_time + r2_time * 2 + r4_time * 4 + r8_time * 8

                    # if total_time / num_inference_steps * (num_inference_steps + 1) > latency_threshold:
                    if total_time > latency_threshold:
                        # Cannot meet latency requirement
                        if total_time < failed_total_time:
                            failed_gpu_hour = total_gpu_hour
                            failed_shape = rank_shape
                            failed_total_time = total_time
                        continue

                    if total_gpu_hour < best_gpu_hour:
                        best_gpu_hour = total_gpu_hour
                        best_shape = rank_shape
                        best_total_time = total_time

        # If no valid solution found with minimum step constraint, try without it
        if best_gpu_hour == float("inf"):
            logger.info(
                f"No solution found with minimum {MIN_STEPS_PER_CONFIG} steps for req id {req_id}, trying without constraint..."
            )

            for r1 in range(0, num_inference_steps + 1):
                for r2 in range(0, num_inference_steps - r1 + 1):
                    for r4 in range(0, num_inference_steps - r1 - r2 + 1):
                        r8 = num_inference_steps - r1 - r2 - r4
                        rank_shape = [r1, r2, r4, r8]

                        # Check basic constraints only (no minimum step constraint)
                        if r8 < 0:
                            continue
                        if r2 + r4 + r8 > 0 and current_world_size < 2:
                            continue
                        if r4 + r8 > 0 and current_world_size < 4:
                            continue
                        if r8 > 0 and current_world_size < 8:
                            continue

                        # Check if configurations are available
                        skip = False
                        for i, (steps, time_per_step) in enumerate(zip(rank_shape, times_per_step)):
                            if steps > 0 and time_per_step < 0:
                                skip = True
                                break
                        if skip:
                            continue

                        # Calculate times
                        r1_time = 0 if r1 == 0 else time_1gpu * r1
                        r2_time = 0 if r2 == 0 else time_2gpu * r2
                        r4_time = 0 if r4 == 0 else time_4gpu * r4
                        r8_time = 0 if r8 == 0 else time_8gpu * r8

                        total_time = r1_time + r2_time + r4_time + r8_time
                        total_gpu_hour = r1_time + r2_time * 2 + r4_time * 4 + r8_time * 8

                        # if total_time / num_inference_steps * (num_inference_steps + 1) > latency_threshold:
                        if total_time > latency_threshold:
                            if total_time < failed_total_time:
                                failed_gpu_hour = total_gpu_hour
                                failed_shape = rank_shape
                                failed_total_time = total_time
                            continue

                        if total_gpu_hour < best_gpu_hour:
                            best_gpu_hour = total_gpu_hour
                            best_shape = rank_shape
                            best_total_time = total_time

        # All configs failed to meet the goal, use the best failed solution
        if best_gpu_hour == float("inf"):
            best_gpu_hour = failed_gpu_hour
            best_shape = failed_shape
            best_total_time = failed_total_time

        # Convert to expected format
        optimal_shape_list = []
        for i, steps in enumerate(best_shape):
            if steps > 0:
                optimal_shape_list.append((gpu_counts[i], steps))

        total_gpu_hours = best_gpu_hour / 3600.0
        return total_gpu_hours, optimal_shape_list, best_total_time

    def _get_time_per_step(self, batch_size: int, image_size: int, gpu_num: int) -> float:
        """
        Get execution time per step for a given GPU configuration.

        Args:
            batch_size (int): Batch size for the request
            image_size (int): Image size (height * width)
            gpu_num (int): Number of GPUs to use

        Returns:
            float: Time per step in seconds, or -1.0 if configuration is unavailable
        """
        try:
            u, r, time = self.e2e_cost_estimator.estimate_cost(
                bs=batch_size,
                image_size=image_size,
                running_steps=1,
                gpu_num=gpu_num,
                fixed_gpu_num=gpu_num,
                is_text_encoding_time_included=False,
            )
            return time
        except (KeyError, Exception):
            return -1.0  # Indicate unavailable configuration

    def fcfs_optimal_shape_schedule(self) -> List[ScheduledRequestsBlock]:
        """
        First-Come-First-Serve optimal shape scheduling.

        For each request in FCFS order, this method:
        1. Finds the optimal GPU allocation using get_optimal_shape()
        2. Schedules the first stage of the optimal allocation
        3. Remaining stages will be scheduled in subsequent calls

        This approach allows for dynamic stage-by-stage scheduling while maintaining
        optimal resource utilization.

        Returns:
            List[ScheduledRequestsBlock]: List of scheduled request blocks for the first stage
        """
        if not self.request_tracker.get_requests():
            return []

        ranks = deepcopy(self.ranks)
        ret_blocks = []
        unscheduled_reqs = self.request_tracker.get_merged_requests()

        # Remove requests that are already running
        to_remove_index = []
        for req_index, unscheduled_req in enumerate(unscheduled_reqs):
            if set(unscheduled_req.sub_req_ids) & set(self.running_requests):
                to_remove_index.append(req_index)

        unscheduled_reqs = [
            unscheduled_req
            for req_index, unscheduled_req in enumerate(unscheduled_reqs)
            if req_index not in to_remove_index
        ]

        cnt = 0
        while len(ranks) > 0 and unscheduled_reqs and cnt < len(unscheduled_reqs):

            unscheduled_req: UnScheduledRequest = unscheduled_reqs[cnt]
            cnt += 1

            best_gpu_hours, best_shape, best_total_time = self.get_optimal_shape(
                current_world_size=len(ranks),
                unscheduled_req=unscheduled_req,
            )

            first_stage_gpu_count = best_shape[-1][0]
            first_stage_steps = best_shape[-1][1]

            if len(ranks) < first_stage_gpu_count:
                continue

            image_size = unscheduled_req.input_config.height * unscheduled_req.input_config.width
            u, r, est_time = self.e2e_cost_estimator.estimate_cost(
                bs=unscheduled_req.input_config.batch_size,
                image_size=image_size,
                running_steps=first_stage_steps,
                gpu_num=first_stage_gpu_count,
                fixed_gpu_num=first_stage_gpu_count,
            )

            stage_input_config = deepcopy(unscheduled_req.input_config)
            stage_input_config.num_inference_steps = first_stage_steps

            # Compute start_step from how many steps have already been completed
            completed_steps = unscheduled_req.origin_steps[0] - unscheduled_req.input_config.num_inference_steps

            sched_req = ScheduledRequests(
                req_ids=[unscheduled_req.req_id],
                sub_req_ids=unscheduled_req.sub_req_ids,
                parallel_ranks=ranks[:first_stage_gpu_count],
                attn_ulysses_degree=u,
                attn_ring_degree=r,
                sp_degree=first_stage_gpu_count,
                latent_length=unscheduled_req.latent_length,
                input_config=stage_input_config,
                origin_steps=unscheduled_req.origin_steps,
                start_step=completed_steps,
                end_step=completed_steps + first_stage_steps,
                is_last_steps=[first_stage_steps == unscheduled_req.input_config.num_inference_steps]
                * unscheduled_req.input_config.batch_size,
                arrival_time=unscheduled_req.arrival_time,
                latency_threshold=unscheduled_req.latency_threshold,
                estimated_time=est_time,
            )

            stage_block = ScheduledRequestsBlock(
                requests=[sched_req],
                begin_condition_ranks=ranks[:first_stage_gpu_count],
                # end_condition_req_ids=[unscheduled_req.req_id] if stage_idx == len(best_shape) - 1 else [], # NOTE: seems like we don't need this for now
            )

            ret_blocks.append(stage_block)

            ranks = ranks[first_stage_gpu_count:]

        return ret_blocks

    def get_split_batch(self, batch_size: int, seq_len: int, gpu_num: int) -> Tuple[int, int]:
        """
        Calculate optimal batch splitting across GPUs.

        Determines how to split batch processing across available GPUs:
        - If batch_size >= gpu_num: Split batch evenly across GPUs
        - If batch_size < gpu_num: Use multiple GPUs per batch item

        Args:
            batch_size (int): Total batch size
            seq_len (int): Sequence length (unused in current implementation)
            gpu_num (int): Number of available GPUs

        Returns:
            Tuple[int, int]: (batch_size_per_gpu, gpus_per_prompt)
        """

        splitable = False
        if batch_size > 1 and gpu_num > 1:
            splitable = True

        if splitable:
            if batch_size >= gpu_num:
                # Case 1: Batch size is larger than or equal to GPU number
                # Divide batch size evenly across all GPUs
                bs_per_gpu = batch_size // gpu_num
                gpu_num_per_prompt = 1
            else:
                # Case 2: Batch size is smaller than GPU number
                bs_per_gpu = 1
                gpu_num_per_prompt = gpu_num // batch_size
        else:
            bs_per_gpu = batch_size
            gpu_num_per_prompt = gpu_num

        return bs_per_gpu, gpu_num_per_prompt

    def get_split_batch_time(self, bs: int, seq_len: int, running_steps: int, gpu_num: int) -> Tuple[int, int, float]:
        """
        Get execution time for split batch configuration.

        Results are cached to avoid recomputing for the same input arguments.

        Args:
            bs (int): Batch size
            seq_len (int): Sequence length
            running_steps (int): Number of inference steps
            gpu_num (int): Number of GPUs

        Returns:
            Tuple: (ulysses_degree, ring_degree, estimated_time)
        """
        bs_per_gpu, gpu_num_per_prompt = self.get_split_batch(bs, seq_len, gpu_num)

        result = self.e2e_cost_estimator.estimate_cost(
            bs=bs_per_gpu,
            image_size=seq_len,
            running_steps=running_steps,
            is_text_encoding_time_included=False,
            gpu_num=gpu_num_per_prompt,
            fixed_gpu_num=gpu_num_per_prompt,
        )

        return result

    def update_scheduled_request(self, unscheduled_req: UnScheduledRequest):
        """
        Update internal tracking of scheduled requests.

        Args:
            unscheduled_req (UnScheduledRequest): Request to track
        """
        if unscheduled_req.req_id not in self.scheduled_requests:
            self.scheduled_requests[unscheduled_req.req_id] = deepcopy(unscheduled_req)

    def _filter_and_prepare_requests(self) -> List[UnScheduledRequest]:
        """
        Filter out requests that are already running and limit to available GPU count.

        Returns:
            List[UnScheduledRequest]: Filtered list of unscheduled requests
        """
        unscheduled_reqs = self.request_tracker.get_merged_requests()

        # Remove requests that are already running
        to_remove_index = []
        for req_index, unscheduled_req in enumerate(unscheduled_reqs):
            if set(unscheduled_req.sub_req_ids) & set(self.running_requests):
                to_remove_index.append(req_index)

        unscheduled_reqs = [
            unscheduled_req
            for req_index, unscheduled_req in enumerate(unscheduled_reqs)
            if req_index not in to_remove_index
        ]

        # Limit to number of available GPUs
        current_world_size = len(self.ranks)
        unscheduled_reqs = unscheduled_reqs[:current_world_size]

        return unscheduled_reqs

    def _group_requests_by_sequence_length(
        self, unscheduled_reqs: List[UnScheduledRequest]
    ) -> Dict[int, List[UnScheduledRequest]]:
        """
        Group requests by their sequence length (height * width) and sort by latency threshold.

        Args:
            unscheduled_reqs: List of unscheduled requests to group

        Returns:
            Dict mapping sequence length to sorted list of requests (by latency threshold)
        """
        reqs_dict: Dict[int, List[UnScheduledRequest]] = {}

        for req in unscheduled_reqs:
            seq_len = req.input_config.height * req.input_config.width
            if seq_len not in reqs_dict:
                reqs_dict[seq_len] = []
            reqs_dict[seq_len].append(req)

        # Sort each group by latency threshold (lower threshold = higher priority)
        for seq_len in reqs_dict:
            reqs_dict[seq_len].sort(key=lambda req: req.latency_threshold)

        return reqs_dict

    def _compute_optimal_shapes(self, unscheduled_reqs: List[UnScheduledRequest]) -> List[List[int]]:
        """
        Compute optimal GPU allocation shapes for each request.

        The optimal shape is a list of 4 integers representing steps allocated to
        1, 2, 4, and 8 GPU configurations respectively.

        Args:
            unscheduled_reqs: List of unscheduled requests

        Returns:
            List of optimal shapes, where each shape is [steps_1gpu, steps_2gpu, steps_4gpu, steps_8gpu]
        """
        optimal_shape = []
        current_world_size = len(self.ranks)

        for req in unscheduled_reqs:
            best_gpu_time, best_shape, best_total_time = self.get_optimal_shape(
                current_world_size=current_world_size,
                unscheduled_req=req,
            )

            # Convert best_shape from [(gpu_count, steps), ...] to [steps_1gpu, steps_2gpu, steps_4gpu, steps_8gpu]
            converted_best_shape = [0, 0, 0, 0]
            for gpu_count, steps in best_shape:
                gpu_log2 = int(math.log2(gpu_count))
                converted_best_shape[gpu_log2] = steps

            optimal_shape.append(converted_best_shape)

        return optimal_shape

    def _calculate_round_duration(
        self, shape_state: List[List[int]], unscheduled_reqs: List[UnScheduledRequest]
    ) -> float:
        """
        Calculate the round duration based on step granularity of all remaining requests.

        The round duration is the maximum time needed to complete one step granularity
        for any request with remaining work.

        Args:
            shape_state: Current shape state for each request
            unscheduled_reqs: List of unscheduled requests

        Returns:
            Round duration in seconds
        """
        max_round_duration = 0.0

        # Find maximum GPU need in shape_state for all remaining requests
        for req_idx in range(len(unscheduled_reqs)):
            if sum(shape_state[req_idx]) > 0:  # Only consider requests with remaining steps
                req = unscheduled_reqs[req_idx]
                seq_len = req.input_config.height * req.input_config.width
                req_bsz = req.input_config.batch_size

                # Calculate step granularity for this request based on GPU type and resolution
                if get_gpu_name() == "H100_HBM3":
                    if seq_len >= 1024 * 1024:
                        step_granularity = 4
                    else:
                        step_granularity = 10
                else:
                    if seq_len >= 1024 * 1024:
                        step_granularity = 2
                    else:
                        step_granularity = 20

                # Find the maximum GPU need for this request in shape_state
                for gpu_log2 in range(len(shape_state[req_idx])):
                    if shape_state[req_idx][gpu_log2] > 0:
                        gpu_num = 2**gpu_log2

                        # Calculate time for this request with step_granularity
                        req_time = self.get_split_batch_time(
                            bs=req_bsz, seq_len=seq_len, running_steps=step_granularity, gpu_num=gpu_num
                        )[2]
                        max_round_duration = max(max_round_duration, req_time)

        return max_round_duration

    def _scale_up_ranks_to_fill_bubble(
        self,
        best_combo_list: List[Tuple],
        shape_state: List[List[int]],
        unscheduled_reqs: List[UnScheduledRequest],
        best_rank_used: int,
        current_world_size: int,
    ) -> Tuple[int, List[Tuple]]:
        """
        Scale up GPU allocations to better utilize available GPUs when there's unused capacity.

        This attempts to improve performance by using more GPUs when it doesn't increase
        execution time, effectively "filling the bubble" of unused GPU capacity.

        Args:
            best_combo_list: Current best combination of requests to schedule
            shape_state: Current shape state for each request
            unscheduled_reqs: List of unscheduled requests
            best_rank_used: Currently used number of ranks
            current_world_size: Total available GPU count

        Returns:
            Tuple of (updated best_rank_used, updated best_combo_list)
        """
        # Check if scale-up is enabled for this GPU type and schedule logic
        should_scale_up = get_gpu_name() != "H100_HBM3" or (
            get_gpu_name() == "H100_HBM3"
            and (
                self.schedule_logic == "dyn_slo_schedule_global_scale_up"
                or self.schedule_logic == "dyn_slo_schedule_global_dp_scale_up"
            )
        )

        if not should_scale_up or best_rank_used >= current_world_size:
            return best_rank_used, best_combo_list

        updated_best_rank_used = best_rank_used

        # Try to scale up each combo if it improves performance
        for combo_idx in range(len(best_combo_list)):
            combo = best_combo_list[combo_idx]
            req_idx, rank_group, req_step, req_time = combo

            seq_len = unscheduled_reqs[req_idx].input_config.height * unscheduled_reqs[req_idx].input_config.width
            req_bsz = unscheduled_reqs[req_idx].input_config.batch_size

            rank_needed = len(rank_group)
            original_rank_needed = rank_needed

            # Try doubling the rank count if it doesn't exceed capacity and improves performance
            while updated_best_rank_used + rank_needed * 2 - len(rank_group) <= current_world_size:
                time_current = self._get_time_per_step(req_bsz, seq_len, rank_needed)
                time_doubled = self._get_time_per_step(req_bsz, seq_len, rank_needed * 2)

                # Stop if doubling doesn't improve performance
                if time_current <= time_doubled:
                    break
                # Don't scale up to 8 GPUs
                elif rank_needed * 2 == 8:
                    break
                else:
                    rank_needed *= 2

            # Update the combo if we scaled up
            if rank_needed != original_rank_needed:
                updated_best_rank_used += rank_needed - len(rank_group)

                # Update shape_state to reflect the new GPU allocation
                shape_state[req_idx][int(math.log2(len(rank_group)))] -= req_step
                rank_group = [0] * rank_needed
                shape_state[req_idx][int(math.log2(len(rank_group)))] += req_step

                # Recalculate execution time with new GPU count
                req_time = self.get_split_batch_time(
                    bs=req_bsz, seq_len=seq_len, running_steps=req_step, gpu_num=rank_needed
                )[2]

                best_combo_list[combo_idx] = (req_idx, tuple(rank_group), req_step, req_time)

        return updated_best_rank_used, best_combo_list

    def _assign_ranks_to_combos(
        self,
        best_combo_list: List[Tuple],
        rank_assign: List[int],
        current_world_size: int,
    ) -> List[List[int]]:
        """
        Assign GPU ranks to each combo, prioritizing ranks already assigned to the same request.

        This ensures continuity by reusing ranks that were previously assigned to the same request.

        Args:
            best_combo_list: List of combos to assign ranks to
            rank_assign: Current rank assignment mapping (rank_idx -> req_idx)
            current_world_size: Total available GPU count

        Returns:
            List of rank groups, one per combo
        """
        rank_group_new = [[] for _ in range(len(best_combo_list))]
        rank_reserved = [False] * current_world_size

        # Phase 1: Reserve ranks that are already assigned to the same request
        if "wo_placement" not in self.schedule_logic:
            for combo_idx in range(len(best_combo_list)):
                combo = best_combo_list[combo_idx]
                req_idx, rank_group, req_step, req_time = combo
                rank_needed = len(rank_group)

                for i in range(current_world_size):
                    if rank_assign[i] == req_idx:
                        rank_needed -= 1
                        rank_group_new[combo_idx].append(i)
                        rank_reserved[i] = True

                    if rank_needed == 0:
                        break

        # Phase 2: Assign remaining ranks from available pool
        for combo_idx in range(len(best_combo_list)):
            combo = best_combo_list[combo_idx]
            req_idx, rank_group, req_step, req_time = combo
            rank_needed = len(rank_group) - len(rank_group_new[combo_idx])

            if rank_needed == 0:
                continue

            for i in range(current_world_size):
                if not rank_reserved[i]:
                    rank_needed -= 1
                    rank_group_new[combo_idx].append(i)
                    rank_reserved[i] = True
                    rank_assign[i] = req_idx

                if rank_needed == 0:
                    break

        return rank_group_new

    def _adjust_steps_for_synchronization(
        self,
        best_combo_list: List[Tuple],
        rank_group_new: List[List[int]],
        rank_time: List[float],
        req_time_list: List[float],
        shape_state: List[List[int]],
        unscheduled_reqs: List[UnScheduledRequest],
        result_missed_list: List[int],
    ) -> Tuple[List[Tuple], float]:
        """
        Adjust the number of steps for each combo to ensure synchronization across ranks.

        This ensures all combos in a round finish at approximately the same time by
        reducing steps for combos that would finish later.

        Args:
            best_combo_list: List of combos to adjust
            rank_group_new: Assigned rank groups for each combo
            rank_time: Current time for each rank
            req_time_list: Current time for each request
            shape_state: Current shape state for each request
            unscheduled_reqs: List of unscheduled requests
            result_missed_list: List of already missed request indices

        Returns:
            Tuple of (revised_combo_list, new_best_time)
        """
        revised_combo_list = []
        new_best_time = float("inf")
        min_time_required = float("inf")

        # Calculate the minimum time required for all combos to finish
        for combo_idx in range(len(best_combo_list)):
            combo = best_combo_list[combo_idx]
            req_idx, rank_group, req_step, req_time = combo
            combo_previous_time = 0

            # Find the maximum previous time among assigned ranks
            for i in rank_group_new[combo_idx]:
                combo_previous_time = max(combo_previous_time, rank_time[i])

            min_time_required = min(min_time_required, combo_previous_time + req_time)

        # Adjust steps for each combo to fit within min_time_required
        for combo_idx in range(len(best_combo_list)):
            combo = best_combo_list[combo_idx]
            req_idx, rank_group, req_step, req_time = combo

            seq_len = unscheduled_reqs[req_idx].input_config.height * unscheduled_reqs[req_idx].input_config.width
            req_bsz = unscheduled_reqs[req_idx].input_config.batch_size
            max_step = 1

            # Calculate previous time for this combo's ranks
            combo_previous_time = 0
            for i in rank_group_new[combo_idx]:
                combo_previous_time = max(combo_previous_time, rank_time[i])

            # If previous time already exceeds minimum, can't schedule any steps
            if combo_previous_time >= min_time_required:
                max_step = 0
            else:
                # Find maximum steps that can be completed within time budget
                for step in range(1, req_step + 1):
                    step_time = self.get_split_batch_time(
                        bs=req_bsz, seq_len=seq_len, running_steps=step, gpu_num=len(rank_group)
                    )[2]
                    if combo_previous_time + step_time <= min_time_required:
                        max_step = step
                    else:
                        break

            # Calculate new execution time for adjusted steps
            new_req_time = self.get_split_batch_time(
                bs=req_bsz, seq_len=seq_len, running_steps=max_step, gpu_num=len(rank_group)
            )[2]

            # Update rank times and request times
            for i in rank_group_new[combo_idx]:
                rank_time[i] = combo_previous_time + new_req_time
                req_time_list[req_idx] = max(req_time_list[req_idx], rank_time[i])

            # Add to revised combo list if we can schedule at least one step
            if max_step > 0:
                revised_combo_list.append(
                    (
                        best_combo_list[combo_idx][0],
                        tuple(rank_group_new[combo_idx]),
                        max_step,
                        new_req_time,
                    )
                )
                new_best_time = min(new_best_time, new_req_time)

            # Update shape_state to reflect completed steps
            shape_state[req_idx][int(math.log2(len(rank_group)))] -= max_step

            # Check if request is complete and mark as missed if it exceeded latency threshold
            if sum(shape_state[req_idx]) == 0:
                if combo_previous_time + new_req_time > unscheduled_reqs[req_idx].latency_threshold:
                    result_missed_list.append(req_idx)

        return revised_combo_list, new_best_time

    def _get_step_granularity(self, seq_len: int) -> int:
        """
        Get the step granularity for a given sequence length based on GPU type.

        Step granularity determines how many steps are grouped together in a single
        scheduled block. This helps balance overhead vs. flexibility.

        Args:
            seq_len: Sequence length (height * width)

        Returns:
            Step granularity (number of steps per block)
        """
        if get_gpu_name() == "H100_HBM3":
            if seq_len > 2048 * 2048:
                return 2
            elif seq_len >= 1024 * 1024:
                return 4
            else:
                return 10
        else:
            if seq_len > 2048 * 2048:
                return 2
            elif seq_len >= 1024 * 1024:
                return 4
            else:
                return 10

    def _build_schedule_DAG(self, combo: List[Tuple], unscheduled_reqs: List[UnScheduledRequest]) -> nx.DiGraph:
        """
        Build a directed acyclic graph (DAG) representing the schedule dependencies.

        Each node in the DAG represents a scheduled request block. Edges represent
        dependencies where two blocks share GPU ranks and must be executed sequentially.

        Args:
            combo: List of (req_idx, rank_group, req_step, req_time) tuples
            unscheduled_reqs: List of unscheduled requests

        Returns:
            NetworkX DiGraph representing the schedule DAG
        """
        G = nx.DiGraph()
        acc_dict = {}  # Track accumulated steps per request
        node_idx = 0

        # Create nodes for each combo entry
        for req_idx, rank_group, req_step, req_time in combo:
            req_id = unscheduled_reqs[req_idx].req_id
            rank_group = [self.ranks[i] for i in rank_group]

            # Update scheduled requests tracking
            if req_id not in self.scheduled_requests:
                for req in unscheduled_reqs:
                    if req.req_id == req_id:
                        self.update_scheduled_request(req)
                        break

            # Get attention configuration (ulysses and ring degrees)
            u, r, _ = self.get_split_batch_time(
                bs=unscheduled_reqs[req_idx].input_config.batch_size,
                seq_len=unscheduled_reqs[req_idx].input_config.height * unscheduled_reqs[req_idx].input_config.width,
                running_steps=1,
                gpu_num=len(rank_group),
            )

            if req_idx not in acc_dict:
                acc_dict[req_idx] = 0

            seq_len = unscheduled_reqs[req_idx].input_config.height * unscheduled_reqs[req_idx].input_config.width
            step_granularity = self._get_step_granularity(seq_len)
            # Split request steps into blocks based on step granularity
            for j in range(0, req_step, step_granularity):
                steps_in_block = min(step_granularity, req_step - j)
                acc_dict[req_idx] += steps_in_block

                # Create input config for this block
                new_input_config = deepcopy(unscheduled_reqs[req_idx].input_config)
                new_input_config.num_inference_steps = steps_in_block

                # Validate that rank group is divisible by attention configuration
                assert len(rank_group) % (u * r) == 0, "split batch should be divisible by u * r"

                # Calculate batch splitting
                split_batch_cnt = len(rank_group) // (u * r)
                rank_degree = len(rank_group) // split_batch_cnt
                sub_req_degree = len(unscheduled_reqs[req_idx].sub_req_ids) // split_batch_cnt

                new_input_config.batch_size = new_input_config.batch_size // split_batch_cnt

                # Create scheduled requests for each batch split
                requests_list = []
                for i in range(0, split_batch_cnt):
                    scheduled_reqs = ScheduledRequests(
                        req_ids=[unscheduled_reqs[req_idx].req_id],
                        sub_req_ids=unscheduled_reqs[req_idx].sub_req_ids[
                            sub_req_degree * i : sub_req_degree * (i + 1)
                        ],
                        parallel_ranks=rank_group[rank_degree * i : rank_degree * (i + 1)],
                        attn_ulysses_degree=u,
                        attn_ring_degree=r,
                        sp_degree=u * r,
                        latent_length=unscheduled_reqs[req_idx].latent_length,
                        input_config=new_input_config,
                        origin_steps=unscheduled_reqs[req_idx].origin_steps,
                        start_step=acc_dict[req_idx] - steps_in_block,
                        end_step=acc_dict[req_idx],
                        is_last_steps=[acc_dict[req_idx] == unscheduled_reqs[req_idx].input_config.num_inference_steps]
                        * new_input_config.batch_size,
                        arrival_time=unscheduled_reqs[req_idx].arrival_time,
                        latency_threshold=unscheduled_reqs[req_idx].latency_threshold,
                        estimated_time=req_time * steps_in_block / req_step,
                    )
                    requests_list.append(scheduled_reqs)

                # Create request block and add to DAG
                req_block = ScheduledRequestsBlock(
                    requests=requests_list,
                    begin_condition_ranks=list(rank_group),
                    end_condition_req_ids=[unscheduled_reqs[req_idx].req_id],
                )
                G.add_node(node_idx, req_block=req_block)
                node_idx += 1

        # Add edges between nodes that share GPU ranks (creating dependencies)
        for node_idx_i in G.nodes:
            for node_idx_j in G.nodes:
                if node_idx_i >= node_idx_j:
                    continue
                req_block_i = G.nodes[node_idx_i]["req_block"]
                req_block_j = G.nodes[node_idx_j]["req_block"]
                if set(req_block_i.begin_condition_ranks) & set(req_block_j.begin_condition_ranks):
                    G.add_edge(node_idx_i, node_idx_j)

        return G

    def dyn_slo_schedule_global(self) -> List[ScheduledRequestsBlock]:
        """
        Dynamic SLO-aware global scheduling (Work in Progress).

        This method implements a global optimization approach for scheduling requests
        across multiple GPUs while considering SLO constraints. It uses:

        1. Request grouping by sequence length
        2. SLO-based priority ordering
        3. Optimal shape calculation for each request
        4. Greedy bin packing with exhaustive search or dynamic programming
        5. DAG-based scheduling for complex dependencies

        The algorithm attempts to minimize missed requests while maximizing GPU utilization
        and meeting latency thresholds.

        Returns:
            List[ScheduledRequestsBlock]: List of scheduled request blocks from the DAG

        Note:
            This is a complex experimental scheduling algorithm that may have performance
            implications for large request loads.
        """
        # Early return if no GPUs available
        if len(self.ranks) == 0:
            return []

        # Step 1: Filter and prepare requests
        # Remove requests that are already running and limit to available GPU count
        unscheduled_reqs = self._filter_and_prepare_requests()
        current_world_size = len(self.ranks)

        # Step 2: Group requests by sequence length and sort by latency threshold
        # This helps prioritize requests with tighter deadlines
        reqs_dict = self._group_requests_by_sequence_length(unscheduled_reqs)

        # Step 3: Compute optimal GPU allocation shapes for each request
        # The optimal shape determines how many steps should run on 1, 2, 4, or 8 GPUs
        optimal_shape = self._compute_optimal_shapes(unscheduled_reqs)
        optimal_shape = tuple(optimal_shape)

        # Step 4: Greedy bin packing algorithm
        # This iteratively schedules requests in rounds, updating optimal shapes as work progresses
        def greedy_bin_packing_exhaustive(shape_state):
            """
            Greedy bin packing algorithm that schedules requests in iterative rounds.

            Each round:
            1. Recomputes optimal shapes for remaining work
            2. Finds best combination of requests to schedule (using DP or exhaustive search)
            3. Scales up GPU allocations if beneficial
            4. Assigns ranks to combos
            5. Adjusts steps for synchronization
            6. Updates state and continues until all work is done

            Args:
                shape_state: Initial optimal shapes for all requests

            Returns:
                Tuple of (missed_request_count, total_time, final_combo_list)
            """
            # Initialize state
            shape_state = [list(x) for x in shape_state]
            result_missed_list = []
            result_time = 0
            result_combo = []
            rank_time = [0] * current_world_size
            req_time_list = [0] * len(unscheduled_reqs)
            rank_assign = [-1] * current_world_size

            # Main scheduling loop: continue until all requests are fully scheduled
            while not all(sum(x) == 0 for x in shape_state):
                # Recompute optimal shapes for remaining work
                # As steps are completed, the optimal allocation may change
                new_optimal_shape = []
                gpu_time_state = []

                for req_idx in range(len(unscheduled_reqs)):
                    if sum(shape_state[req_idx]) == 0:
                        # Request is complete, keep empty shape
                        new_optimal_shape.append(shape_state[req_idx])
                        gpu_time_state.append(0)
                        continue

                    req = unscheduled_reqs[req_idx]
                    remaining_steps = sum(shape_state[req_idx])

                    # Recompute optimal shape for remaining steps
                    best_gpu_time, best_shape, _ = self.get_optimal_shape(
                        current_world_size=len(self.ranks),
                        unscheduled_req=req,
                        num_inference_steps=remaining_steps,
                    )

                    # Convert to [steps_1gpu, steps_2gpu, steps_4gpu, steps_8gpu] format
                    converted_best_shape = [0, 0, 0, 0]
                    for gpu_count, steps in best_shape:
                        gpu_log2 = int(math.log2(gpu_count))
                        converted_best_shape[gpu_log2] = steps

                    gpu_time_state.append(best_gpu_time)
                    new_optimal_shape.append(converted_best_shape)

                shape_state = new_optimal_shape

                # Calculate round duration based on step granularity
                round_duration = self._calculate_round_duration(shape_state, unscheduled_reqs)

                # Find best combination of requests to schedule in this round
                # Use dynamic programming if schedule_logic contains "_dp", otherwise exhaustive search
                use_dp = "_dp" in self.schedule_logic
                combo_search_start_time = time.time()
                (
                    best_missed,
                    best_missed_list,
                    best_filled,
                    best_gpu_time,
                    best_time,
                    best_rank_used,
                    best_combo_list,
                ) = self._combo_search_with_algorithm_selection(
                    shape_state,
                    unscheduled_reqs,
                    rank_time,
                    req_time_list,
                    result_missed_list,
                    gpu_time_state,
                    use_dp=use_dp,
                    round_duration=round_duration,
                )
                combo_search_end_time = time.time()
                combo_search_execution_time = combo_search_end_time - combo_search_start_time
                logger.info(
                    f"_combo_search_with_algorithm_selection execution time: {combo_search_execution_time:.4f} seconds"
                )

                # Sort combos by rank group size (largest first) for better allocation
                best_combo_list = sorted(best_combo_list, key=lambda x: len(x[1]), reverse=True)
                logger.info(f"best_combo_list: {best_combo_list}")

                # Scale up ranks to better utilize available GPUs
                best_rank_used, best_combo_list = self._scale_up_ranks_to_fill_bubble(
                    best_combo_list,
                    shape_state,
                    unscheduled_reqs,
                    best_rank_used,
                    current_world_size,
                )

                # Assign GPU ranks to each combo
                rank_group_new = self._assign_ranks_to_combos(best_combo_list, rank_assign, current_world_size)

                # Adjust steps for synchronization and update state
                revised_combo_list, new_best_time = self._adjust_steps_for_synchronization(
                    best_combo_list,
                    rank_group_new,
                    rank_time,
                    req_time_list,
                    shape_state,
                    unscheduled_reqs,
                    result_missed_list,
                )

                # Accumulate results
                result_combo += revised_combo_list
                result_missed_list = list(set(result_missed_list + best_missed_list))
                result_time += new_best_time

                break

            return len(result_missed_list), result_time, result_combo

        # Execute the greedy bin packing algorithm
        best_missed_reqs, best_time, best_combo = greedy_bin_packing_exhaustive(optimal_shape)

        # Step 5: Build schedule DAG
        # Convert the combo list into a directed acyclic graph where nodes represent
        # scheduled blocks and edges represent dependencies (shared GPU ranks)
        schedule_DAG = self._build_schedule_DAG(best_combo, unscheduled_reqs)
        self.schedule_cache.update_schedule_DAG(schedule_DAG)

        # Step 6: Extract ready-to-execute blocks from the DAG
        # These are blocks that have no dependencies (all required ranks are available)
        ret_scheduled_reqs_blocks = []
        free_reqs_blocks = self.schedule_cache.get_new_free_nodes()
        for reqs_block, node_idx in free_reqs_blocks:
            if reqs_block.meet_begin_condition(self.ranks):
                ret_scheduled_reqs_blocks.append(reqs_block)
                self.schedule_cache.remove_node(node_idx)

        return ret_scheduled_reqs_blocks

    def dyn_slo_schedule_exhaustive_optimal(self) -> List[ScheduledRequestsBlock]:
        """
        Exhaustive optimal SLO-aware scheduling with maximal SLO rate.

        This method performs an exhaustive search over all possible GPU allocations
        for each request to find the combination that maximizes SLO rate. For each
        request, it tries every possible way to allocate steps across GPU configurations
        (1, 2, 4, 8 GPUs), then evaluates all combinations to find the optimal solution.

        Note: This is an extremely expensive search with exponential time complexity.
        Use only for small numbers of requests or when optimal solution is critical.

        Returns:
            List[ScheduledRequestsBlock]: List of scheduled request blocks ready for execution
        """
        # Early return if no GPUs available
        if len(self.ranks) == 0:
            return []

        logger.info("Starting exhaustive optimal SLO scheduling")
        search_start_time = time.time()

        # Step 1: Filter and prepare requests
        unscheduled_reqs = self._filter_and_prepare_requests()
        current_world_size = len(self.ranks)

        if not unscheduled_reqs:
            logger.info("No requests to schedule")
            return []

        logger.info(f"Exhaustive search for {len(unscheduled_reqs)} requests with {current_world_size} GPUs")

        # Step 2: Generate all possible step allocations for each request
        # For each request, generate all ways to split its steps across GPU configs (1, 2, 4, 8 GPUs)
        def generate_all_step_allocations_for_request(req: UnScheduledRequest, num_steps: int) -> List[List[int]]:
            """
            Generate all possible step allocations for a request.
            Returns list of [steps_1gpu, steps_2gpu, steps_4gpu, steps_8gpu] allocations.
            """
            all_allocations = []

            # Generate all combinations of step allocations
            for r1 in range(num_steps + 1):
                for r2 in range(num_steps - r1 + 1):
                    for r4 in range(num_steps - r1 - r2 + 1):
                        r8 = num_steps - r1 - r2 - r4
                        if r8 < 0:
                            continue

                        # Check GPU availability constraints
                        if r2 + r4 + r8 > 0 and current_world_size < 2:
                            continue
                        if r4 + r8 > 0 and current_world_size < 4:
                            continue
                        if r8 > 0 and current_world_size < 8:
                            continue

                        # Check if configurations are available
                        seq_len = req.input_config.height * req.input_config.width
                        req_bsz = req.input_config.batch_size
                        valid_allocation = True

                        # Check if each GPU configuration is available
                        if r1 > 0:
                            time_1gpu = self._get_time_per_step(req_bsz, seq_len, 1)
                            if time_1gpu < 0:
                                valid_allocation = False
                        if r2 > 0:
                            time_2gpu = self._get_time_per_step(req_bsz, seq_len, 2)
                            if time_2gpu < 0:
                                valid_allocation = False
                        if r4 > 0:
                            time_4gpu = self._get_time_per_step(req_bsz, seq_len, 4)
                            if time_4gpu < 0:
                                valid_allocation = False
                        if r8 > 0:
                            time_8gpu = self._get_time_per_step(req_bsz, seq_len, 8)
                            if time_8gpu < 0:
                                valid_allocation = False

                        if valid_allocation:
                            all_allocations.append([r1, r2, r4, r8])

            return all_allocations

        # Generate allocation options for each request
        request_allocation_options = []
        for req in unscheduled_reqs:
            num_steps = req.input_config.num_inference_steps
            allocations = generate_all_step_allocations_for_request(req, num_steps)
            request_allocation_options.append(allocations)
            logger.info(f"Request {req.req_id}: {len(allocations)} possible step allocations for {num_steps} steps")

        # Step 3: Generate all combinations of allocations across requests
        # This creates a cartesian product of all allocation options
        def generate_all_combinations(allocation_options_list: List[List[List[int]]]) -> List[List[List[int]]]:
            """Generate all combinations of step allocations across all requests."""
            if not allocation_options_list:
                return [[]]

            result = []
            for allocation in allocation_options_list[0]:
                for rest_combination in generate_all_combinations(allocation_options_list[1:]):
                    result.append([allocation] + rest_combination)
            return result

        all_allocation_combinations = generate_all_combinations(request_allocation_options)
        total_combinations = len(all_allocation_combinations)
        logger.info(f"Total allocation combinations to evaluate: {total_combinations}")

        if total_combinations == 0:
            logger.warning("No valid allocation combinations found")
            return []

        # Step 4: Evaluate each combination and find the one with maximum SLO rate
        best_allocation_combination = None
        best_slo_rate = -1.0
        best_combo_list = []
        best_missed_count = float("inf")
        current_time = get_current_time()

        for combo_idx, allocation_combination in enumerate(all_allocation_combinations):
            # if combo_idx % max(1, total_combinations // 100) == 0:
            #         f"Evaluating combination {combo_idx + 1}/{total_combinations} "
            #         f"({100.0 * (combo_idx + 1) / total_combinations:.1f}%)"
            #     )

            # Check if we can schedule all requests with available GPUs
            # Calculate total GPU capacity needed (conservative estimate)
            total_gpu_capacity_needed = 0
            for req_idx, allocation in enumerate(allocation_combination):
                # Find the maximum GPU count needed for this request
                max_gpus_for_req = 0
                for gpu_log2 in range(4):
                    if allocation[gpu_log2] > 0:
                        max_gpus_for_req = max(max_gpus_for_req, 2**gpu_log2)
                total_gpu_capacity_needed += max_gpus_for_req

            # Skip if total GPU need exceeds capacity (conservative check)
            if total_gpu_capacity_needed > current_world_size:
                continue

            # Calculate execution time and SLO compliance for each request
            req_completion_times = []
            req_meets_slo = []

            for req_idx, req in enumerate(unscheduled_reqs):
                allocation = allocation_combination[req_idx]
                seq_len = req.input_config.height * req.input_config.width
                req_bsz = req.input_config.batch_size

                # Calculate total execution time for this request
                total_execution_time = 0.0
                for gpu_log2 in range(4):
                    if allocation[gpu_log2] > 0:
                        gpu_num = 2**gpu_log2
                        steps = allocation[gpu_log2]
                        step_time = self.get_split_batch_time(
                            bs=req_bsz,
                            seq_len=seq_len,
                            running_steps=steps,
                            gpu_num=gpu_num,
                        )[2]
                        total_execution_time += step_time

                completion_time = current_time + total_execution_time
                req_completion_times.append(completion_time)

                # Check if request meets SLO
                deadline = req.latency_threshold + req.arrival_time
                meets_slo = (
                    completion_time <= deadline
                    or abs(completion_time - deadline) / max(req.latency_threshold, 1.0) <= SLO_TOLERANCE
                )
                req_meets_slo.append(meets_slo)

            # Calculate SLO rate
            slo_rate = sum(req_meets_slo) / len(unscheduled_reqs) if unscheduled_reqs else 0.0
            missed_count = len(unscheduled_reqs) - sum(req_meets_slo)

            # Update best solution
            # Prioritize: higher SLO rate > fewer missed requests
            if (slo_rate > best_slo_rate) or (slo_rate == best_slo_rate and missed_count < best_missed_count):
                best_slo_rate = slo_rate
                best_missed_count = missed_count
                best_allocation_combination = allocation_combination

                # Generate combo list for this allocation combination
                # Convert allocation to combo format: (req_idx, rank_group, steps, time)
                combo_list = []
                rank_idx = 0

                for req_idx, allocation in enumerate(allocation_combination):
                    req = unscheduled_reqs[req_idx]
                    seq_len = req.input_config.height * req.input_config.width
                    req_bsz = req.input_config.batch_size

                    # Process each GPU configuration for this request
                    for gpu_log2 in range(4):
                        if allocation[gpu_log2] > 0:
                            gpu_num = 2**gpu_log2
                            steps = allocation[gpu_log2]

                            # Check if we have enough GPUs available
                            if rank_idx + gpu_num > current_world_size:
                                # Not enough GPUs, skip this configuration
                                continue

                            # Calculate execution time
                            req_time = self.get_split_batch_time(
                                bs=req_bsz,
                                seq_len=seq_len,
                                running_steps=steps,
                                gpu_num=gpu_num,
                            )[2]

                            # Create rank group (using rank indices, not actual ranks)
                            rank_group = list(range(rank_idx, rank_idx + gpu_num))
                            combo_list.append((req_idx, tuple(rank_group), steps, req_time))
                            rank_idx += gpu_num

                            # Stop if we've used all available GPUs
                            if rank_idx >= current_world_size:
                                break

                    # Stop if we've used all available GPUs
                    if rank_idx >= current_world_size:
                        break

                best_combo_list = combo_list

        search_end_time = time.time()
        search_execution_time = search_end_time - search_start_time
        logger.info(
            f"Exhaustive optimal SLO search completed in {search_execution_time:.4f} seconds. "
            f"Best SLO rate: {best_slo_rate:.4f}, Missed requests: {best_missed_count}/{len(unscheduled_reqs)}"
        )

        if best_allocation_combination is None or not best_combo_list:
            logger.warning("No valid solution found in exhaustive search")
            return []

        # Step 5: Build schedule DAG from the optimal solution
        schedule_DAG = self._build_schedule_DAG(best_combo_list, unscheduled_reqs)
        self.schedule_cache.update_schedule_DAG(schedule_DAG)

        # Step 6: Extract ready-to-execute blocks from the DAG
        ret_scheduled_reqs_blocks = []
        free_reqs_blocks = self.schedule_cache.get_new_free_nodes()
        for reqs_block, node_idx in free_reqs_blocks:
            if reqs_block.meet_begin_condition(self.ranks):
                ret_scheduled_reqs_blocks.append(reqs_block)
                self.schedule_cache.remove_node(node_idx)

        return ret_scheduled_reqs_blocks

    def _exhaustive_combo_search(
        self, shape_state, unscheduled_reqs, rank_time, req_time_list, result_missed_list, gpu_time_state
    ):
        """
        Exhaustive search for the best combination of requests to schedule.

        This method implements the original exhaustive enumeration approach that generates
        all possible combinations of requests and GPU allocations, then evaluates each
        combination to find the optimal one based on the specified criteria.

        Note: This approach has exponential time complexity and may be slow for large
        numbers of requests or GPU configurations.

        Args:
            shape_state: Current shape state for each request
            unscheduled_reqs: List of unscheduled requests
            rank_time: Current rank times
            req_time_list: Request time list
            result_missed_list: List of already missed requests
            gpu_time_state: GPU time state for each request

        Returns:
            Tuple of (best_missed, best_missed_list, best_filled, best_gpu_time, best_time, best_rank_used, best_combo_list)
        """
        current_world_size = len(self.ranks)

        def get_possible_combo(shape_state, req_idx, rank_used):
            if req_idx == len(unscheduled_reqs) or rank_used == current_world_size:
                yield ()
            else:
                # Option 1 - not use the req
                for rest in list(get_possible_combo(shape_state, req_idx + 1, rank_used)):
                    yield rest

                # Option 2 - use the req
                for i in range(4):
                    if len(shape_state[req_idx]) < i:
                        continue

                    if shape_state[req_idx][i] == 0:
                        continue

                    if rank_used + 2**i > current_world_size:
                        continue

                    for rest in list(get_possible_combo(shape_state, req_idx + 1, rank_used + 2**i)):
                        yield ((req_idx, i),) + rest

        # Initialize best values
        best_filled = 0
        best_gpu_time = 0
        best_missed = float("inf")
        best_missed_list = []
        best_time = 0
        best_rank_used = 0
        best_combo_list = []

        # Get all possible combinations
        possible_combo_list = list(get_possible_combo(shape_state, 0, 0))

        for combo_list in possible_combo_list:
            new_filled = len(combo_list)
            new_gpu_time = 0
            new_missed = 0
            new_missed_list = []
            new_time = float("inf")
            new_rank_used = 0
            new_combo_list = []

            for req_idx in range(len(unscheduled_reqs)):
                # unscheduled_reqs are finished
                if sum(shape_state[req_idx]) == 0:
                    continue

                seq_len = unscheduled_reqs[req_idx].input_config.height * unscheduled_reqs[req_idx].input_config.width
                req_bsz = unscheduled_reqs[req_idx].input_config.batch_size
                req = unscheduled_reqs[req_idx]

                ideal_req_time = 0
                # unscheduled_reqs that will definitely be missed with the current schedule
                for i in range(4):
                    if shape_state[req_idx][i] == 0:
                        continue

                    ideal_req_time += self.get_split_batch_time(
                        bs=req_bsz, seq_len=seq_len, running_steps=shape_state[req_idx][i], gpu_num=2**i
                    )[2]

                if req_idx in result_missed_list:
                    continue

                # req will be missed in the future
                if (
                    max(min(rank_time), req_time_list[req_idx]) + ideal_req_time
                    > req.latency_threshold + req.arrival_time - get_current_time()
                ) or (req_idx not in [req_idx for req_idx, rank_used_log in combo_list]):
                    new_missed += 1
                    new_missed_list.append(req_idx)

            rank_idx = 0
            for req_idx, rank_used_log in combo_list:
                seq_len = unscheduled_reqs[req_idx].input_config.height * unscheduled_reqs[req_idx].input_config.width
                req_bsz = unscheduled_reqs[req_idx].input_config.batch_size
                rank_used = 2**rank_used_log

                req_time = self.get_split_batch_time(
                    bs=req_bsz,
                    seq_len=seq_len,
                    running_steps=shape_state[req_idx][rank_used_log],
                    gpu_num=rank_used,
                )[2]
                new_time = min(new_time, req_time)
                new_gpu_time += gpu_time_state[req_idx]
                new_combo_list.append(
                    (
                        req_idx,
                        list(range(rank_idx, rank_idx + rank_used)),
                        shape_state[req_idx][rank_used_log],
                        req_time,
                    )
                )
                rank_idx += rank_used

            new_rank_used = rank_idx

            # Update best solution based on optimization criteria
            if (
                new_missed < best_missed
                or (new_missed == best_missed and new_filled > best_filled)
                or (new_missed == best_missed and new_filled == best_filled and new_gpu_time < best_gpu_time)
                or (
                    new_missed == best_missed
                    and new_filled == best_filled
                    and new_gpu_time == best_gpu_time
                    and new_rank_used > best_rank_used
                )
                or (
                    new_missed == best_missed
                    and new_filled == best_filled
                    and new_gpu_time == best_gpu_time
                    and new_rank_used == best_rank_used
                    and new_time > best_time
                )
            ):
                best_missed = new_missed
                best_missed_list = new_missed_list
                best_filled = new_filled
                best_gpu_time = new_gpu_time
                best_time = new_time
                best_rank_used = new_rank_used
                best_combo_list = new_combo_list

        return best_missed, best_missed_list, best_filled, best_gpu_time, best_time, best_rank_used, best_combo_list

    def _dynamic_programming_combo_search(
        self,
        shape_state,
        unscheduled_reqs,
        rank_time,
        req_time_list,
        result_missed_list,
        gpu_time_state,
        round_duration=10,
    ):
        """
        Dynamic programming approach following the DP Round Scheduler algorithm from the paper.

        Args:
            shape_state: Current shape state for each request (equivalent to s_i^m)
            unscheduled_reqs: List of unscheduled requests R
            rank_time: Current rank times
            req_time_list: Request time list
            result_missed_list: List of already missed requests
            gpu_time_state: GPU time state for each request
            round_duration: Round length τ (in seconds)

        Returns:
            Tuple of (best_missed, best_missed_list, best_filled, best_gpu_time, best_time, best_rank_used, best_combo_list)
        """
        current_time = get_current_time()
        round_start_time = current_time
        round_end_time = round_start_time + round_duration
        total_gpu_capacity = len(self.ranks)  # total available GPU capacity

        # Calculate steps completable in this round for each request and GPU configuration
        req_scheduling_options = {}  # Maps request index to list of scheduling options
        for req_idx, unscheduled_req in enumerate(unscheduled_reqs):
            if sum(shape_state[req_idx]) == 0:
                continue
            seq_len = unscheduled_req.input_config.height * unscheduled_req.input_config.width
            req_batch_size = unscheduled_req.input_config.batch_size

            # Generate scheduling options for this request
            scheduling_options = []

            # Option 1: Skip this request (don't schedule)
            # Calculate remaining work lower bound for skipped request
            remaining_work_lower_bound_skip = 0
            for config_idx in range(4):
                if shape_state[req_idx][config_idx] > 0:
                    remaining_work_lower_bound_skip += self.get_split_batch_time(
                        bs=req_batch_size,
                        seq_len=seq_len,
                        running_steps=shape_state[req_idx][config_idx],
                        gpu_num=2**config_idx,
                    )[2]

            # Calculate survival value for skipped request
            request_deadline = unscheduled_req.latency_threshold + unscheduled_req.arrival_time
            survival_value_skip = (
                1
                if (
                    round_end_time + remaining_work_lower_bound_skip <= request_deadline
                    or abs((round_end_time + remaining_work_lower_bound_skip) - request_deadline)
                    / unscheduled_req.latency_threshold
                    <= SLO_TOLERANCE
                )
                else 0
            )

            scheduling_options.append(
                {
                    "option_type": "skip",
                    "gpu_config_idx": None,
                    "steps_to_complete": 0,
                    "gpu_capacity_needed": 0,
                    "survival_value": survival_value_skip,
                    "remaining_work_lower_bound": remaining_work_lower_bound_skip,
                }
            )

            # Option 2: Schedule with different GPU configurations
            for gpu_config_idx in range(4):  # GPU configurations: 1, 2, 4, 8 GPUs
                if shape_state[req_idx][gpu_config_idx] == 0:
                    continue

                remaining_steps_for_config = shape_state[req_idx][gpu_config_idx]  # remaining steps for this GPU config
                gpu_capacity_needed = 2**gpu_config_idx  # number of GPUs needed for this config

                if gpu_capacity_needed > total_gpu_capacity:  # Skip if exceeds available capacity
                    break

                # Calculate steps that can be completed in this round using for loop
                steps_completable_in_round = 0
                cumulative_time = 0.0

                for step in range(1, remaining_steps_for_config + 1):
                    # Get total time for running 'step' steps
                    total_time_for_steps = self.get_split_batch_time(
                        bs=req_batch_size, seq_len=seq_len, running_steps=step, gpu_num=gpu_capacity_needed
                    )[2]

                    if total_time_for_steps <= round_duration:
                        steps_completable_in_round = step
                        cumulative_time = total_time_for_steps
                    else:
                        break
                if steps_completable_in_round <= 0:  # Skip if no steps can be completed
                    continue

                # Calculate remaining steps after this round
                remaining_steps_after_round = {}
                for config_idx in range(4):
                    remaining_steps_after_round[config_idx] = shape_state[req_idx][config_idx]
                remaining_steps_after_round[gpu_config_idx] -= steps_completable_in_round

                # Calculate lower bound for remaining work using get_split_batch_time
                remaining_work_lower_bound = 0
                for config_idx in range(4):
                    if remaining_steps_after_round[config_idx] > 0:
                        remaining_work_lower_bound += self.get_split_batch_time(
                            bs=req_batch_size,
                            seq_len=seq_len,
                            running_steps=remaining_steps_after_round[config_idx],
                            gpu_num=2**config_idx,
                        )[2]

                # Calculate survival value for skipped request
                request_deadline = unscheduled_req.latency_threshold + unscheduled_req.arrival_time
                round_spend_time = self.get_split_batch_time(
                    bs=req_batch_size,
                    seq_len=seq_len,
                    running_steps=steps_completable_in_round,
                    gpu_num=gpu_capacity_needed,
                )[2]

                req_estimate_end_time = round_start_time + round_spend_time + remaining_work_lower_bound
                survival_value = (
                    1
                    if (
                        req_estimate_end_time <= request_deadline
                        or abs(req_estimate_end_time - request_deadline) / unscheduled_req.latency_threshold <= SLO_TOLERANCE
                    )
                    else 0
                )

                # survival_value = 1

                scheduling_options.append(
                    {
                        "option_type": "schedule",
                        "gpu_config_idx": gpu_config_idx,
                        "steps_to_complete": steps_completable_in_round,
                        "gpu_capacity_needed": gpu_capacity_needed,
                        "survival_value": survival_value,
                        "remaining_work_lower_bound": remaining_work_lower_bound,
                        "remaining_steps_after_round": remaining_steps_after_round,
                    }
                )

            req_scheduling_options[req_idx] = scheduling_options


        # Step 2: Dynamic Programming with Multi-Objective Optimization
        # DP state: (survival_value, scheduled_count, gpu_used)
        # Initialize dp_table[0..total_gpu_capacity] ← (-∞, 0, 0), dp_table[0] ← (0, 0, 0)
        dp_table = [(-float("inf"), 0, 0)] * (total_gpu_capacity + 1)
        dp_table[0] = (0, 0, 0)

        # Back-pointer for plan reconstruction
        plan_back_pointers = [([None] * (total_gpu_capacity + 1)) for _ in range(len(req_scheduling_options))]

        # For each request in the scheduling options
        req_cnt = 0
        for req_idx in req_scheduling_options:
            next_dp_table = dp_table.copy()
            if req_cnt > 0:

                plan_back_pointers[req_cnt] = deepcopy(plan_back_pointers[req_cnt - 1])

            # For each GPU capacity from 0 to total_gpu_capacity
            for current_capacity in range(total_gpu_capacity + 1):
                # For each scheduling option for this request
                for scheduling_option in req_scheduling_options[req_idx]:
                    gpu_capacity_needed = scheduling_option["gpu_capacity_needed"]

                    # if current_capacity == 4:

                    if gpu_capacity_needed <= current_capacity:  # If capacity needed ≤ current capacity
                        # Get previous state
                        prev_survival, prev_scheduled, prev_gpu_used = dp_table[current_capacity - gpu_capacity_needed]

                        if prev_survival == -float("inf"):
                            continue

                        # Calculate new state
                        new_survival = prev_survival + scheduling_option["survival_value"]
                        new_scheduled = prev_scheduled + (1 if scheduling_option["option_type"] == "schedule" else 0)
                        new_gpu_used = prev_gpu_used + gpu_capacity_needed

                        # Compare with current best state
                        current_survival, current_scheduled, current_gpu_used = next_dp_table[current_capacity]

                        # Multi-objective comparison: survival_value > scheduled_count > gpu_used
                        if (
                            new_survival > current_survival
                            or (new_survival == current_survival and new_scheduled > current_scheduled)
                            or (
                                new_survival == current_survival
                                and new_scheduled == current_scheduled
                                and new_gpu_used > current_gpu_used
                            )
                        ):
                            next_dp_table[current_capacity] = (new_survival, new_scheduled, new_gpu_used)

                            if scheduling_option["option_type"] == "schedule":
                                plan_back_pointers[req_cnt][current_capacity] = (req_idx, scheduling_option)
                            else:
                                # keep the same schedule as the current one skip
                                if req_cnt > 0:
                                    plan_back_pointers[req_cnt][current_capacity] = deepcopy(
                                        plan_back_pointers[req_cnt - 1][current_capacity]
                                    )

            dp_table = next_dp_table
            req_cnt += 1

        # Step 3: Find optimal GPU capacity allocation using multi-objective comparison
        optimal_capacity = 0
        best_state = dp_table[0]
        for capacity in range(1, total_gpu_capacity + 1):
            current_state = dp_table[capacity]
            # Multi-objective comparison: survival_value > scheduled_count > gpu_used
            if (
                current_state[0] > best_state[0]
                or (current_state[0] == best_state[0] and current_state[1] > best_state[1])
                or (
                    current_state[0] == best_state[0]
                    and current_state[1] == best_state[1]
                    and current_state[2] > best_state[2]
                )
            ):
                optimal_capacity = capacity
                best_state = current_state

        # Step 4: Reconstruct optimal plan from back-pointers
        best_combo_list = []
        best_missed = 0
        best_missed_list = []
        best_filled = 0
        best_gpu_time = 0
        best_time = float("inf")
        best_rank_used = 0

        # Track GPU capacity usage during plan reconstruction
        used_gpu_capacity = 0
        current_capacity = optimal_capacity
        current_req_cnt = len(req_scheduling_options) - 1
        # Reconstruct the optimal plan by following back-pointers
        while current_capacity > 0 and plan_back_pointers[current_req_cnt][current_capacity] is not None:
            req_idx, scheduling_option = plan_back_pointers[current_req_cnt][current_capacity]

            if scheduling_option["option_type"] == "schedule":
                gpu_config_idx = scheduling_option["gpu_config_idx"]
                steps_to_complete = scheduling_option["steps_to_complete"]
                gpu_capacity_needed = scheduling_option["gpu_capacity_needed"]

                # Create scheduling combo entry
                gpu_rank_group = list(range(used_gpu_capacity, used_gpu_capacity + gpu_capacity_needed))
                unscheduled_req = unscheduled_reqs[req_idx]
                seq_len = unscheduled_req.input_config.height * unscheduled_req.input_config.width
                req_batch_size = unscheduled_req.input_config.batch_size

                request_execution_time = self.get_split_batch_time(
                    bs=req_batch_size, seq_len=seq_len, running_steps=steps_to_complete, gpu_num=gpu_capacity_needed
                )[2]

                best_combo_list.append((req_idx, gpu_rank_group, steps_to_complete, request_execution_time))
                best_filled += 1
                best_gpu_time += gpu_time_state[req_idx]
                best_time = min(best_time, request_execution_time)
                best_rank_used += gpu_capacity_needed
                used_gpu_capacity += gpu_capacity_needed

            # Move to previous capacity state
            current_capacity -= scheduling_option["gpu_capacity_needed"]
            current_req_cnt -= 1

        # Calculate missed requests (requests not scheduled that will miss their deadlines)
        scheduled_request_indices = {combo[0] for combo in best_combo_list}
        for req_idx, unscheduled_req in enumerate(unscheduled_reqs):
            if req_idx not in scheduled_request_indices and sum(shape_state[req_idx]) > 0:
                # Check if this request will be missed
                seq_len = unscheduled_req.input_config.height * unscheduled_req.input_config.width
                req_batch_size = unscheduled_req.input_config.batch_size

                ideal_execution_time = 0
                for gpu_config_idx in range(4):
                    if shape_state[req_idx][gpu_config_idx] > 0:
                        ideal_execution_time += self.get_split_batch_time(
                            bs=req_batch_size,
                            seq_len=seq_len,
                            running_steps=shape_state[req_idx][gpu_config_idx],
                            gpu_num=2**gpu_config_idx,
                        )[2]

                will_be_missed = (
                    max(min(rank_time), req_time_list[req_idx]) + ideal_execution_time
                    > unscheduled_req.latency_threshold + unscheduled_req.arrival_time - current_time
                ) or (req_idx in result_missed_list)

                if will_be_missed:
                    best_missed += 1
                    best_missed_list.append(req_idx)


        return best_missed, best_missed_list, best_filled, best_gpu_time, best_time, best_rank_used, best_combo_list

    def _combo_search_with_algorithm_selection(
        self,
        shape_state,
        unscheduled_reqs,
        rank_time,
        req_time_list,
        result_missed_list,
        gpu_time_state,
        use_dp=True,
        round_duration=5.0,
    ):
        """
        Wrapper method to choose between exhaustive search and dynamic programming approaches.

        Args:
            shape_state: Current shape state for each request
            unscheduled_reqs: List of unscheduled requests
            rank_time: Current rank times
            req_time_list: Request time list
            result_missed_list: List of already missed requests
            gpu_time_state: GPU time state for each request
            use_dp: Whether to use dynamic programming (True) or exhaustive search (False)
            round_duration: Maximum time duration for the optimization round (seconds)

        Returns:
            Tuple of (best_missed, best_missed_list, best_filled, best_gpu_time, best_time, best_rank_used, best_combo_list)
        """
        if use_dp:
            return self._dynamic_programming_combo_search(
                shape_state,
                unscheduled_reqs,
                rank_time,
                req_time_list,
                result_missed_list,
                gpu_time_state,
                round_duration,
            )
        else:
            return self._exhaustive_combo_search(
                shape_state, unscheduled_reqs, rank_time, req_time_list, result_missed_list, gpu_time_state
            )
