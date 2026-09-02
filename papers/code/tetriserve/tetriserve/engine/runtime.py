import asyncio
from copy import deepcopy
from typing import List, Union, Dict, AsyncGenerator

from tetriserve.light_ray.latent_future import LatentFuture
from tetriserve.zmq_utils import (
    find_free_port,
)
from tetriserve.engine.proc_launcher import launch_worker
from tetriserve.config.request import UnScheduledRequest, ScheduledRequests, ScheduledRequestsBlock

from .struct import EngineOutput, RunnerOutput, ReqState
import multiprocessing as mp
from tetriserve.xfuser.config import InputConfig
from tetriserve.scheduler import DynStepScheduler
from tetriserve.zmq_utils import Router
from tetriserve.config.args import ServerArgs
from tetriserve.utils import get_gen_req_id, get_gen_sub_req_id, reset_start_time
from tetriserve.utils import get_current_time
from tetriserve.xfuser.logger import init_logger
from tetriserve.tracer import FluidServeTracer
from tetriserve import SUPPORTED_MODEL_DICT

logger = init_logger(__name__)


def _set_envs_and_config():
    mp.set_start_method("spawn", force=True)


class _CServeEngine:

    def __init__(
        self,
        server_args: ServerArgs,
    ):
        self.engine_processes = []

        parallel_degree = server_args.nproc_per_node

        self.parallel_degree = parallel_degree
        self.zmq_router_port = find_free_port()

        self.master_addr = server_args.master_addr
        self.dist_master_port = find_free_port()

        _set_envs_and_config()

        # Create a list of multiprocessing Queues instead of Manager.dict
        # One queue for each worker rank to receive latent requests
        self.latent_request_queues = [mp.Queue() for _ in range(parallel_degree)]

        # Create dedicated sync queues for barrier synchronization (separate from latent transfer)
        self.sync_queues = [mp.Queue() for _ in range(parallel_degree)]

        for rank in range(parallel_degree):
            proc = mp.Process(
                target=launch_worker,
                args=(
                    rank,
                    parallel_degree,
                    self.zmq_router_port,
                    self.master_addr,
                    self.dist_master_port,
                    server_args.engine_config,
                    server_args.fake_engine,
                    server_args.skip_latent_transfer,
                    self.latent_request_queues,
                    self.sync_queues,
                ),
            )
            proc.start()
            self.engine_processes.append(proc)

        self.enable_tracer = False
        if server_args.enable_tracer:
            logger.info("Tracer Enabled")
            self.enable_tracer = True
            proc_names = [f"GPU-Worker-{rank}" for rank in range(parallel_degree)]
            proc_names.append("scheduler")
            self.tracer = FluidServeTracer(proc_names)
            self.tracer_lock = asyncio.Lock()

        LOG_DIR = server_args.perf_model_dir

        # get the key from the value of SUPPORTED_MODEL_DICT
        for key, value in SUPPORTED_MODEL_DICT.items():
            if value == server_args.engine_config.model_config.model:
                model_simplified_name = key
                break
        else:
            raise ValueError(f"Model {server_args.engine_config.model_config.model} not supported")

        from tetriserve.utils import get_gpu_name

        self.scheduler = DynStepScheduler(
            ranks=list(range(parallel_degree)),
            schedule_logic=server_args.schedule_logic,
            e2e_scaling_efficiency_path=f"{LOG_DIR}/{model_simplified_name}/{get_gpu_name()}/e2e_scaling_efficiency.json",
        )

        self.scheduler_lock = asyncio.Lock()  # NOTE: this lock couldn't be moved to the scheduler class

        self.zmq_router = Router("localhost", self.zmq_router_port, parallel_degree)

        self.log_requests = server_args.log_requests

        logger.info("Engine init done")

    async def init_zmq_router(self) -> None:
        await self.zmq_router.init()

    async def add_request(self, input_config: InputConfig, request_id: int, latency_threshold: float = None) -> None:
        if isinstance(input_config, InputConfig):
            tasks = [get_gen_sub_req_id() for _ in range(input_config.batch_size)]
            sub_req_ids = await asyncio.gather(*tasks)
            if self.log_requests:
                logger.info(
                    f"Add Request: req_id: {request_id}, sub_req_ids: {sub_req_ids}, batch_size: {input_config.batch_size}"
                )
            arrival_time = get_current_time()
            self.scheduler.add_request(
                request_id=request_id,
                sub_req_ids=sub_req_ids,
                input_config=input_config,
                arrival_time=arrival_time,
                latency_threshold=latency_threshold,
            )
        else:
            raise ValueError("input_config must be a single InputConfig")

    async def handle_single_bunch_requests(
        self, sched_reqs: ScheduledRequests, trace_enable: bool = True
    ) -> RunnerOutput:

        latent_future_dict = {}
        for sub_req_id in sched_reqs.sub_req_ids:
            latent_future = self.scheduler.request_tracker.sub_req_id_to_latent_images.get(sub_req_id, None)
            assert (
                isinstance(latent_future, LatentFuture) or latent_future is None
            ), f"latent_future must be a LatentFuture or None"
            latent_future_dict[sub_req_id] = latent_future

        input_packet = {
            "type": "generate",
            "Requests": sched_reqs,
            "latent_future_dict": latent_future_dict,  # {sub_req_id: LatentFuture}
        }

        if self.enable_tracer and trace_enable:
            async with self.tracer_lock:

                req_ids = sched_reqs.req_ids
                sub_req_ids = sched_reqs.sub_req_ids
                steps = sched_reqs.input_config.num_inference_steps
                bs = sched_reqs.input_config.batch_size
                lat_len = sched_reqs.latent_length
                est_time = sched_reqs.estimated_time

                for rank in sched_reqs.parallel_ranks:
                    self.tracer.add_event(
                        f"ReqId:{req_ids}, SubId:{sub_req_ids}, Step:{steps}, BS:{bs}, Lat-Len:{lat_len}",
                        f"0",  # different cats may have different colors
                        f"GPU-Worker-{rank}",
                        get_current_time(),
                        start=True,
                        estimated_time=est_time,
                        parallel_degree=sched_reqs.sp_degree,
                    )

        if self.log_requests:
            for i in sched_reqs.parallel_ranks:
                logger.info(
                    f"Router send req ids: {sched_reqs.req_ids}, sub_req_ids: {sched_reqs.sub_req_ids} to worker-{i} at {get_current_time()}, steps: {sched_reqs.input_config.num_inference_steps}, is_last_steps: {sched_reqs.is_last_steps}"
                )

        await self.zmq_router.send_to_workers([f"Worker-{i}" for i in sched_reqs.parallel_ranks], input_packet)

        worker_id, output = await self.zmq_router.recv_from_worker()

        if self.log_requests:
            logger.info(
                f"Router received {output.get('output').sub_req_ids} from worker-{worker_id} at {get_current_time()}"
            )

        if output.get("status") == "success":
            runner_output = output.get("output")
            async with self.scheduler_lock:
                self.scheduler.add_ranks(runner_output.parallel_ranks)
        else:
            runner_output = output.get("output")
            async with self.scheduler_lock:
                self.scheduler.add_ranks(runner_output.parallel_ranks)
            raise Exception(f"Worker-{worker_id} failed to generate images")

        return runner_output

    async def schedule_tasks(self):
        """
        Only in this function, we will invoke the scheduler.schedule() function.
        """

        pending = []

        async with self.scheduler_lock:
            if self.enable_tracer:
                async with self.tracer_lock:
                    self.tracer.add_event(
                        f"schedule_idx: {self.scheduler.schedule_idx}",
                        "schedule",
                        "scheduler",
                        get_current_time(),
                        start=True,
                    )

            if self.log_requests:
                logger.info(
                    f"schedule_idx: {self.scheduler.schedule_idx}, running_requests: {self.scheduler.running_requests}"
                )

            scheduled_blocks: List[ScheduledRequestsBlock] = self.scheduler.schedule()

            if len(scheduled_blocks) == 0:
                return pending, scheduled_blocks
            else:
                for selected_req_block in scheduled_blocks:

                    assert selected_req_block.meet_begin_condition(
                        self.scheduler.ranks
                    ), "the selected block does not meet the begin condition"
                    self.scheduler.remove_ranks(selected_req_block.begin_condition_ranks)

                    pending += [
                        asyncio.create_task(self.handle_single_bunch_requests(reqs))
                        for reqs in selected_req_block.requests
                    ]

                    for sched_reqs in selected_req_block.requests:
                        if self.log_requests:
                            logger.info(
                                f"running_requests add req_ids: {sched_reqs.req_ids}, sub_req_ids: {sched_reqs.sub_req_ids}"
                            )
                        self.scheduler.running_requests.extend(sched_reqs.sub_req_ids)

            if self.enable_tracer:
                async with self.tracer_lock:
                    self.tracer.add_event(
                        f"schedule_idx: {self.scheduler.schedule_idx}",
                        "schedule",
                        "scheduler",
                        get_current_time(),
                        start=False,
                    )

        return pending, scheduled_blocks

    async def step(self) -> AsyncGenerator[EngineOutput, None]:

        pending_tasks, _ = await self.schedule_tasks()

        if not pending_tasks:
            return

        new_task = None

        while pending_tasks:
            done, pending_tasks = await asyncio.wait(pending_tasks, return_when=asyncio.FIRST_COMPLETED)
            pending_tasks = list(pending_tasks)

            for task in done:
                try:
                    result: RunnerOutput = task.result()

                    if self.enable_tracer:
                        async with self.tracer_lock:
                            assert isinstance(
                                result, RunnerOutput
                            ), f"result is not a RunnerOutput instance: {type(result)}"
                            sub_req_ids = result.sub_req_ids
                            steps = result.run_steps
                            bs = result.bs
                            lat_len = result.latent_length
                            req_ids = [self.scheduler.request_tracker[sub_req_id].req_id for sub_req_id in sub_req_ids]

                            for rank in result.parallel_ranks:
                                self.tracer.add_event(
                                    f"ReqId:{req_ids}, SubId:{sub_req_ids}, Step:{steps}, BS:{bs}, Lat-Len:{lat_len}",
                                    f"0",
                                    f"GPU-Worker-{rank}",
                                    get_current_time(),
                                    start=False,
                                )

                    async with self.scheduler_lock:
                        for sub_req_id in result.sub_req_ids:
                            self.scheduler.running_requests.remove(sub_req_id)
                        ret_images_list = self.scheduler.request_tracker.update_requests(result)

                        if self.log_requests:
                            logger.info(f"RequestTracker: {self.scheduler.request_tracker}")

                    for req_id, images in ret_images_list.items():
                        yield EngineOutput(
                            req_id=req_id,
                            image=images,
                        )

                    # if there are still requests that not done, the cache will be not empty
                    is_scheduled = False
                    async with self.scheduler_lock:
                        if (
                            self.scheduler.is_new_request_coming
                        ):  # new request comes, they will be scheduled by invoking self.step(), but maybe they won't be scheduled immediately, so here we need to schedule them
                            is_scheduled = True
                        if (
                            self.scheduler.remaining_requests() > 0
                        ):  # no new request comes, but there are still requests that are not done
                            is_scheduled = True

                        if self.scheduler.remaining_requests() == 0:
                            is_scheduled = False

                    if is_scheduled:
                        new_task, _ = await self.schedule_tasks()
                        if new_task is not None:
                            pending_tasks.extend(new_task)

                except Exception as e:
                    logger.error(f"Task failed with error: {str(e)}")
                    import traceback

                    logger.error(traceback.format_exc())
                    continue

    async def engine_step(self) -> None:
        raise NotImplementedError("please implement this method in the subclass")

    def export_trace(self, trace_file: str = "trace.json") -> None:
        assert self.enable_tracer, "Tracer is not enabled"
        self.tracer.export(trace_file)

    def export_trace_readable(self, trace_readable_file: str = "trace_readable.json") -> None:
        assert self.enable_tracer, "Tracer is not enabled"
        self.tracer.export_readable_json(trace_readable_file)

    def shutdown(self) -> None:
        """Terminate all worker processes."""
        for process in self.engine_processes:
            process.terminate()


class CServeRuntime:
    """
    Online Async Server version of `CServeEngine`
    """

    def __init__(self, server_args: ServerArgs):
        self.engine = _CServeEngine(server_args)
        self.req_states: Dict[int, ReqState] = {}  # request id -> request state
        self.server_args = server_args

    async def init_zmq_router(self):
        await self.engine.init_zmq_router()

    def _create_warmup_request(self, ranks, batch_size, height, width, steps, req_id):
        """Create a ScheduledRequests for warmup."""
        sp_degree = len(ranks)
        sub_req_ids = [-(req_id * 10 + i) for i in range(batch_size)]
        input_config = InputConfig(
            batch_size=batch_size,
            height=height,
            width=width,
            prompt=["warmup"] * batch_size,
            num_inference_steps=steps,
            output_type="latent",
            seed=42,
        )
        return ScheduledRequests(
            req_ids=[req_id],
            sub_req_ids=sub_req_ids,
            parallel_ranks=list(ranks),
            attn_ulysses_degree=sp_degree,
            attn_ring_degree=1,
            sp_degree=sp_degree,
            latent_length=None,
            input_config=input_config,
            origin_steps=[steps] * batch_size,
            start_step=0,
            end_step=steps,
            is_last_steps=[True] * batch_size,
            arrival_time=get_current_time(),
            latency_threshold=float("inf"),
            estimated_time=0.1,
        )

    async def _run_warmup_group(self, group, batch_size, height, width, steps, req_id_base, label):
        """Run a group of non-overlapping GPU combinations concurrently."""
        tasks = []
        for idx, combo in enumerate(group):
            req = self._create_warmup_request(combo, batch_size, height, width, steps, req_id_base + idx)
            tasks.append(asyncio.create_task(
                self.engine.handle_single_bunch_requests(req, trace_enable=False)
            ))
        if tasks:
            try:
                await asyncio.gather(*tasks)
            except Exception as e:
                logger.warning(f"Some {label} warmup requests failed: {e}")
            await asyncio.sleep(1.0)

    async def warmup(self, if_continuous: bool = True):
        """
        Warm up GPU workers by testing all GPU combinations.

        Tests all C(n,2) combinations, continuous/semi-continuous 4-GPU patterns,
        and C(n,8) combinations. Non-overlapping combinations run in parallel.

        Args:
            if_continuous: If True, use continuous 4-GPU windows; if False, use all C(n,4).
        """
        if self.server_args.fake_engine:
            logger.info("Warmup skipped for fake engine")
            return

        import itertools

        parallel_degree = self.engine.parallel_degree
        batch_size, height, width, steps = 2, 1024, 1024, 10

        logger.info(f"Starting warmup on {parallel_degree} GPUs (continuous_4={if_continuous})")

        def find_non_overlapping_groups(combinations, max_group_size):
            """Find non-overlapping groups from a list of GPU combinations."""
            groups = []
            remaining = list(combinations)
            while remaining:
                current_group = []
                used_gpus = set()
                for combo in remaining[:]:
                    if not any(gpu in used_gpus for gpu in combo):
                        current_group.append(combo)
                        used_gpus.update(combo)
                        remaining.remove(combo)
                        if len(current_group) >= max_group_size:
                            break
                if current_group:
                    groups.append(current_group)
            return groups

        warmup_start_time = get_current_time()
        total_combos = 0
        req_id_counter = 200000

        # Test 2-GPU combinations
        if parallel_degree >= 2:
            combos = list(itertools.combinations(range(parallel_degree), 2))
            groups = find_non_overlapping_groups(combos, parallel_degree // 2)
            logger.info(f"Testing {len(combos)} 2-GPU combinations in {len(groups)} groups")
            for group in groups:
                await self._run_warmup_group(group, batch_size, height, width, steps, req_id_counter, "2-GPU")
                req_id_counter += len(group)
            total_combos += len(combos)

        # Test 4-GPU combinations
        if parallel_degree >= 4:
            if if_continuous:
                # Continuous windows: [0,1,2,3], [1,2,3,4], ...
                combos_4 = [tuple(range(s, s + 4)) for s in range(parallel_degree - 3)]
                # Semi-continuous: one isolated GPU + 3 continuous
                for skip in range(parallel_degree - 3):
                    for start in range(skip + 1, parallel_degree - 2):
                        if start + 2 < parallel_degree:
                            combo = tuple(sorted([skip] + list(range(start, start + 3))))
                            if len(set(combo)) == 4:
                                combos_4.append(combo)
            else:
                combos_4 = list(itertools.combinations(range(parallel_degree), 4))[0:-1:2]

            groups = find_non_overlapping_groups(combos_4, parallel_degree // 4)
            logger.info(f"Testing {len(combos_4)} 4-GPU combinations in {len(groups)} groups")
            for group in groups:
                await self._run_warmup_group(group, batch_size, height, width, steps, req_id_counter, "4-GPU")
                req_id_counter += len(group)
            total_combos += len(combos_4)

        # Test 8-GPU combinations
        if parallel_degree >= 8:
            combos = list(itertools.combinations(range(parallel_degree), 8))
            groups = find_non_overlapping_groups(combos, max(1, parallel_degree // 8))
            logger.info(f"Testing {len(combos)} 8-GPU combinations in {len(groups)} groups")
            for group in groups:
                await self._run_warmup_group(group, batch_size, height, width, steps, req_id_counter, "8-GPU")
                req_id_counter += len(group)
            total_combos += len(combos)

        duration = get_current_time() - warmup_start_time
        logger.info(f"Warmup completed: {total_combos} combinations in {duration:.2f}s on {parallel_degree} GPUs")

    # Keep old name as alias for backward compatibility
    async def warmup_test_all_combinations(self, if_continous: bool = True):
        await self.warmup(if_continuous=if_continous)

    async def engine_step(self):
        try:
            async for output in self.engine.step():
                if output is None:
                    continue

                request_id = output.req_id
                assert request_id in self.req_states, f"Request {request_id} not found in req_states"

                self.req_states[request_id].output = output
                self.req_states[request_id].success = True
                self.req_states[request_id].event.set()

        except Exception as e:
            logger.error(f"engine_step crashed: {e}")
            import traceback

            logger.error(traceback.format_exc())
            await asyncio.sleep(0.1)

    async def generate(
        self, input_config: InputConfig, request_id: int = None, latency_threshold: float = float("inf")
    ):

        if request_id is None:
            request_id = await get_gen_req_id()

        request_event = asyncio.Event()
        self.req_states[request_id] = ReqState(request_event, None, False)

        if self.engine.log_requests:
            logger.info(
                f"Engine received request {request_id}, bs: {input_config.batch_size}, steps: {input_config.num_inference_steps}, height: {input_config.height}, width: {input_config.width}"
            )

        async with self.engine.scheduler_lock:
            await self.engine.add_request(
                input_config=input_config, request_id=request_id, latency_threshold=latency_threshold
            )

        asyncio.create_task(self.engine_step())

        try:
            await request_event.wait()

            if request_id not in self.req_states:
                raise RuntimeError(f"Event set for {request_id}, but state missing.")

            req_state = self.req_states[request_id]

            if self.engine.log_requests:
                logger.info(f"Engine finished request {request_id}")

            request_output = req_state.output
            del self.req_states[request_id]

            if not req_state.success:
                raise Exception(f"Request {request_id} failed during processing.")

            return request_output
        except Exception as e:
            if request_id in self.req_states:
                del self.req_states[request_id]
            logger.error(f"Error awaiting request {request_id}: {e}")
            raise

    async def _send_done_packet(self):
        done_packet = {
            "type": "done",
        }
        async with self.engine.scheduler_lock:
            ranks_to_notify = deepcopy(self.engine.scheduler.ranks)
        await self.engine.zmq_router.send_to_workers([f"Worker-{i}" for i in ranks_to_notify], done_packet)

    def export_trace(self, trace_file: str = "trace.json"):
        self.engine.export_trace(trace_file)

    def export_trace_readable(self, trace_readable_file: str = "trace_readable.json"):
        self.engine.export_trace_readable(trace_readable_file)

    def reset_tracer(self):
        self.engine.tracer.reset()
        reset_start_time()
