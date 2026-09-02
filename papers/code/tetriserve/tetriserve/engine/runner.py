import os
import time
from typing import Optional, Union, List, Generator, Tuple, Dict

import numpy as np
import PIL.Image

import torch
import torch.distributed as dist
from tetriserve.xfuser.config import EngineConfig, InputConfig
from tetriserve.zmq_utils import Dealer
from tetriserve.config.request import ScheduledRequests
from tetriserve.engine.struct import RunnerOutput
from tetriserve.xfuser.logger import init_logger
from tetriserve.light_ray.latent_manager import LatentManager
from tetriserve.light_ray.latent_future import LatentFuture
from tetriserve.distributed.globals import PROCESS_GROUP
from tetriserve.utils import get_device

from tetriserve.xfuser import xFuserFluxPipeline, xFuserStableDiffusion3Pipeline

import multiprocessing as mp

logger = init_logger(__name__)


def random_rgb_image(width, height):
    """
    Generate a random RGB image for testing purposes.

    Args:
        width (int): Width of the image in pixels
        height (int): Height of the image in pixels

    Returns:
        PIL.Image.Image: Random RGB image
    """
    # create an array of shape (height, width, 3), with random ints in [0,255]
    data = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    # convert to a PIL Image
    img = PIL.Image.fromarray(data, "RGB")
    return img


class FluidServeRunner:
    """
    FluidServe inference runner that executes diffusion model inference on individual workers.

    This class handles the execution of scheduled requests on worker processes, managing:
    - Model loading and inference pipeline setup
    - Distributed communication via ZMQ and multiprocessing queues
    - Latent tensor management and transfer between inference steps
    - Synchronization across multiple GPU ranks for parallel execution

    The runner operates in two modes:
    1. Normal mode: Loads and executes actual diffusion models
    2. Fake mode: Simulates inference with random outputs for testing

    Attributes:
        MODEL_REGISTRY (Dict[str, Type]): Mapping of model names to pipeline classes
        rank (int): GPU rank of this worker
        pipeline: The loaded diffusion pipeline for inference
        latent_manager (LatentManager): Manages latent tensor storage and transfer
        zmq_dealer (Dealer): ZMQ dealer for communication with scheduler
        request_queues (List[mp.Queue]): Queues for latent tensor transfer
        sync_queues (List[mp.Queue]): Queues for synchronization between ranks
    """

    MODEL_REGISTRY = {
        "black-forest-labs/FLUX.1-dev": xFuserFluxPipeline,
        "black-forest-labs/FLUX.1-schnell": xFuserFluxPipeline,
        "stabilityai/stable-diffusion-3-medium-diffusers": xFuserStableDiffusion3Pipeline,
    }

    def __init__(
        self,
        pretrained_model_name_or_path: str,
        engine_config: EngineConfig,
        zmq_router_port: int = None,
        fake_engine: bool = False,
        request_queues: List[mp.Queue] = None,  # Updated parameter
        sync_queues: List[mp.Queue] = None,  # Add sync queues parameter
        **kwargs,
    ):
        """
        Initialize the FluidServeRunner.

        Args:
            pretrained_model_name_or_path (str): Path or name of the pretrained model
            engine_config (EngineConfig): Configuration for the inference engine
            zmq_router_port (int, optional): Port for ZMQ router communication
            fake_engine (bool): If True, runs in simulation mode without loading actual models
            request_queues (List[mp.Queue], optional): Queues for latent tensor transfer between ranks
            sync_queues (List[mp.Queue], optional): Queues for synchronization between ranks
            **kwargs: Additional arguments passed to pipeline initialization
        """

        local_rank = int(os.environ["LOCAL_RANK"])
        self.rank = local_rank
        self.fake_engine = fake_engine
        self.skip_latent_transfer = kwargs.pop("skip_latent_transfer", False)
        self.request_queues = request_queues  # Store the queues
        self.sync_queues = sync_queues  # Store the sync queues (created at engine level)

        if fake_engine:
            self.pipeline = None
        else:
            self.pipeline = self._load_pipeline(pretrained_model_name_or_path, engine_config, **kwargs)

            if torch.cuda.is_available():
                # Ensure pipeline is initialized before moving to device if not None
                if hasattr(self, "pipeline") and self.pipeline is not None:
                    self.pipeline = self.pipeline.to(f"cuda:{self.rank}")  # Use specific rank
                else:
                    # Handle case where pipeline might not be initialized (e.g., placeholder)
                    logger.warning("Pipeline attribute exists but is None, cannot move to CUDA.")

        self.latent_manager = None
        if self.request_queues is not None:
            self.latent_manager = LatentManager(rank=self.rank, request_queues=self.request_queues)

        self.zmq_dealer: Optional[Dealer] = None
        if zmq_router_port is not None:
            self.zmq_dealer = Dealer(
                router_address="localhost",
                router_port=zmq_router_port,
                worker_id=f"Worker-{self.rank}",
            )

    def _load_pipeline(self, pretrained_model_name_or_path: str, engine_config: EngineConfig, **kwargs):
        """
        Load the diffusion model pipeline from the model registry.

        Args:
            pretrained_model_name_or_path (str): Path or name of the pretrained model
            engine_config (EngineConfig): Configuration for the inference engine
            **kwargs: Additional arguments passed to pipeline initialization

        Returns:
            Pipeline: The loaded diffusion pipeline

        Raises:
            ValueError: If the model is not supported in MODEL_REGISTRY
        """
        if pretrained_model_name_or_path not in self.MODEL_REGISTRY:
            raise ValueError(f"Model {pretrained_model_name_or_path} not supported")
        pipeline_class = self.MODEL_REGISTRY[pretrained_model_name_or_path]
        return pipeline_class.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            engine_config=engine_config,
            torch_dtype=torch.bfloat16,
            **kwargs,
        )

    def serve(self) -> None:
        """
        Main serving loop that processes requests from the scheduler.

        This method continuously listens for requests via ZMQ and processes them:
        1. Receives ScheduledRequests from the scheduler via ZMQ
        2. Validates request parameters and rank assignments
        3. Executes inference (real or fake mode)
        4. Sends results back to the scheduler

        Request types handled:
        - "generate": Execute inference on a ScheduledRequests object
        - "done": Signal to stop the serving loop
        - Other types: Return error response

        The method runs until it receives a "done" signal or encounters an error.
        """
        if self.zmq_dealer is None:
            raise RuntimeError("zmq_dealer is not initialized")
        if self.request_queues is None:
            raise RuntimeError("request_queues must be initialized for serve loop")
        try:
            while True:
                request = self.zmq_dealer.recv_from_router()

                if request.get("type") == "generate":
                    reqs: ScheduledRequests = request.get("Requests")
                    logger.info(
                        f"Runner rank {self.rank} received request {reqs.req_ids}, sub_req_ids: {reqs.sub_req_ids}, height: {reqs.input_config.height}, width: {reqs.input_config.width}, parallel_ranks: {reqs.parallel_ranks}"
                    )
                    if len(reqs.parallel_ranks) < 1:
                        raise ValueError("parallel_ranks must not be empty")

                    if not (len(reqs.is_last_steps) == len(reqs.sub_req_ids) == reqs.input_config.batch_size):
                        raise ValueError(
                            f"is_last_steps: {len(reqs.is_last_steps)}, sub_req_ids: {len(reqs.sub_req_ids)}, batch_size: {reqs.input_config.batch_size}"
                        )

                    latent_future_dict = request.get("latent_future_dict")

                    # Check if the current rank is part of the processing ranks *after* potential communication
                    if self.rank not in reqs.parallel_ranks:
                        raise RuntimeError(
                            f"Rank {self.rank} received request {reqs.req_ids} intended for ranks {reqs.parallel_ranks}. ZMQ routing error."
                        )

                    comm_src_rank = reqs.parallel_ranks[0]
                    if self.fake_engine:
                        time.sleep(reqs.estimated_time)
                        output = []
                        for is_last_step, sub_req_id in zip(reqs.is_last_steps, reqs.sub_req_ids):
                            if is_last_step:
                                output.append(random_rgb_image(reqs.input_config.width, reqs.input_config.height))
                            else:
                                # NOTE: this shape is for stable diffusion 3, just for simulation and testing
                                bs = 1
                                height = reqs.input_config.height
                                width = reqs.input_config.width
                                tensors = {
                                    "latents": torch.randn(
                                        (bs, 4, height // 8, width // 8), dtype=torch.float16, device=get_device()
                                    ),
                                    "prompt_embeds": torch.randn(
                                        (bs * 2, 333, 4096), dtype=torch.float16, device=get_device()
                                    ),
                                    "pooled_prompt_embeds": torch.randn(
                                        (bs * 2, 2048), dtype=torch.float16, device=get_device()
                                    ),
                                    "timesteps": torch.rand(
                                        (50,), dtype=torch.float32, device=get_device()
                                    ),  # Shape is for SD3 fake engine simulation only
                                    "sigmas": torch.rand((51,), dtype=torch.float32, device=get_device()),
                                }
                                output.append(self.latent_manager.save_latent(sub_req_id=sub_req_id, tensors=tensors))
                    else:
                        # NOTE: self.generate saves the intermediate tensors into self.latent_manager
                        # NOTE: should return a list of LatentFuture or PIL.Image.Image
                        output = self.generate(reqs, latent_future_dict=latent_future_dict)

                    if self.rank == comm_src_rank:
                        processed_output = []
                        for item in output:
                            if isinstance(item, (LatentFuture, PIL.Image.Image)) or item is None:
                                processed_output.append(item)
                            else:
                                raise TypeError(f"Rank {self.rank} found unexpected output type: {type(item)}")

                        self.zmq_dealer.send_to_router(
                            {
                                "status": "success",
                                "output": RunnerOutput(
                                    sub_req_ids=reqs.sub_req_ids,
                                    images=processed_output,
                                    run_steps=reqs.input_config.num_inference_steps,
                                    parallel_ranks=reqs.parallel_ranks,
                                    bs=reqs.input_config.batch_size,
                                    latent_length=reqs.latent_length,
                                ),
                            }
                        )

                elif request.get("type") == "done":
                    logger.info(f"rank {self.rank} finished")
                    break
                else:
                    reqs = request.get("Requests")
                    req_ids_for_error = reqs.req_ids if hasattr(reqs, "req_ids") else "Unknown"
                    sub_req_ids_for_error = reqs.sub_req_ids if hasattr(reqs, "sub_req_ids") else []
                    parallel_ranks_for_error = reqs.parallel_ranks if hasattr(reqs, "parallel_ranks") else [self.rank]
                    self.zmq_dealer.send_to_router(
                        {
                            "status": "error",
                            "message": f"Invalid request type: {request.get('type')}",
                            "req_ids": req_ids_for_error,
                            "output": RunnerOutput(  # Send back minimal RunnerOutput on error if expected
                                sub_req_ids=sub_req_ids_for_error,
                                images=[],
                                run_steps=0,
                                parallel_ranks=parallel_ranks_for_error,
                                bs=0,
                                latent_length=0,
                            ),
                        }
                    )
                    logger.error(
                        f"Rank {self.rank} received invalid request type: {request.get('type')} for req_ids: {req_ids_for_error}"
                    )

        finally:
            self.latent_manager.stop_sender_thread()
            logger.info(f"[Rank {self.rank}] FluidServeRunner serve loop finished. Signaled sender thread stop.")

    def _mp_barrier_sync(self, parallel_ranks: List[int], sync_id: str) -> None:
        """
        Synchronize multiple ranks using dedicated multiprocessing queues.

        This method implements a barrier synchronization mechanism that ensures all
        ranks in parallel_ranks reach the same synchronization point before proceeding.
        Uses dedicated sync queues to avoid interference with NCCL operations and
        latent transfer queues.

        Args:
            parallel_ranks (List[int]): List of ranks that need to synchronize
            sync_id (str): Unique identifier for this synchronization point

        The synchronization works as follows:
        1. Each rank sends a signal to all other ranks
        2. Each rank waits to receive signals from all other ranks
        3. Once all signals are received, the barrier is complete

        Note:
            This method uses timeouts to prevent indefinite blocking and will
            log warnings if synchronization is incomplete.
        """
        if self.sync_queues is None:
            logger.error(f"[Rank {self.rank}] CRITICAL: Sync queues not initialized for {sync_id}")
            return

        # Send sync signal to all other ranks using dedicated sync queues
        sent_signals = 0
        for rank in parallel_ranks:
            if rank != self.rank and rank < len(self.sync_queues):
                try:
                    self.sync_queues[rank].put((self.rank, sync_id), timeout=2.0)
                    sent_signals += 1
                except Exception as e:
                    logger.error(f"[Rank {self.rank}] BARRIER_SEND_FAIL: to rank {rank} for {sync_id}: {e}")

        # Wait for sync signals from all other ranks
        expected_signals = len(parallel_ranks) - 1
        received_signals = 0

        max_attempts = expected_signals * 3  # Bound total iterations to avoid infinite loop
        attempts = 0
        while received_signals < expected_signals and attempts < max_attempts:
            attempts += 1
            try:
                sender_rank, received_sync_id = self.sync_queues[self.rank].get(timeout=5.0)

                if received_sync_id == sync_id and sender_rank in parallel_ranks:
                    received_signals += 1
                else:
                    logger.warning(
                        f"[Rank {self.rank}] BARRIER_RECV_INVALID: rank {sender_rank}, sync_id '{received_sync_id}', expected '{sync_id}', parallel_ranks {parallel_ranks}"
                    )

            except Exception as e:
                logger.error(
                    f"[Rank {self.rank}] BARRIER_TIMEOUT: {sync_id} - received {received_signals}/{expected_signals} signals: {e}"
                )
                logger.error(f"[Rank {self.rank}] BARRIER_ABORT: {sync_id} - proceeding without full sync")
                break

        # Drain any stale signals left in the queue to prevent corruption of future barriers
        drained = 0
        while True:
            try:
                self.sync_queues[self.rank].get_nowait()
                drained += 1
            except Exception:
                break
        if drained > 0:
            logger.warning(f"[Rank {self.rank}] BARRIER_DRAIN: drained {drained} stale signals after {sync_id}")

        if received_signals == expected_signals:
            logger.info(f"[Rank {self.rank}] BARRIER_SUCCESS: {sync_id} - all {expected_signals} signals received")
        else:
            logger.error(
                f"[Rank {self.rank}] BARRIER_PARTIAL: {sync_id} - only {received_signals}/{expected_signals} signals received"
            )

    def pre_process_reqs(
        self,
        reqs: ScheduledRequests,
        latent_future_dict: Optional[dict] = None,
        use_real_transfer: bool = True,
    ):
        """
        Preprocess scheduled requests to prepare inputs for inference.

        This method handles the complex process of preparing inference inputs, including:
        1. Collecting latent tensors from previous inference steps via LatentFuture objects
        2. Broadcasting tensors across ranks for distributed inference
        3. Running inference prologue for new requests
        4. Synchronizing ranks to ensure consistent state

        The preprocessing happens in two phases:
        - Phase 1: Collect all latent futures from source ranks
        - Phase 2: Broadcast tensors to all participating ranks

        Args:
            reqs (ScheduledRequests): The scheduled requests to preprocess
            latent_future_dict (Optional[dict]): Dictionary mapping sub_req_id to LatentFuture
                                               objects for continuing multi-step inference

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
                - latents: Concatenated latent tensors for all batch items
                - prompt_embeds: Concatenated prompt embeddings
                - pooled_prompt_embeds: Concatenated pooled prompt embeddings
                - timesteps_list: List of timestep tensors for each batch item
                - sigmas_list: List of sigma tensors for each batch item
        """
        latents_list = []
        prompt_embeds_list = []
        pooled_prompt_embeds_list = []
        timesteps_list = []
        sigmas_list = []

        comm_src_rank = reqs.parallel_ranks[0]
        sorted_ranks = sorted(reqs.parallel_ranks)
        comm_group = PROCESS_GROUP.get_process_group(sorted_ranks)
        has_latent_futures = False

        # Dictionary to store collected tensors from latent futures
        collected_tensors = {}

        t_first_loop_start = time.perf_counter()
        # FIRST LOOP: Collect all latent futures without broadcasting
        for idx, sub_req_id in enumerate(reqs.sub_req_ids):
            if latent_future_dict is not None and latent_future_dict.get(sub_req_id, None) is not None:
                has_latent_futures = True
                latent_future: LatentFuture = latent_future_dict.get(sub_req_id)
                if not isinstance(latent_future, LatentFuture):
                    raise TypeError(f"latent_future for {sub_req_id} must be LatentFuture, got {type(latent_future)}")
                # Only collect latent futures that are relevant to this rank
                if not use_real_transfer:
                    continue
                if latent_future.src_rank == self.rank == comm_src_rank:
                    # Latent is stored locally AND we are comm_src_rank: pop directly from local store.
                    # NOTE: Only pop when self.rank == comm_src_rank. If src_rank == self.rank but
                    # self.rank != comm_src_rank, do NOT pop — the sender thread must send the tensor
                    # to comm_src_rank via NCCL. Popping here would race with the sender thread and
                    # cause it to find the tensor missing, sending status=0 and deadlocking comm_src_rank.
                    try:
                        tensors = latent_future.get()
                    except Exception as e:
                        raise RuntimeError(
                            f"[Rank {self.rank}] Failed to collect local tensors for sub_req_id={sub_req_id}: {e}"
                        ) from e
                    if not isinstance(tensors, dict):
                        raise RuntimeError(f"[Rank {self.rank}] Expected tensors dict for sub_req_id={sub_req_id}")
                    collected_tensors[sub_req_id] = tensors
                elif self.rank == comm_src_rank:
                    try:
                        tensors = latent_future.get()
                    except Exception as e:
                        raise RuntimeError(
                            f"[Rank {self.rank}] Failed to receive tensors for sub_req_id={sub_req_id} from rank {latent_future.src_rank}: {e}"
                        ) from e
                    if not isinstance(tensors, dict):
                        raise RuntimeError(f"[Rank {self.rank}] Expected tensors dict for sub_req_id={sub_req_id}")
                    collected_tensors[sub_req_id] = tensors
                # Other ranks will receive tensors via broadcast in the second loop

        t_barrier_start = time.perf_counter()
        # BARRIER: Ensure all latent transfers are complete before broadcast phase
        if use_real_transfer and has_latent_futures and len(reqs.parallel_ranks) > 1:
            sync_id = f"latent_transfer_complete_{reqs.req_ids[0]}_{reqs.start_step}_{reqs.end_step}"
            self._mp_barrier_sync(reqs.parallel_ranks, sync_id)

        t_second_loop_start = time.perf_counter()
        # SECOND LOOP: Handle broadcast operations
        for idx, sub_req_id in enumerate(reqs.sub_req_ids):
            if latent_future_dict is not None and latent_future_dict.get(sub_req_id, None) is not None:
                latent_future: LatentFuture = latent_future_dict.get(sub_req_id)
                if not isinstance(latent_future, LatentFuture):
                    raise TypeError(f"latent_future for {sub_req_id} must be LatentFuture, got {type(latent_future)}")
                if not use_real_transfer:
                    shape_info = latent_future.tensors_info
                    prompt_embeds = torch.randn(
                        shape_info["prompt_embeds"].shape,
                        dtype=shape_info["prompt_embeds"].dtype,
                        device=get_device(),
                    )
                    pooled_prompt_embeds = torch.randn(
                        shape_info["pooled_prompt_embeds"].shape,
                        dtype=shape_info["pooled_prompt_embeds"].dtype,
                        device=get_device(),
                    )
                    latents = torch.randn(
                        shape_info["latents"].shape, dtype=shape_info["latents"].dtype, device=get_device()
                    )
                    timesteps = torch.randn(
                        shape_info["timesteps"].shape, dtype=shape_info["timesteps"].dtype, device=get_device()
                    )
                    sigmas = torch.randn(
                        shape_info["sigmas"].shape, dtype=shape_info["sigmas"].dtype, device=get_device()
                    )
                elif self.rank == comm_src_rank:
                    # Send tensors to other ranks via NCCL point-to-point (send/recv).
                    # Uses dedicated latent transfer groups to avoid deadlocks with
                    # the denoising loop's NCCL collectives (broadcast deadlocks,
                    # but point-to-point on separate communicators is safe).
                    if sub_req_id not in collected_tensors:
                        raise RuntimeError(
                            f"Rank {self.rank} expected to send sub_req_id={sub_req_id} but tensors not collected"
                        )
                    tensors = collected_tensors[sub_req_id]
                    prompt_embeds = tensors["prompt_embeds"]
                    pooled_prompt_embeds = tensors["pooled_prompt_embeds"]
                    latents = tensors["latents"]
                    timesteps = tensors["timesteps"]
                    sigmas = tensors["sigmas"]

                    for dst_rank in reqs.parallel_ranks:
                        if dst_rank == self.rank:
                            continue
                        p2p_group = PROCESS_GROUP.get_latent_transfer_group(self.rank, dst_rank)
                        for t in [prompt_embeds, pooled_prompt_embeds, latents, timesteps, sigmas]:
                            dist.send(t.contiguous(), dst=dst_rank, group=p2p_group)
                elif use_real_transfer:
                    # Receive tensors from comm_src_rank via NCCL point-to-point.
                    # Use irecv+wait(timeout) instead of blocking recv so that if
                    # comm_src_rank crashed (e.g. CUDA OOM in FIRST LOOP), we fail
                    # fast (60s) rather than waiting 600s for NCCL's watchdog.
                    _SECOND_LOOP_RECV_TIMEOUT = 60.0
                    ti = latent_future.tensors_info
                    dev = f"cuda:{self.rank}"
                    prompt_embeds = torch.empty(ti["prompt_embeds"].shape, dtype=ti["prompt_embeds"].dtype, device=dev)
                    pooled_prompt_embeds = torch.empty(ti["pooled_prompt_embeds"].shape, dtype=ti["pooled_prompt_embeds"].dtype, device=dev)
                    latents = torch.empty(ti["latents"].shape, dtype=ti["latents"].dtype, device=dev)
                    timesteps = torch.empty(ti["timesteps"].shape, dtype=ti["timesteps"].dtype, device=dev)
                    sigmas = torch.empty(ti["sigmas"].shape, dtype=ti["sigmas"].dtype, device=dev)

                    p2p_group = PROCESS_GROUP.get_latent_transfer_group(comm_src_rank, self.rank)
                    for t in [prompt_embeds, pooled_prompt_embeds, latents, timesteps, sigmas]:
                        work = dist.irecv(t, src=comm_src_rank, group=p2p_group)
                        if not work.wait(timeout=_SECOND_LOOP_RECV_TIMEOUT):
                            raise RuntimeError(
                                f"[Rank {self.rank}] SECOND LOOP: timeout waiting for tensor "
                                f"from comm_src_rank={comm_src_rank} (sub_req_id={sub_req_id}). "
                                f"comm_src_rank may have crashed."
                            )

                # Store the tensors for this sub_req_id
                latents_list.append(latents)
                prompt_embeds_list.append(prompt_embeds)
                pooled_prompt_embeds_list.append(pooled_prompt_embeds)
                timesteps_list.append(timesteps)
                sigmas_list.append(sigmas)

                self.pipeline.dummy_inference_prologue()
            else:
                # TODO: support batching operation
                prompt_embeds, pooled_prompt_embeds, latents, timesteps, sigmas = self.pipeline.inference_prologue(
                    prompt=reqs.input_config.prompt[idx],
                    height=reqs.input_config.height,
                    width=reqs.input_config.width,
                    num_inference_steps=reqs.origin_steps[0],
                    generator=torch.Generator(device="cuda").manual_seed(reqs.input_config.seed),
                )

                latents_list.append(latents)
                prompt_embeds_list.append(prompt_embeds)
                pooled_prompt_embeds_list.append(pooled_prompt_embeds)
                timesteps_list.append(timesteps)
                sigmas_list.append(sigmas)

        t_second_loop_end = time.perf_counter()

        if use_real_transfer and has_latent_futures and self.rank == comm_src_rank:
            logger.info(
                f"[Rank {self.rank}] LATENT_TIMING req={reqs.req_ids[0]} "
                f"step={reqs.start_step}-{reqs.end_step} sp={reqs.sp_degree}: "
                f"first_loop={t_barrier_start - t_first_loop_start:.3f}s "
                f"barrier={t_second_loop_start - t_barrier_start:.3f}s "
                f"second_loop={t_second_loop_end - t_second_loop_start:.3f}s "
                f"total={t_second_loop_end - t_first_loop_start:.3f}s"
            )

        latents = torch.cat(latents_list, dim=0)
        prompt_embeds = torch.cat(prompt_embeds_list, dim=0)
        pooled_prompt_embeds = torch.cat(pooled_prompt_embeds_list, dim=0)
        return latents, prompt_embeds, pooled_prompt_embeds, timesteps_list, sigmas_list

    @torch.no_grad()
    @torch.inference_mode()
    def generate(
        self,
        reqs: ScheduledRequests,
        latent_future_dict: Optional[dict] = None,  # Add the new argument
        verbose=True,
        **kwargs,
    ) -> Union[
        Generator[Union[List[PIL.Image.Image], np.ndarray, LatentFuture], None, None],
        Union[List[PIL.Image.Image], np.ndarray, LatentFuture],
    ]:
        """
        Generate images or intermediate latents for the given scheduled requests.

        This is the core inference method that:
        1. Configures sequence parallelism settings for distributed inference
        2. Preprocesses requests to prepare inputs (handles latent transfers)
        3. Runs the diffusion model inference
        4. Returns either final images or intermediate latents for multi-step processing

        The method supports both single-step and multi-step inference:
        - For final steps: Returns PIL images
        - For intermediate steps: Returns LatentFuture objects for later use

        Args:
            reqs (ScheduledRequests): The scheduled requests to process
            latent_future_dict (Optional[dict]): Dictionary mapping sub_req_id to LatentFuture
                                               objects for continuing multi-step inference
            verbose (bool): Whether to enable verbose logging
            **kwargs: Additional arguments passed to the inference pipeline

        Returns:
            Union[List[PIL.Image.Image], List[LatentFuture]]:
                - PIL images for final inference steps
                - LatentFuture objects for intermediate steps

        Note:
            This method configures distributed inference settings and should only be
            called after proper initialization of the parallel process groups.
        """
        input_config: InputConfig = reqs.input_config

        from tetriserve.distributed.globals import set_seq_parallel_pg

        set_seq_parallel_pg(
            sp_ulysses_degree=reqs.attn_ulysses_degree,
            sp_ring_degree=reqs.attn_ring_degree,
            rank=self.rank,
            ranks=reqs.parallel_ranks,
            world_size=reqs.sp_degree,
        )

        from tetriserve.xfuser.core.distributed.parallel_state import get_sp_group

        get_sp_group().update_sp_group(ranks=reqs.parallel_ranks)

        from tetriserve.xfuser.config.config import SequenceParallelConfig

        self.pipeline.engine_config.parallel_config.sp_config = SequenceParallelConfig(
            ulysses_degree=reqs.attn_ulysses_degree,
            ring_degree=reqs.attn_ring_degree,
            dit_parallel_size=reqs.sp_degree,
        )

        t_preprocess_start = time.perf_counter()
        latents, prompt_embeds, pooled_prompt_embeds, timesteps_list, sigmas_list = self.pre_process_reqs(
            reqs, latent_future_dict, use_real_transfer=not self.skip_latent_transfer
        )
        t_preprocess_end = time.perf_counter()

        # Release fragmented PyTorch cache before inference so NCCL all_to_all
        # (seq-parallel attention) can allocate work buffers. H100 with FLUX loaded
        # leaves only ~14MB free — fragmentation causes CUDA OOM mid-denoising.
        # Skip for SP=1 (no NCCL all_to_all during inference) to save ~40ms/round.
        if reqs.sp_degree > 1:
            torch.cuda.empty_cache()

        ret_list = []
        ret = None
        image_latent_ret = None
        t_infer_start = time.perf_counter()
        try:
            start_step = reqs.start_step
            end_step = reqs.end_step
            ret = self.pipeline.inference_denoising(
                height=input_config.height,
                width=input_config.width,
                batch_size=input_config.batch_size,
                num_inference_steps=input_config.num_inference_steps,
                timesteps=timesteps_list[0][start_step:end_step],
                sigmas=sigmas_list[0][start_step : end_step + 1],
                latents=latents,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                is_last_steps=reqs.is_last_steps,
                output_type=kwargs.get("output_type", "pil"),
            )

            image_latent_ret = ret.images
            t_infer_end = time.perf_counter()
            if self.rank == reqs.parallel_ranks[0]:
                has_transfer = latent_future_dict is not None and any(
                    latent_future_dict.get(s) is not None for s in reqs.sub_req_ids
                )
                logger.info(
                    f"[Rank {self.rank}] GENERATE_TIMING req={reqs.req_ids[0]} "
                    f"step={reqs.start_step}-{reqs.end_step} sp={reqs.sp_degree} "
                    f"h={reqs.input_config.height} w={reqs.input_config.width} "
                    f"transfer={'real' if has_transfer and not self.skip_latent_transfer else 'none'}: "
                    f"preprocess={t_preprocess_end - t_preprocess_start:.3f}s "
                    f"empty_cache={t_infer_start - t_preprocess_end:.3f}s "
                    f"inference={t_infer_end - t_infer_start:.3f}s"
                )

            return self._collect_results(
                reqs, image_latent_ret,
                prompt_embeds, pooled_prompt_embeds,
                timesteps_list, sigmas_list,
                output_type=kwargs.get("output_type", "pil"),
            )

        finally:
            self._cleanup_generate_tensors(locals())

    def _collect_results(
        self,
        reqs: ScheduledRequests,
        image_latent_ret,
        prompt_embeds,
        pooled_prompt_embeds,
        timesteps_list,
        sigmas_list,
        output_type: str = "pil",
    ) -> list:
        """Collect inference results — save intermediate latents or return final images."""
        ret_list = []
        for idx, (sub_req_id, is_last_step) in enumerate(zip(reqs.sub_req_ids, reqs.is_last_steps)):
            if not is_last_step:
                lat_future = self.latent_manager.save_latent(
                    sub_req_id,
                    {
                        "prompt_embeds": prompt_embeds.chunk(reqs.input_config.batch_size)[idx],
                        "pooled_prompt_embeds": pooled_prompt_embeds.chunk(reqs.input_config.batch_size)[idx],
                        "latents": image_latent_ret[idx].unsqueeze(0),
                        "timesteps": timesteps_list[idx],
                        "sigmas": sigmas_list[idx],
                    },
                    save_tensors=True,
                    step_range=(reqs.start_step, reqs.end_step),
                )
                ret_list.append(lat_future)
            else:
                if output_type == "pil":
                    assert isinstance(
                        image_latent_ret[idx], PIL.Image.Image
                    ), f"image_latent_ret[idx] is not a PIL.Image.Image: {type(image_latent_ret[idx])}"
                ret_list.append(image_latent_ret[idx])
        return ret_list

    @staticmethod
    def _cleanup_generate_tensors(local_vars: dict):
        """Delete large tensors from generate() to help CUDA caching allocator."""
        for name in ("latents", "prompt_embeds", "pooled_prompt_embeds",
                      "timesteps_list", "sigmas_list", "image_latent_ret", "ret"):
            try:
                del local_vars[name]
            except KeyError:
                pass
        # NOTE: Do NOT call torch.cuda.empty_cache() here unconditionally —
        # it adds ~200ms per request (defrag overhead), which hurts SLO.
        # NCCL OOM is handled by the irecv timeout path instead.
