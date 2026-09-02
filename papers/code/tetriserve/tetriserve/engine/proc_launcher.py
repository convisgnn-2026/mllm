import os
from tetriserve.xfuser.config import EngineConfig
from tetriserve.engine.runner import FluidServeRunner
from tetriserve.distributed.distributed import init_distributed_environment
from tetriserve.utils import set_device
import multiprocessing as mp
from typing import List

from tetriserve.xfuser.logger import init_logger

logger = init_logger(__name__)


def launch_worker(
    rank,
    parallel_degree,
    zmq_router_port,
    master_addr,
    master_port,
    engine_config: EngineConfig,
    fake_engine: bool,
    skip_latent_transfer: bool,
    request_queues: List[mp.Queue],
    sync_queues: List[mp.Queue],
):
    # support single node only now
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(parallel_degree)
    os.environ["OMP_NUM_THREADS"] = "1"

    # Add explicit CUDA device assignment
    if not fake_engine:
        import torch

        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            # Ensure rank maps to available device
            local_device_id = rank % device_count

            # Log before setting device
            current_device_before = torch.cuda.current_device() if torch.cuda.device_count() > 0 else -1
            logger.info(
                f"[Rank {rank}] Before set_device: current_device={current_device_before}, will set to device {local_device_id}"
            )

            torch.cuda.set_device(local_device_id)

            # Verify device was set correctly
            current_device_after = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device_after)
            device_uuid = (
                torch.cuda.get_device_properties(current_device_after).uuid
                if hasattr(torch.cuda.get_device_properties(current_device_after), "uuid")
                else "N/A"
            )
            logger.info(
                f"[Rank {rank}] After set_device: current_device={current_device_after}, device_name='{device_name}', device_count={device_count}"
            )

    dist_init_method = f"tcp://{master_addr}:{master_port}"

    backend = "nccl" if not fake_engine else "gloo"
    logger.info(f"[Rank {rank}] Initializing distributed environment with backend: {backend}")
    set_device(gpu=not fake_engine)

    # Set skip_latent_transfer env var for distributed init
    if skip_latent_transfer:
        os.environ["TETRISERVE_SKIP_LATENT_TRANSFER"] = "1"

    try:
        if not fake_engine:
            init_distributed_environment(
                local_rank=rank,
                rank=rank,
                world_size=parallel_degree,
                distributed_init_method=dist_init_method,
                backend=backend,
            )

    except Exception as e:
        logger.error(f"[Rank {rank}] Failed to initialize distributed environment: {e}")
        raise e

    runner = FluidServeRunner(
        pretrained_model_name_or_path=engine_config.model_config.model,
        engine_config=engine_config,
        zmq_router_port=zmq_router_port,
        fake_engine=fake_engine,
        request_queues=request_queues,
        sync_queues=sync_queues,
        skip_latent_transfer=skip_latent_transfer,
    )

    logger.info(f"[Rank {rank}] Starting runner serve loop.")
    runner.serve()
    logger.info(f"[Rank {rank}] Runner serve loop finished.")
