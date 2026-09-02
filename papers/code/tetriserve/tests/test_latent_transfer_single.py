"""
Test: Latent transfer round-trip on a single rank with the new API.
"""

import os
import socket
import torch
import torch.distributed as dist
import multiprocessing as mp

from tetriserve.light_ray.latent_manager import LatentManager
from tetriserve.light_ray.latent_future import LatentFuture, TensorInfo
from tetriserve.distributed.globals import PROCESS_GROUP
from tetriserve.utils import get_device, set_device


def _free_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def test_single_rank_latent_round_trip():
    if get_device() is None:
        set_device(gpu=False)

    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(_free_port())

    dist.init_process_group(backend="gloo", rank=0, world_size=1)
    if not PROCESS_GROUP.generated:
        PROCESS_GROUP.generate_all_process_groups(1)

    request_queues = [mp.Queue()]
    manager = LatentManager(rank=0, request_queues=request_queues)

    tensor = torch.arange(0, 16, dtype=torch.float32, device=get_device()).reshape(1, 4, 2, 2)
    future = manager.save_latent(sub_req_id=1, tensors={"latents": tensor}, save_tensors=True)
    assert isinstance(future, LatentFuture)

    retrieved = future.get(timeout=5.0)
    assert torch.equal(retrieved, tensor)

    manager.stop_sender_thread()
    dist.destroy_process_group()
