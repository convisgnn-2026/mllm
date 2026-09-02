"""
Test: Latent transfer between two ranks with the new API.
"""

import os
import socket
import torch
import torch.distributed as dist
import multiprocessing as py_mp
import torch.multiprocessing as mp

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


def _worker(rank, world_size, master_port, request_queues, result_queue):
    if get_device() is None:
        set_device(gpu=False)

    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)

    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    if not PROCESS_GROUP.generated:
        PROCESS_GROUP.generate_all_process_groups(world_size)

    manager = LatentManager(rank=rank, request_queues=request_queues)
    tensor = torch.arange(0, 16, dtype=torch.float32, device=get_device()).reshape(1, 4, 2, 2)

    if rank == 0:
        manager.save_latent(sub_req_id=7, tensors={"latents": tensor}, save_tensors=True)
        dist.barrier()
    else:
        info = {"latents": TensorInfo(shape=tuple(tensor.shape), dtype=tensor.dtype, device=str(tensor.device))}
        future = LatentFuture(sub_req_id=7, src_rank=0, tensors_info=info)
        received = future.get(timeout=10.0)
        result_queue.put(torch.equal(received, tensor))
        dist.barrier()

    manager.stop_sender_thread()
    dist.destroy_process_group()


def test_multi_rank_latent_transfer():
    world_size = 2
    master_port = _free_port()
    ctx = py_mp.get_context("spawn")
    request_queues = [ctx.Queue() for _ in range(world_size)]
    result_queue = ctx.Queue()

    mp.spawn(
        _worker,
        args=(world_size, master_port, request_queues, result_queue),
        nprocs=world_size,
        join=True,
    )

    assert result_queue.get(timeout=5.0) is True
