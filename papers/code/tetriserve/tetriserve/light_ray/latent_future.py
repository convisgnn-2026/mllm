from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Union
import time

import torch
import torch.distributed as dist

from tetriserve.utils import get_device
from tetriserve.xfuser.logger import init_logger
from tetriserve.distributed.globals import PROCESS_GROUP

logger = init_logger(__name__)

_LATENT_MANAGERS = {}
_REQUEST_QUEUES = None


def reset_latent_globals() -> None:
    """Reset module globals. For testing only."""
    global _LATENT_MANAGERS, _REQUEST_QUEUES
    _LATENT_MANAGERS = {}
    _REQUEST_QUEUES = None


@dataclass(frozen=True)
class TensorInfo:
    shape: Tuple[int, ...]
    dtype: torch.dtype
    device: str


def register_latent_manager(rank: int, manager, request_queues: list) -> None:
    global _REQUEST_QUEUES
    _LATENT_MANAGERS[rank] = manager
    if _REQUEST_QUEUES is None:
        _REQUEST_QUEUES = request_queues
    elif _REQUEST_QUEUES is not request_queues:
        logger.warning("LatentFuture received a different request_queues reference.")


def unregister_latent_manager(rank: int) -> None:
    _LATENT_MANAGERS.pop(rank, None)


def get_latent_manager(rank: int) -> Optional[object]:
    """Get the LatentManager registered for a given rank."""
    return _LATENT_MANAGERS.get(rank)


def get_request_queues() -> Optional[list]:
    return _REQUEST_QUEUES


class LatentFuture:
    """Follows ray.future pattern - represents a future tensor result."""

    def __init__(self, sub_req_id: int, src_rank: int, tensors_info, step_range: Optional[Tuple[int, int]] = None):
        """
        Initialize LatentFuture with information about multiple tensors.

        Args:
            sub_req_id: The sub request ID
            src_rank: The source rank that has the tensors
            tensors_info: Dict[str, TensorInfo] or legacy dict format
            step_range: (start_step, end_step) for the tensors
        """
        self.sub_req_id = sub_req_id
        self.src_rank = src_rank
        self.step_range = step_range
        self.tensors_info = self._normalize_tensors_info(tensors_info)

    def _normalize_tensors_info(self, tensors_info: dict) -> Dict[str, TensorInfo]:
        normalized = {}
        for name, info in tensors_info.items():
            if isinstance(info, TensorInfo):
                normalized[name] = info
                continue
            if not isinstance(info, dict):
                raise TypeError(f"Invalid tensors_info entry for {name}: {type(info)}")
            normalized[name] = TensorInfo(
                shape=tuple(info["shape"]),
                dtype=info["dtype"],
                device=info.get("device", str(get_device())),
            )
        return normalized

    def ready(self) -> bool:
        """Non-blocking check if tensors available."""
        if not dist.is_initialized():
            return False
        my_rank = dist.get_rank()
        if self.src_rank != my_rank:
            return False
        latent_manager = get_latent_manager(my_rank)
        if latent_manager is None:
            return False
        return latent_manager.has_local_tensor(self.sub_req_id)

    def get(self, timeout: float = 30.0) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Block until tensors available, with timeout.
        """
        if not dist.is_initialized():
            raise RuntimeError("torch.distributed is not initialized; cannot resolve LatentFuture.")
        my_rank = dist.get_rank()

        if self.src_rank == my_rank:
            latent_manager = get_latent_manager(my_rank)
            if latent_manager is None:
                raise RuntimeError(f"No latent_manager registered for rank {my_rank}")
            start_time = time.time()
            while time.time() - start_time < timeout:
                local_tensors = latent_manager.pop_local_tensor(self.sub_req_id)
                if local_tensors is not None:
                    return local_tensors
                time.sleep(0.001)
            raise TimeoutError(f"[Rank {my_rank}] Timeout waiting for local tensor sub_req_id={self.sub_req_id}")

        request_queues = get_request_queues()
        if request_queues is None:
            raise RuntimeError("LatentFuture request queues not registered.")
        if self.src_rank >= len(request_queues):
            raise IndexError(
                f"src_rank {self.src_rank} out of range for request_queues list (length {len(request_queues)})"
            )

        target_queue = request_queues[self.src_rank]
        target_queue.put((my_rank, self.sub_req_id))

        latent_group = PROCESS_GROUP.get_latent_transfer_group(self.src_rank, my_rank)

        status = torch.empty(1, dtype=torch.int32, device=get_device())
        status_work = dist.irecv(tensor=status, src=self.src_rank, group=latent_group)
        if not status_work.wait(timeout=timeout):
            raise TimeoutError(
                f"[Rank {my_rank}] Timeout waiting for status from {self.src_rank} (sub_req_id={self.sub_req_id})"
            )
        if int(status.item()) != 1:
            raise RuntimeError(f"[Rank {my_rank}] Sender reported missing tensors for sub_req_id={self.sub_req_id}")

        tensors = {}
        for tensor_name, info in self.tensors_info.items():
            tensor = torch.empty(size=info.shape, dtype=info.dtype, device=get_device())
            work = dist.irecv(tensor=tensor, src=self.src_rank, group=latent_group)
            if not work.wait(timeout=timeout):
                raise TimeoutError(
                    f"[Rank {my_rank}] Timeout waiting for {tensor_name} from {self.src_rank} (sub_req_id={self.sub_req_id})"
                )
            tensors[tensor_name] = tensor

        logger.info(f"[Rank {my_rank}] Received tensors from {self.src_rank}")
        if len(tensors) == 1:
            return next(iter(tensors.values()))
        return tensors
