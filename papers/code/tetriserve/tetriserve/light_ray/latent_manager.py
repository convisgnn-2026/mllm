import queue
import torch
import torch.distributed as dist
from typing import Union, Dict

import time
import threading

from tetriserve.utils import get_device
from tetriserve.xfuser.logger import init_logger

# Import the process group singleton
from tetriserve.distributed.globals import PROCESS_GROUP
from tetriserve.light_ray.latent_future import (
    LatentFuture,
    TensorInfo,
    register_latent_manager,
    unregister_latent_manager,
)

logger = init_logger(__name__)

# Use None as the sentinel object for stopping the sender thread
_STOP_SIGNAL = None

DEFAULT_LATENT_TTL_SECONDS = 300.0


class LatentManager:
    def __init__(self, rank: int, request_queues: list, auto_start: bool = True):
        """Initializes LatentManager with its rank and the list of request queues.

        Args:
            rank: GPU rank for this manager.
            request_queues: List of multiprocessing queues (one per rank).
            auto_start: If True (default), starts the background sender thread
                immediately. Set to False for unit testing without threading.
        """
        self.rank = rank
        self.latent_store = {}  # {sub_req_id: {"tensors": {name: tensor}, "saved_at": float}}
        self.latent_ttl_seconds = DEFAULT_LATENT_TTL_SECONDS
        self._last_cleanup_ts = time.time()
        if rank >= len(request_queues):
            raise IndexError(f"rank {rank} out of range for request_queues list (length {len(request_queues)})")
        self.request_queue = request_queues[self.rank]
        self.sender_thread = None

        register_latent_manager(rank, self, request_queues)

        if auto_start:
            self.start_sender_thread()

    def _prune_expired(self):
        now = time.time()
        if now - self._last_cleanup_ts < 1.0:
            return
        self._last_cleanup_ts = now
        expired = [
            sub_req_id
            for sub_req_id, entry in self.latent_store.items()
            if now - entry["saved_at"] > self.latent_ttl_seconds
        ]
        for sub_req_id in expired:
            logger.warning(f"[Rank {self.rank}] Expiring latent sub_req_id={sub_req_id}")
            del self.latent_store[sub_req_id]

    def start_sender_thread(self) -> None:
        """Start the background thread to process incoming requests from the queue."""

        def loop():
            # Set CUDA device for this thread. In torchrun, each process has a specific GPU
            # (its LOCAL_RANK). New threads don't automatically inherit the main thread's current
            # CUDA device — torch.cuda.set_device() is per-thread. Without this, dist.send()
            # raises "Event device does not match recording stream's device".
            if torch.cuda.is_available():
                torch.cuda.set_device(self.rank)
            logger.info(f"[Rank {self.rank} Sender Thread] Starting loop, listening on its queue.")
            while True:
                try:
                    request_info = self.request_queue.get(block=True, timeout=0.05)
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.warning(f"[Rank {self.rank} Sender Thread] Queue error: {e}")
                    time.sleep(0.1)
                    continue

                # Check for stop signal *after* successful get()
                if request_info is None:
                    logger.info(f"[Rank {self.rank} Sender Thread] Received stop signal (None).")
                    break  # Exit the loop cleanly

                if not isinstance(request_info, tuple) or len(request_info) != 2:
                    logger.warning(
                        f"[Rank {self.rank} Sender Thread] Received invalid request format (expected tuple(2)): {request_info}"
                    )
                    continue

                requester_rank, sub_req_id = request_info
                # logger.info(f"[Rank {self.rank} Sender Thread] Got request from Rank {requester_rank} for sub_req_id {sub_req_id}")

                self._prune_expired()
                # If tensor not in store, it may still be being computed by the main thread
                # (round N and round N+1 dispatched concurrently). Retry briefly before giving up.
                if sub_req_id not in self.latent_store and requester_rank != self.rank:
                    _RETRY_DEADLINE = time.time() + 10.0
                    while sub_req_id not in self.latent_store and time.time() < _RETRY_DEADLINE:
                        time.sleep(0.02)
                if sub_req_id in self.latent_store:
                    tensors = self.latent_store[sub_req_id]["tensors"]
                    try:
                        # Check if this is self-communication (same rank sending to itself)
                        if requester_rank == self.rank:
                            logger.debug(
                                f"[Rank {self.rank} Sender Thread] Self-communication detected for sub_req_id={sub_req_id}, skipping distributed send"
                            )
                            # For self-communication, we don't need to send via distributed communication
                            # The tensors are already available in local storage and accessed directly
                        else:
                            # Get dedicated latent transfer group for this communication
                            latent_group = PROCESS_GROUP.get_latent_transfer_group(self.rank, requester_rank)

                            status = torch.tensor([1], dtype=torch.int32, device=get_device())
                            dist.send(tensor=status, dst=requester_rank, group=latent_group)

                            # Send each tensor using distributed communication with dedicated group.
                            # NOTE: Do NOT call torch.cuda.synchronize() here — it causes
                            # "Event device does not match recording stream's device" errors in
                            # multi-threaded NCCL contexts. NCCL handles stream synchronization
                            # internally; explicit synchronize from a background thread is unsafe.
                            for tensor_name, tensor in tensors.items():
                                dist.send(tensor=tensor.contiguous(), dst=requester_rank, group=latent_group)
                    except Exception as e:
                        logger.error(
                            f"[Rank {self.rank} Sender Thread] Failed to send tensors for {sub_req_id} to rank {requester_rank}: {e}"
                        )
                    finally:
                        # Only delete from store after sending, or if it's self-communication
                        if requester_rank != self.rank:
                            del self.latent_store[sub_req_id]
                        # For self-communication, keep the tensor in store as the receiver might still need it
                else:
                    # This might happen if save_latent hasn't completed yet, or if it was already sent and deleted.
                    logger.warning(
                        f"[Rank {self.rank} Sender Thread] Send request for sub_req_id={sub_req_id}, but not found in local store."
                    )
                    if requester_rank != self.rank:
                        latent_group = PROCESS_GROUP.get_latent_transfer_group(self.rank, requester_rank)
                        status = torch.tensor([0], dtype=torch.int32, device=get_device())
                        dist.send(tensor=status, dst=requester_rank, group=latent_group)

            logger.info(f"[Rank {self.rank} Sender Thread] Exiting loop.")

        # Start the daemon thread
        self.sender_thread = threading.Thread(target=loop, daemon=True)
        self.sender_thread.start()
        logger.info(f"[Rank {self.rank}] Sender thread started.")

    def save_latent(
        self,
        sub_req_id: int,
        tensors: Union[torch.Tensor, Dict[str, torch.Tensor]],
        save_tensors: bool = True,
        step_range=None,
    ) -> LatentFuture:
        """
        Saves the tensors locally.

        Args:
            sub_req_id: The sub request ID
            tensors: Either a single tensor or a dictionary mapping tensor names to tensors
        """
        # Convert single tensor to dictionary format
        if isinstance(tensors, torch.Tensor):
            tensors = {"latents": tensors}

        self._prune_expired()

        if save_tensors:
            # Move all tensors to device, clone to avoid corruption, and store them
            stored_tensors = {}
            for name, tensor in tensors.items():
                stored_tensors[name] = tensor.to(get_device()).detach().clone()
            self.latent_store[sub_req_id] = {"tensors": stored_tensors, "saved_at": time.time()}

        # Create tensor info for LatentFuture
        tensors_info = {
            name: TensorInfo(shape=tuple(tensor.shape), dtype=tensor.dtype, device=str(tensor.device))
            for name, tensor in tensors.items()
        }

        return LatentFuture(
            sub_req_id=sub_req_id,
            src_rank=self.rank,
            tensors_info=tensors_info,
            step_range=step_range,
        )

    def has_local_tensor(self, sub_req_id: int) -> bool:
        return sub_req_id in self.latent_store

    def get_local_tensor(self, sub_req_id: int) -> Union[torch.Tensor, Dict[str, torch.Tensor], None]:
        """
        Get tensor directly from local storage without distributed communication.
        Used for self-communication cases.

        Args:
            sub_req_id: The sub request ID

        Returns:
            The tensors if available, None otherwise
        """
        self._prune_expired()
        if sub_req_id in self.latent_store:
            tensors = self.latent_store[sub_req_id]["tensors"]
            # If there's only one tensor, return it directly for backward compatibility
            if len(tensors) == 1:
                return next(iter(tensors.values()))
            return tensors
        return None

    def pop_local_tensor(self, sub_req_id: int) -> Union[torch.Tensor, Dict[str, torch.Tensor], None]:
        self._prune_expired()
        if sub_req_id not in self.latent_store:
            return None
        tensors = self.latent_store[sub_req_id]["tensors"]
        del self.latent_store[sub_req_id]
        if len(tensors) == 1:
            return next(iter(tensors.values()))
        return tensors

    def stop_sender_thread(self):
        """Signals the sender thread to stop by putting None on its queue."""
        if self.sender_thread and self.sender_thread.is_alive():
            logger.info(f"[Rank {self.rank}] Signaling sender thread to stop.")
            # Put None onto this rank's queue
            self.request_queue.put(None)
            # Optionally join the thread if you need to wait for it
            # self.sender_thread.join(timeout=1.0)

        # Remove from global registry when stopping
        unregister_latent_manager(self.rank)
