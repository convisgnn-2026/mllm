"""
adapted from yunchang: https://github.com/feifeibear/long-context-attention
"""

from typing import List, Dict

import torch
import time

from tetriserve.utils import generate_all_subsets, get_device
from tetriserve.distributed.comm import all_to_all_4D
from tetriserve.xfuser.logger import init_logger

logger = init_logger(__name__)

HAS_SPARSE_SAGE_ATTENTION = False


class Singleton:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Singleton, cls).__new__(cls, *args, **kwargs)
        return cls._instance


class ProcessGroupSingleton(Singleton):
    def __init__(self):
        self.all_process_groups: List[torch.distributed.ProcessGroup] = []
        self.all_process_groups_map: Dict[str, torch.distributed.ProcessGroup] = {}

        self.all_cpu_groups: List[torch.distributed.ProcessGroupGloo] = []
        self.all_cpu_groups_map: Dict[str, torch.distributed.ProcessGroupGloo] = {}

        self.ATTN_SP_PG = None
        self.ULYSSES_PG = None
        self.RING_PG = None

        # Dedicated process groups for latent transfer (separate from general purpose groups)
        self.latent_transfer_groups: Dict[str, torch.distributed.ProcessGroup] = {}
        self.latent_transfer_cpu_groups: Dict[str, torch.distributed.ProcessGroupGloo] = {}

        self.parallel_ranks = []

        self.generated = False

        self.custom_nccl_comm = None

    def get_process_group(self, ranks: List[int], device: str = "cuda"):
        assert self.generated, "all process groups are not generated"
        # Ensure ranks are sorted to match the key used during creation
        ranks_key = str(sorted(ranks))
        try:
            if device == "cuda":
                return self.all_process_groups_map[ranks_key]
            elif device == "cpu":
                return self.all_cpu_groups_map[ranks_key]
            else:
                raise ValueError(f"Invalid device: {device}")
        except KeyError:
            logger.error(
                f"Process group for ranks {ranks} (key: {ranks_key}) not found in map. Available keys: {list(self.all_process_groups_map.keys())}"
            )
            raise

    def get_latent_transfer_group(self, src_rank: int, dst_rank: int, device: str = "cuda"):
        """
        Get a dedicated DIRECTED process group for latent transfer from src_rank to dst_rank.
        Each directed pair (src, dst) has its own communicator — distinct from (dst, src).
        This prevents NCCL sequence-number races when two ranks simultaneously transfer
        latents to each other (bidirectional concurrent use of an undirected group causes
        non-deterministic SeqNum ordering → ncclSystemError deadlock).
        """
        assert self.generated, "all process groups are not generated"

        # Use directed key: "[src_rank, dst_rank]" is different from "[dst_rank, src_rank]"
        ranks_key = str([src_rank, dst_rank])

        try:
            if device == "cuda":
                return self.latent_transfer_groups[ranks_key]
            elif device == "cpu":
                return self.latent_transfer_cpu_groups[ranks_key]
            else:
                raise ValueError(f"Invalid device: {device}")
        except KeyError:
            logger.error(
                f"Directed latent transfer group [{src_rank}→{dst_rank}] (key: {ranks_key}) not found. "
                f"Available keys: {list(self.latent_transfer_groups.keys())}"
            )
            raise

    def _create_latent_transfer_groups(self, world_size: int):
        """
        Create dedicated DIRECTED process groups for latent transfer.

        Each directed pair (i→j) gets its own communicator, separate from (j→i).
        This prevents Bug 8: concurrent bidirectional use of an undirected [i,j] group
        causes non-deterministic NCCL SeqNum ordering when rank i's sender thread sends
        to rank j while rank j's sender thread simultaneously sends to rank i — both
        using the same communicator from different threads. With directed groups, each
        communicator is used only by one sender thread and one receiver thread.

        Creates 56 directed groups (N*(N-1) = 8*7 = 56 for N=8) instead of 28 undirected.
        Memory overhead is ~2× but still negligible vs. FLUX model (~50GB/GPU).
        Each rank warms up 14 communicators (7 outgoing + 7 incoming) instead of 7.
        """
        if not torch.distributed.is_initialized():
            logger.error("Distributed environment not initialized")
            raise RuntimeError("Distributed environment not initialized.")

        initialized_backend = torch.distributed.get_backend()

        # Create directed process groups for all ordered pairs (i→j) where i != j.
        # new_group() is a collective — ALL ranks must call it for EVERY group,
        # regardless of whether they participate. We create two groups per pair (i,j):
        # one for i→j and one for j→i.
        created_count = 0
        for i in range(world_size):
            for j in range(world_size):
                if i == j:
                    continue
                ranks = [i, j]
                ranks_key = str(ranks)  # e.g. "[0, 4]" for 0→4 (different from "[4, 0]")

                try:
                    latent_cuda_group = torch.distributed.new_group(ranks, backend=initialized_backend)
                    latent_cpu_group = torch.distributed.new_group(ranks, backend="gloo")

                    time.sleep(0.001)

                    if latent_cuda_group is not None and latent_cpu_group is not None:
                        self.latent_transfer_groups[ranks_key] = latent_cuda_group
                        self.latent_transfer_cpu_groups[ranks_key] = latent_cpu_group
                        created_count += 1
                    else:
                        logger.warning(f"Failed to create directed latent transfer group [{i}→{j}]")

                except Exception as e:
                    logger.error(f"Failed to create directed latent transfer group [{i}→{j}]: {e}")

        logger.info(f"Created {created_count} directed latent transfer process groups")

        # Warm up NCCL communicators in the main thread before any sender threads start.
        # For directed group (i→j): rank i sends, rank j receives.
        # Each rank warms up 7 outgoing groups (as sender) + 7 incoming (as receiver) = 14.
        current_rank = torch.distributed.get_rank()
        warmup_count = 0
        for i in range(world_size):
            for j in range(world_size):
                if i == j:
                    continue
                if current_rank not in (i, j):
                    continue
                ranks_key = str([i, j])
                if ranks_key not in self.latent_transfer_groups:
                    continue
                group = self.latent_transfer_groups[ranks_key]
                dummy = torch.zeros(1, device=get_device())
                # In group [i, j]: rank i is the designated sender, rank j the receiver
                if current_rank == i:
                    torch.distributed.send(dummy, dst=j, group=group)
                else:
                    torch.distributed.recv(dummy, src=i, group=group)
                warmup_count += 1
        logger.info(f"[Rank {current_rank}] Warmed up {warmup_count} directed latent transfer NCCL communicators")

    def generate_all_process_groups(self, world_size, skip_latent_transfer: bool = False):
        assert not self.generated, "all process groups are already generated"
        # Get the backend of the already initialized default group
        if not torch.distributed.is_initialized():
            logger.error("Default process group not initialized before calling generate_all_process_groups")
            raise RuntimeError("Distributed environment not initialized.")
        initialized_backend = torch.distributed.get_backend()

        # First, create dedicated latent transfer groups (skip if not needed)
        if not skip_latent_transfer:
            self._create_latent_transfer_groups(world_size)
        else:
            logger.info("Skipping latent transfer group creation (skip_latent_transfer=True)")

        # Then create general purpose process groups
        all_subsets = generate_all_subsets(world_size)
        # continuous_subsets = filter_continuous_sequences(all_subsets)
        continuous_subsets = all_subsets  # Using all subsets for now
        created_count = 0
        skipped_count = 0
        failed_count = 0
        for subset in continuous_subsets:
            sorted_subset = sorted(list(subset))
            group_key = str(sorted_subset)
            # Avoid recreating if somehow called multiple times or subsets overlap unexpectedly
            if group_key in self.all_process_groups_map:
                continue

            try:
                # Use the same backend as the initialized default group
                # new_group is collective, all ranks in sorted_subset must call it
                group = torch.distributed.new_group(sorted_subset, backend=initialized_backend)
                cpu_group = torch.distributed.new_group(sorted_subset, backend="gloo")

                # Small delay to prevent NCCL race conditions
                time.sleep(0.01 * (len(sorted_subset) / world_size))  # Scale delay with group complexity

                if group is not None:  # new_group returns None if ranks are invalid/disappeared
                    self.all_process_groups.append(group)
                    self.all_cpu_groups.append(cpu_group)
                    # Use the sorted list string for mapping
                    self.all_process_groups_map[group_key] = group
                    self.all_cpu_groups_map[group_key] = cpu_group
                    created_count += 1
                else:
                    # This case might indicate ranks that are gone, less likely at init
                    #  logger.warning(f"torch.distributed.new_group returned None for ranks {sorted_subset} with backend {initialized_backend}")
                    failed_count += 1
            except Exception as e:
                # Log error if group creation fails for this specific subset
                logger.error(
                    f"Failed to create process group for ranks {sorted_subset} with backend {initialized_backend}: {e}",
                    exc_info=True,
                )
                failed_count += 1
                # Decide how to handle: raise? continue?
                # If one fails, likely others will too. Maybe raise the first error?
                # For now, log and continue to see if any succeed.

        self.generated = True
        logger.info(
            f"Generated {created_count} general process subgroups (skipped {skipped_count}, failed {failed_count}). Map size: {len(self.all_process_groups_map)}"
        )
        if failed_count > 0:
            logger.warning(f"{failed_count} subgroups failed to be created. This may cause errors later.")

    def set_seq_parallel_pg(self, sp_name: str = "ulysses", ranks: List[int] = []):
        if sp_name == "ulysses":
            self.ULYSSES_PG = self.get_process_group(ranks)
        elif sp_name == "ring":
            self.RING_PG = self.get_process_group(ranks)
        elif sp_name == "attn_sp":
            self.ATTN_SP_PG = self.get_process_group(ranks)
        else:
            raise ValueError(f"Invalid SP name: {sp_name}")

    def destroy(self):
        logger.info(f"Destroying {len(self.all_process_groups)} custom process groups.")
        for pg in self.all_process_groups:
            try:
                torch.distributed.destroy_process_group(pg)
            except Exception as e:
                logger.error(f"Error destroying process group {pg}: {e}")

        # Destroy dedicated latent transfer groups
        logger.info(f"Destroying {len(self.latent_transfer_groups)} latent transfer process groups.")
        for pg in self.latent_transfer_groups.values():
            try:
                torch.distributed.destroy_process_group(pg)
            except Exception as e:
                logger.error(f"Error destroying latent transfer group {pg}: {e}")

        for pg in self.latent_transfer_cpu_groups.values():
            try:
                torch.distributed.destroy_process_group(pg)
            except Exception as e:
                logger.error(f"Error destroying latent transfer CPU group {pg}: {e}")

        self.all_process_groups = []
        self.all_process_groups_map = {}
        self.latent_transfer_groups = {}
        self.latent_transfer_cpu_groups = {}
        self.generated = False


PROCESS_GROUP = ProcessGroupSingleton()


def set_seq_parallel_pg(sp_ulysses_degree, sp_ring_degree, rank, ranks, world_size):
    """
    sp_ulysses_degree x sp_ring_degree = seq_parallel_degree
    (ulysses_degree, dp_degree)
    """
    sp_degree = sp_ring_degree * sp_ulysses_degree

    assert len(ranks) == world_size, f"len(ranks) {len(ranks)} != world_size {world_size}"
    assert world_size == sp_degree, f"attn world_size {world_size} != sp_degree {sp_degree}"

    num_ulysses_pgs = sp_ring_degree  # world_size // sp_ulysses_degree
    num_ring_pgs = sp_ulysses_degree  # world_size // sp_ring_degree

    PROCESS_GROUP.set_seq_parallel_pg(sp_name="attn_sp", ranks=ranks)

    for i in range(num_ulysses_pgs):
        ulysses_ranks_idx = list(
            range(
                i * sp_ulysses_degree,
                (i + 1) * sp_ulysses_degree,
            )
        )
        ulysses_ranks = [ranks[index_req] for index_req in ulysses_ranks_idx]
        if rank in ulysses_ranks:
            PROCESS_GROUP.set_seq_parallel_pg(sp_name="ulysses", ranks=ulysses_ranks)

    for i in range(num_ring_pgs):
        ring_ranks_idx = list(range(i, sp_degree, num_ring_pgs))
        ring_ranks = [ranks[index_req] for index_req in ring_ranks_idx]
        if rank in ring_ranks:
            PROCESS_GROUP.set_seq_parallel_pg(sp_name="ring", ranks=ring_ranks)
