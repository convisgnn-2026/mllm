from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import networkx as nx
from tetriserve.xfuser.config import InputConfig

from tetriserve.xfuser.logger import init_logger

logger = init_logger(__name__)


@dataclass
class UnScheduledRequest:
    """
    Represents a request that has been submitted but not yet assigned GPU resources.

    ID hierarchy:
        req_id  — The original API request identifier (1 per API call).
        sub_req_id — One per batch item within a req_id. When batch_size=4,
                     a single req_id produces 4 sub_req_ids. Sub-requests
                     can be scheduled independently on different GPU groups.
        The mapping is tracked in RequestTracker.req_id_to_sub_req_ids.

    Attributes:
        req_id (int): Unique identifier for the original request
        sub_req_ids (List[int]): Unique ID for each batch item in this request
        input_config (InputConfig): Configuration for model input parameters
        latent_length (int): Length of latent representation for the request
        arrival_time (float): Timestamp when the request arrived
        origin_steps (List[int], optional): Original inference steps for each sub-request
        latency_threshold (float): Maximum allowed latency for the request
    """

    req_id: int
    sub_req_ids: List[int]
    input_config: InputConfig
    latent_length: int
    arrival_time: float
    origin_steps: List[int] = None
    latency_threshold: float = float("inf")

    def __post_init__(self) -> None:
        if self.origin_steps is None:
            self.origin_steps = [self.input_config.num_inference_steps] * len(self.sub_req_ids)


@dataclass
class ScheduledRequests:
    """
    Represents a request that has been assigned GPU resources and execution parameters.

    Typically contains a single request (len(req_ids) = 1), but can batch multiple
    requests together for efficiency (len(req_ids) > 1).

    Attributes:
        req_ids (List[int]): List of request IDs (usually just one)
        sub_req_ids (List[int]): List of sub-request IDs for batch items
        parallel_ranks (List[int]): GPU ranks assigned for parallel execution
        attn_ulysses_degree (int): Ulysses attention parallelism degree
        attn_ring_degree (int): Ring attention parallelism degree
        sp_degree (int): Sequence parallelism degree
        latent_length (int): Length of latent representation
        input_config (InputConfig): Model input configuration
        origin_steps (List[int]): Original inference steps for each sub-request
        start_step (int): Start step (inclusive) for this scheduled stage
        end_step (int): End step (exclusive) for this scheduled stage
        is_last_steps (List[bool]): Whether each sub-request is in its final step
        arrival_time (float): When the request arrived
        latency_threshold (float): Maximum allowed latency
        estimated_time (float): Estimated execution time
    """

    req_ids: List[int]
    sub_req_ids: List[int]
    parallel_ranks: List[int]
    attn_ulysses_degree: int
    attn_ring_degree: int
    sp_degree: int
    latent_length: int
    input_config: InputConfig
    origin_steps: List[int]
    start_step: int
    end_step: int
    is_last_steps: List[bool]
    arrival_time: float
    latency_threshold: float
    estimated_time: float


@dataclass
class ScheduledRequestsBlock:
    """
    Basic scheduling unit returned by scheduler.schedule() and sent to workers.

    Contains a group of ScheduledRequests that can be executed together in the same
    scheduling iteration. Includes conditions for when execution can begin and end.

    Attributes:
        requests (List[ScheduledRequests]): List of scheduled requests to execute
        begin_condition_ranks (List[int]): GPU ranks that must be free before execution
        end_condition_req_ids (List[int]): Request IDs that must complete before dependent tasks
    """

    requests: List[ScheduledRequests] = field(default_factory=list)
    begin_condition_ranks: List[int] = field(default_factory=list)
    end_condition_req_ids: List[int] = field(default_factory=list)

    def meet_begin_condition(self, free_ranks: List[int]) -> bool:
        """Check if all required GPU ranks are available for execution."""
        return all(rank in free_ranks for rank in self.begin_condition_ranks)

    def meet_end_condition(self, finished_req_ids: List[int]) -> bool:
        """Check if all required requests have completed execution."""
        return all(req_id in finished_req_ids for req_id in self.end_condition_req_ids)

    def __len__(self) -> int:
        return len(self.requests)


class ScheduledRequestsBlockCache:
    """
    Cache for managing scheduled request blocks using a Directed Acyclic Graph (DAG).

    This cache manages complex scheduling dependencies between request blocks,
    allowing for sophisticated scheduling patterns like multi-stage execution
    and resource dependency management.

    Attributes:
        schedule_DAG (nx.DiGraph): NetworkX directed graph representing scheduling dependencies
    """

    def __init__(self) -> None:
        self.schedule_DAG = nx.DiGraph()

    def convert_to_unscheduled_requests(self) -> List[UnScheduledRequest]:
        """Convert cached scheduled requests back to unscheduled requests and clear the DAG."""

        if len(self.schedule_DAG.nodes) == 0:
            return []

        ret_unscheduled_requests_dict: Dict[int, UnScheduledRequest] = {}

        for node in self.schedule_DAG.nodes:
            req_block = self.schedule_DAG.nodes[node]["req_block"]
            for req in req_block.requests:
                for req_id in req.req_ids:
                    if req_id not in ret_unscheduled_requests_dict:
                        ret_unscheduled_requests_dict[req_id] = UnScheduledRequest(
                            req_id=req_id,
                            sub_req_ids=req.sub_req_ids,
                            input_config=req.input_config,
                            latent_length=req.latent_length,
                            arrival_time=req.arrival_time,
                            latency_threshold=req.latency_threshold,
                        )
                    else:
                        ret_unscheduled_requests_dict[
                            req_id
                        ].input_config.num_inference_steps += req.input_config.num_inference_steps

        ret_unscheduled_requests: List[UnScheduledRequest] = list(ret_unscheduled_requests_dict.values())
        self.schedule_DAG = nx.DiGraph()

        return ret_unscheduled_requests

    def update_schedule_DAG(self, schedule_DAG: nx.DiGraph) -> None:
        """Update the internal scheduling DAG with a new graph."""
        self.schedule_DAG = schedule_DAG

    def get_new_free_nodes(self) -> List[Tuple["ScheduledRequestsBlock", int]]:
        """Get request blocks that have no dependencies and can be executed immediately."""
        ret_scheduled_reqs_block_idxs = [
            node for node in self.schedule_DAG.nodes if self.schedule_DAG.in_degree(node) == 0
        ]
        ret_scheduled_reqs_blocks = [
            (self.schedule_DAG.nodes[node]["req_block"], node) for node in ret_scheduled_reqs_block_idxs
        ]
        return ret_scheduled_reqs_blocks

    def remove_node(self, node_idx: int) -> None:
        """Remove a completed request block from the DAG."""
        self.schedule_DAG.remove_node(node_idx)

    def pop(self) -> List["ScheduledRequestsBlock"]:
        """Get and remove all free request blocks from the DAG."""
        if len(self.schedule_DAG.nodes) == 0:
            return []
        ret_scheduled_reqs_block_idxs = [
            node for node in self.schedule_DAG.nodes if self.schedule_DAG.in_degree(node) == 0
        ]
        ret_scheduled_reqs_blocks = [
            self.schedule_DAG.nodes[node]["req_block"] for node in ret_scheduled_reqs_block_idxs
        ]
        self.schedule_DAG.remove_nodes_from(ret_scheduled_reqs_block_idxs)
        return ret_scheduled_reqs_blocks

    def __len__(self) -> int:
        return len(self.schedule_DAG.nodes)

    def __getitem__(self, idx: int) -> "ScheduledRequestsBlock":
        return self.schedule_DAG.nodes[idx]["req_block"]
