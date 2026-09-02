import asyncio
import math
import json
from abc import ABC
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Tuple, Union, Dict

from tetriserve.xfuser.config import InputConfig

from tetriserve.engine.request_tracker import RequestTracker
from tetriserve.config.request import ScheduledRequests, UnScheduledRequest
from tetriserve.utils import get_current_time

from tetriserve.xfuser.logger import init_logger

logger = init_logger(__name__)


class Scheduler(ABC):
    def __init__(self, ranks: List[int]):
        self.request_tracker = RequestTracker()

        self.running_requests: List[int] = []  # sub_req_ids

        self.world_size = len(ranks)
        self.ranks = ranks

        self.schedule_idx_counter = 0
        self._is_new_request_coming = False

    @property
    def schedule_idx(self):
        return self.schedule_idx_counter

    @property
    def is_new_request_coming(self):
        return self._is_new_request_coming

    def add_request(
        self,
        request_id: int,
        sub_req_ids: List[int],
        input_config: InputConfig,
        arrival_time: float,
        latency_threshold: float = None,
    ):
        self._is_new_request_coming = True

        self.request_tracker.track_request(
            sub_req_ids=sub_req_ids,
            req_id=request_id,
            arrival_time=arrival_time,
            latency_threshold=latency_threshold,
            input_config=input_config,
        )

    def add_ranks(self, ranks: List[int]):
        self.ranks += ranks
        self.ranks = list(set(self.ranks))
        self.ranks.sort()

    def remove_ranks(self, ranks: List[int]):
        self.ranks = [rank for rank in self.ranks if rank not in ranks]
        self.ranks.sort()

    def running_full(self):
        """
        Check if all workers are running full
        """
        return len(self.ranks) == 0

    def schedule(self):
        """
        according to current queue, we should decide which request should be sent to which worker
        """
        raise NotImplementedError

    def remaining_requests(self):
        return len(self.request_tracker.get_requests())
