from typing import List, Union
from dataclasses import dataclass
import asyncio
import PIL
import numpy as np
from tetriserve.light_ray.latent_future import LatentFuture


@dataclass
class RunnerOutput:
    sub_req_ids: List[int]
    images: Union[List[PIL.Image.Image], LatentFuture]
    run_steps: int = None
    parallel_ranks: List[int] = None
    bs: int = None
    latent_length: int = None


@dataclass
class EngineOutput:
    req_id: int
    image: Union[PIL.Image.Image, np.ndarray]


@dataclass
class ReqState:
    "class to store the state of a request"
    event: asyncio.Event
    output: EngineOutput | None
    success: bool
