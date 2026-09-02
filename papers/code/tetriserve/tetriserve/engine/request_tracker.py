from typing import Dict, List
from copy import deepcopy
import PIL.Image

from tetriserve.xfuser.config import InputConfig
from tetriserve.config.request import UnScheduledRequest
from tetriserve.engine.struct import RunnerOutput

from tetriserve.xfuser.logger import init_logger

logger = init_logger(__name__)

# VAE encodes images at 8x spatial downscaling (both height and width)
VAE_SPATIAL_DOWNSCALE = 8


class RequestTracker:
    def __init__(self) -> None:
        self.requests = (
            {}
        )  # {sub_req_id: UnScheduledRequest(req_id=req_id, sub_req_ids=[sub_req_id], input_config=input_config, latent_length=latent_length, arrival_time=arrival_time, latency_threshold=latency_threshold)}
        self.req_id_to_sub_req_ids = (
            {}
        )  # {req_id: {sub_req_id: bool}} True means not finished, False means finished
        self.sub_req_id_to_latent_images = {}  # {sub_req_id: latentfuture or images}

    def __getitem__(self, sub_req_id: int) -> UnScheduledRequest:
        return self.requests[sub_req_id]

    def __len__(self) -> int:
        return len(self.requests)

    def track_request(
        self,
        sub_req_ids: List[int],
        req_id: int,
        arrival_time: float,
        latency_threshold: float,
        input_config: InputConfig,
    ) -> None:
        for prompt_idx, sub_req_id in enumerate(sub_req_ids):
            new_config = deepcopy(input_config)
            new_config.batch_size = 1
            new_config.prompt = input_config.prompt[prompt_idx]
            self.requests[sub_req_id] = UnScheduledRequest(
                req_id=req_id,
                sub_req_ids=[sub_req_id],
                input_config=new_config,
                origin_steps=[input_config.num_inference_steps],
                latent_length=(input_config.height // VAE_SPATIAL_DOWNSCALE)
                * (input_config.width // VAE_SPATIAL_DOWNSCALE),
                arrival_time=arrival_time,
                latency_threshold=latency_threshold,
            )
        self.req_id_to_sub_req_ids[req_id] = {sub_req_id: True for sub_req_id in sub_req_ids}

    def get_requests(self) -> List[UnScheduledRequest]:
        """
        It is a splited version of the request, each sub_req_id is a request, and the request is not merged
        """
        return list(self.requests.values())

    def get_merged_requests(self) -> List[UnScheduledRequest]:
        """
        It is a merged version of the request, each req_id is a request, each request is a merged version of the sub_req_ids
        """
        ret_req: List[UnScheduledRequest] = []
        for req_id in self.req_id_to_sub_req_ids.keys():
            sub_reqs = self.get_requests_by_req_id(req_id)
            merged_req = deepcopy(sub_reqs[0])
            merged_req.sub_req_ids = [sub_req.sub_req_ids[0] for sub_req in sub_reqs]
            merged_req.input_config.batch_size = len(sub_reqs)
            merged_req.input_config.prompt = [sub_req.input_config.prompt for sub_req in sub_reqs]
            ret_req.append(merged_req)
        return ret_req

    def get_requests_by_req_id(self, req_id: int) -> List[UnScheduledRequest]:
        result_list = []
        for sub_req_id in self.req_id_to_sub_req_ids[req_id].keys():
            if self.req_id_to_sub_req_ids[req_id][sub_req_id]:
                result_list.append(self.requests[sub_req_id])

        return result_list

    def update_requests(self, runner_output: RunnerOutput) -> Dict[int, List[PIL.Image.Image]]:

        ret_images_list = {}  # {req_id: [images]}

        for sub_req_id, latent_or_image in zip(runner_output.sub_req_ids, runner_output.images):
            self.requests[sub_req_id].input_config.num_inference_steps -= runner_output.run_steps
            self.sub_req_id_to_latent_images[sub_req_id] = latent_or_image
            assert (
                self.requests[sub_req_id].input_config.num_inference_steps >= 0
            ), f"Steps cannot be negative for sub_req_id {sub_req_id}"
            if self.requests[sub_req_id].input_config.num_inference_steps == 0:
                req_id = self.requests[sub_req_id].req_id

                ret_list = self.remove_request(sub_req_id)
                if len(ret_list) > 0:
                    assert req_id not in ret_images_list, f"Request id must be unique"
                    ret_images_list[req_id] = ret_list

        return ret_images_list

    def remove_request(self, sub_req_id: int) -> List[PIL.Image.Image]:

        req_id = self.requests[sub_req_id].req_id

        del self.requests[sub_req_id]

        assert (
            self.req_id_to_sub_req_ids[req_id][sub_req_id] is True
        ), f"Sub request id must be not finished before removing"
        self.req_id_to_sub_req_ids[req_id][sub_req_id] = False

        if not any(self.req_id_to_sub_req_ids[req_id].values()):
            images = []
            for sub_req_id in self.req_id_to_sub_req_ids[req_id].keys():
                image = self.sub_req_id_to_latent_images[sub_req_id]

                logger.info(f"image: {image}")

                assert isinstance(image, PIL.Image.Image), f"Image must be a PIL.Image.Image"
                images.append(image)
            del self.req_id_to_sub_req_ids[req_id]
            return images
        else:
            return []

    def __repr__(self) -> str:
        """
        return current each requests req_id, corresponding sub_req_ids, steps and latency_threshold
        """
        ret_str = ""
        for req_id, sub_req_ids_dict in self.req_id_to_sub_req_ids.items():
            sub_req_ids = list(sub_req_ids_dict.keys())
            idx = next((k for k, v in enumerate(sub_req_ids) if self.req_id_to_sub_req_ids[req_id][v]), None)
            if idx is not None:
                ret_str += f"req_id: {req_id}, "
                ret_str += f"sub_req_ids: {sub_req_ids} {self.req_id_to_sub_req_ids[req_id]}, "
                ret_str += f"steps: {self.requests[sub_req_ids[idx]].input_config.num_inference_steps}, "
                ret_str += f"latency_threshold: {self.requests[sub_req_ids[idx]].latency_threshold}\n"

        return ret_str
