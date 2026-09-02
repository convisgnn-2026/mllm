import pandas as pd
from dataclasses import dataclass
import json
import math
from typing import Dict, Tuple
from tetriserve.xfuser.logger import init_logger

logger = init_logger(__name__)


class E2ECostEstimator:
    def __init__(self, num_gpus: int, e2e_scaling_efficiency_path: str):
        with open(e2e_scaling_efficiency_path, "r") as f:
            data = json.load(f)

        self.e2e_scaling_efficiency = {}

        # (image_size, batch_size, gpu_num) -> (u_degree, r_degree, text_encoding_time, denoising_time_per_step)
        self.e2e_best_config_dict: Dict[Tuple[int, int, int], Tuple[int, int, float, float]] = {}

        # logger.info(f"scaling efficiency data: {e2e_scaling_efficiency_path}")

        for entry in data:
            if entry["ring_degree"] > 1:
                continue
            key_e2e_scaling_efficiency = (
                entry["bs"],
                entry["height"] * entry["width"],  # image size
                entry["ulysses_degree"],
                entry["ring_degree"],
                entry["mlp_world_size"],
            )
            self.e2e_scaling_efficiency[key_e2e_scaling_efficiency] = (
                entry["avg_text_encoding_time"],
                entry["avg_denoising_time"] / entry["steps"],
            )

            key_e2e_best_config = (
                entry["height"] * entry["width"],
                entry["bs"],
                entry["ulysses_degree"] * entry["ring_degree"],
            )
            if (key_e2e_best_config not in self.e2e_best_config_dict) or (
                self.e2e_best_config_dict[key_e2e_best_config][-1]
                > self.e2e_scaling_efficiency[key_e2e_scaling_efficiency][-1]
            ):
                self.e2e_best_config_dict[key_e2e_best_config] = (
                    entry["ulysses_degree"],
                    entry["ring_degree"],
                    entry["avg_text_encoding_time"],
                    entry["avg_denoising_time"] / entry["steps"],
                )
                # if key_e2e_best_config == (4194304, 1, 1):
                #     logger.info(f"key_e2e_best_config: {key_e2e_best_config}, ulysses_degree: {entry['ulysses_degree']}, ring_degree: {entry['ring_degree']}, avg_text_encoding_time: {entry['avg_text_encoding_time']}, avg_denoising_time: {entry['avg_denoising_time'] / entry['steps']}")
                #     logger.info(f"e2e_scaling_efficiency: {entry['avg_denoising_time']} {entry['steps']}")

        self.num_gpus = num_gpus

    def estimate_cost(
        self, bs, image_size, running_steps, gpu_num=None, is_text_encoding_time_included=True, fixed_gpu_num=None
    ):
        """
        Estimate the cost of serving a request with the given parameters
        """
        if gpu_num is None:
            gpu_num = self.num_gpus

        min_cost = float("inf")
        min_u_degree = 0
        min_r_degree = 0

        for use_gpu_num in [1, 2, 4, 8]:
            if use_gpu_num > gpu_num:
                continue

            if fixed_gpu_num is not None:
                if use_gpu_num != fixed_gpu_num:
                    continue

            u_degree, r_degree, text_encoding_time, denoising_time_per_step = self.e2e_best_config_dict[
                (image_size, bs, use_gpu_num)
            ]

            if is_text_encoding_time_included:
                cost = text_encoding_time + denoising_time_per_step * running_steps
            else:
                cost = denoising_time_per_step * running_steps

            if cost < min_cost:
                min_cost = cost
                min_u_degree = u_degree
                min_r_degree = r_degree

        return min_u_degree, min_r_degree, min_cost

    def compute_optimal_shape(self, bs, image_size, latency_requirement=None):
        """
        Compute the optimal shape for the given parameters
        """
        raise NotImplementedError
