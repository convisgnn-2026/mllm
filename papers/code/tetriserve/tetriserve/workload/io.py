"""
Workload I/O - save and load workloads to/from files.

Supports JSON format for easy inspection and modification.
"""

import json
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

from .core import Workload, Request


def dump_workload(workload: Workload, file_path: str | Path) -> None:
    """
    Save workload to a JSON file.

    Args:
        workload: Workload to save
        file_path: Path to output file

    Example:
        >>> workload = WorkloadBuilder().with_model("flux").build()
        >>> dump_workload(workload, "workload.json")
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert workload to JSON-serializable format
    data = {
        "model_name": workload.get_model_name(),
        "requests": [],
        "arrival_times": workload.arrival_times.tolist(),
    }

    for req in workload.requests:
        req_data = {
            "req_idx": req.req_idx,
            "model_name": req.model_name,
            "batch_size": req.batch_size,
            "height": req.height,
            "width": req.width,
            "num_inference_steps": req.num_inference_steps,
            "prompt": req.prompt,
            "output_type": req.output_type,
            "slo": req.slo,
            "arrival_time": req.arrival_time,
            # Note: submit_time and completion_time are not saved
            # as they are filled during execution
        }
        data["requests"].append(req_data)

    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)


def load_workload(file_path: str | Path) -> Workload:
    """
    Load workload from a JSON file.

    Args:
        file_path: Path to workload file

    Returns:
        Loaded workload

    Example:
        >>> workload = load_workload("workload.json")
        >>> print(f"Loaded {len(workload)} requests")
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Workload file not found: {file_path}")

    with open(file_path, "r") as f:
        data = json.load(f)

    # Reconstruct requests
    requests = []
    for req_data in data["requests"]:
        req = Request(
            req_idx=req_data["req_idx"],
            model_name=req_data["model_name"],
            batch_size=req_data["batch_size"],
            height=req_data["height"],
            width=req_data["width"],
            num_inference_steps=req_data["num_inference_steps"],
            prompt=req_data["prompt"],
            output_type=req_data["output_type"],
            slo=req_data.get("slo"),
            arrival_time=req_data.get("arrival_time", 0.0),
        )
        requests.append(req)

    # Reconstruct workload
    arrival_times = data["arrival_times"]
    workload = Workload(requests, arrival_times)

    return workload
