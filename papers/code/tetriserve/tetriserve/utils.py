##########################
# Time Counter
##########################
import time
import asyncio

START_TIME = None


def get_current_time():
    global START_TIME
    if START_TIME is None:
        START_TIME = time.time()
    return time.time() - START_TIME


def reset_start_time():
    """Reset the global start time. For testing only."""
    global START_TIME
    START_TIME = None


##########################
# Request ID Counter
##########################
class Counter:
    def __init__(self, start: int = 0) -> None:
        self.counter = start
        self._lock = asyncio.Lock()

    async def get_id(self) -> int:
        """Get the next ID in a thread-safe and async-safe manner."""
        async with self._lock:
            i = self.counter
            self.counter += 1
            return i

    async def reset(self) -> None:
        """Reset the counter to 0 in a thread-safe and async-safe manner."""
        async with self._lock:
            self.counter = 0


gen_req_id = Counter()
gen_sub_req_id = Counter(start=1)


async def get_gen_req_id():
    return await gen_req_id.get_id()


async def get_gen_sub_req_id():
    return -(await gen_sub_req_id.get_id())


async def reset_gen_req_id():
    await gen_req_id.reset()
    await gen_sub_req_id.reset()


##########################
# Seed
##########################
import os
import random
import numpy as np
import torch


def set_all_seeds(seed: int = 42):
    """Set all seeds for reproducibility"""
    # Python's random module
    random.seed(seed)

    # Numpy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU

    # Set PYTHONHASHSEED
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Optional: make PyTorch deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Set all seeds at the start of the script
# set_all_seeds(42)

##########################
# Device
##########################

DEVICE = None


def set_device(gpu: bool = True):
    global DEVICE
    if DEVICE is None:
        DEVICE = torch.device("cuda") if torch.cuda.is_available() and gpu else torch.device("cpu")


def get_device():
    global DEVICE
    return DEVICE


def reset_device():
    """Reset the global device. For testing only."""
    global DEVICE
    DEVICE = None


##########################
# Utils
##########################
from itertools import combinations


def generate_all_subsets(n):
    """
    Generate all subsets of a list of numbers from 0 to n-1.

    Parameters:
    n (int): The number specifying the range (0 to n-1).

    Returns:
    list: A list of all subsets represented as lists.
    """
    nums = list(range(n))
    return [list(subset) for r in range(1, n + 1) for subset in combinations(nums, r)]


def get_gpu_name(gpu_name=None):
    """Get formatted GPU information for directory naming."""

    gpu_name = os.environ.get("GPU_NAME", None)

    if gpu_name is not None:
        return gpu_name

    if not torch.cuda.is_available():
        print(
            f"Warning: CUDA is not available, using default GPU name in Environment Variable 'GPU_NAME': {os.environ.get('GPU_NAME', 'A40')}"
        )
        return os.environ.get("GPU_NAME", "H100_HBM3")

    try:
        # Get GPU name and properties
        gpu_name = torch.cuda.get_device_name(0)
        gpu_properties = torch.cuda.get_device_properties(0)

        # Extract GPU model (e.g., "NVIDIA H100 80GB HBM3" -> "H100")
        gpu_model = "Unknown"
        if "H100" in gpu_name:
            gpu_model = "H100"
        elif "A100" in gpu_name:
            gpu_model = "A100"
        elif "V100" in gpu_name:
            gpu_model = "V100"
        elif "RTX" in gpu_name:
            if "4090" in gpu_name:
                gpu_model = "RTX4090"
            elif "3090" in gpu_name:
                gpu_model = "RTX3090"
            else:
                gpu_model = "RTX"
        elif "Tesla" in gpu_name:
            gpu_model = "Tesla"
        else:
            # Fallback: extract first word after "NVIDIA"
            parts = gpu_name.split()
            for i, part in enumerate(parts):
                if "NVIDIA" in part and i + 1 < len(parts):
                    gpu_model = parts[i + 1]
                    break

        # Extract memory type if available in name
        memory_type = ""
        if "HBM3" in gpu_name:
            memory_type = "_HBM3"
        elif "HBM2" in gpu_name:
            memory_type = "_HBM2"
        elif "GDDR6" in gpu_name:
            memory_type = "_GDDR6"

        # Format: ModelxCount_MemoryType (e.g., H100x8_HBM3)
        gpu_folder = f"{gpu_model}{memory_type}"

        os.environ["GPU_NAME"] = gpu_folder

        return gpu_folder
    except Exception as e:
        print(f"Warning: Could not detect GPU info, using default. Error: {e}")
        return "GPU_unknown"
