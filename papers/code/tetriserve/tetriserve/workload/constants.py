"""
Workload configuration constants.

These constants define common configurations for workload generation.
"""

# Default number of inference steps
STEPS = 20

# Shape configurations (height, width)
SHAPE_LIST = [(256, 256), (512, 512), (1024, 1024), (2048, 2048)]

# Batch size distribution (list of tuples: (batch_size, probability))
BS_DISTRIBUTION = [(1, 1.0)]

# Request rate configurations (requests per second)
RATE_LIST = [0.1, 0.3, 0.5, 0.6, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0]

# Inference steps configurations
STEPS_LIST = [4, 10, 20, 30, 40, 50]

__all__ = [
    "STEPS",
    "SHAPE_LIST",
    "BS_DISTRIBUTION",
    "RATE_LIST",
    "STEPS_LIST",
]
