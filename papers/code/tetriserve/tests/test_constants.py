"""Test constants extracted from workload_legacy.py for backward compatibility."""

# Shape configurations for testing
SHAPE_LIST = [(256, 256), (512, 512), (1024, 1024), (2048, 2048)]

# Latent length distribution (for SD3 model)
LATENT_LENGTH_DISTRIBUTION = {
    (1, 256, 256): 256,
    (1, 512, 512): 1024,
    (1, 1024, 1024): 4096,
    (1, 2048, 2048): 16384,
}

# Batch size distribution (single batch for now)
BS_DISTRIBUTION = [1]
BS_DISTRIBUTION = [(bs, 1 / len(BS_DISTRIBUTION)) for bs in BS_DISTRIBUTION]

# Request rate configurations
RATE_LIST = [0.2, 0.5, 1.0, 2.0]

# Inference steps configurations
STEPS_LIST = [20, 50]

# SLO dictionary (baseline latencies for SD3 model)
# NOTE: These are example values, should be loaded from actual profiling data
SLO_DICT = {
    (1, 256, 256): 1.0,
    (1, 512, 512): 2.0,
    (1, 1024, 1024): 4.0,
    (1, 2048, 2048): 8.0,
}
