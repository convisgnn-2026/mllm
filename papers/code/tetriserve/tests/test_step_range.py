"""
Test: Step range slicing for timesteps and sigmas.
"""

import torch


def test_step_range_slice_lengths():
    timesteps = torch.arange(0, 50, dtype=torch.float32)
    sigmas = torch.arange(0, 51, dtype=torch.float32)
    start_step = 0
    end_step = 25

    sliced_timesteps = timesteps[start_step:end_step]
    sliced_sigmas = sigmas[start_step : end_step + 1]

    assert len(sliced_timesteps) == 25
    assert len(sliced_sigmas) == 26


def test_step_range_slice_contents():
    timesteps = torch.arange(0, 10, dtype=torch.float32)
    sigmas = torch.arange(0, 11, dtype=torch.float32)
    start_step = 3
    end_step = 7

    sliced_timesteps = timesteps[start_step:end_step]
    sliced_sigmas = sigmas[start_step : end_step + 1]

    assert torch.equal(sliced_timesteps, torch.tensor([3, 4, 5, 6], dtype=torch.float32))
    assert torch.equal(sliced_sigmas, torch.tensor([3, 4, 5, 6, 7], dtype=torch.float32))
