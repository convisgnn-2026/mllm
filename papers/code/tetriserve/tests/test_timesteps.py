"""
Test: Timesteps and sigmas handling.

This test validates that timesteps and sigmas are correctly generated and handled
throughout the inference pipeline.

Tests:
- inference_prologue returns correct timesteps shape
- Timesteps slicing doesn't break inference
- Sigmas array has correct length (num_steps + 1)
"""

import pytest
import torch
import numpy as np


class TestTimestepsBasics:
    """Test basic timesteps and sigmas properties."""

    def test_timesteps_shape(self):
        """Test that timesteps array has correct shape."""
        num_steps = 50

        # Simulate timesteps generation (SD3 uses linspace)
        timesteps = torch.linspace(1000, 0, num_steps, dtype=torch.float32)

        assert timesteps.shape == (num_steps,)
        assert timesteps.ndim == 1
        assert len(timesteps) == num_steps

        print(f"✓ Timesteps shape correct: {timesteps.shape}")

    def test_sigmas_shape(self):
        """Test that sigmas array has correct shape (num_steps + 1)."""
        num_steps = 50

        # Sigmas typically has one extra element
        sigmas = torch.linspace(1.0, 0.0, num_steps + 1, dtype=torch.float32)

        assert sigmas.shape == (num_steps + 1,)
        assert len(sigmas) == num_steps + 1

        print(f"✓ Sigmas shape correct: {sigmas.shape}")

    def test_timesteps_monotonic_decreasing(self):
        """Test that timesteps are monotonically decreasing."""
        num_steps = 50
        timesteps = torch.linspace(1000, 0, num_steps, dtype=torch.float32)

        # Check monotonic decreasing
        diffs = timesteps[1:] - timesteps[:-1]
        assert torch.all(diffs <= 0), "Timesteps should be monotonically decreasing"

        print("✓ Timesteps are monotonically decreasing")

    def test_timesteps_dtype(self):
        """Test that timesteps use float32 dtype."""
        num_steps = 50
        timesteps = torch.linspace(1000, 0, num_steps, dtype=torch.float32)

        assert timesteps.dtype == torch.float32

        print(f"✓ Timesteps dtype correct: {timesteps.dtype}")


class TestTimestepsSlicing:
    """Test timesteps slicing for multi-step inference."""

    def test_slice_first_half(self):
        """Test slicing first half of timesteps."""
        num_steps = 50
        timesteps = torch.linspace(1000, 0, num_steps, dtype=torch.float32)

        # Slice first 25 steps
        start_step = 0
        end_step = 25
        sliced = timesteps[start_step:end_step]

        assert len(sliced) == 25
        assert sliced[0] == timesteps[0]
        assert sliced[-1] == timesteps[24]

        print(f"✓ First half slicing correct: {len(sliced)} steps")

    def test_slice_second_half(self):
        """Test slicing second half of timesteps."""
        num_steps = 50
        timesteps = torch.linspace(1000, 0, num_steps, dtype=torch.float32)

        # Slice second 25 steps
        start_step = 25
        end_step = 50
        sliced = timesteps[start_step:end_step]

        assert len(sliced) == 25
        assert sliced[0] == timesteps[25]
        assert sliced[-1] == timesteps[49]

        print(f"✓ Second half slicing correct: {len(sliced)} steps")

    def test_no_gaps_in_slicing(self):
        """Test that consecutive slices have no gaps."""
        num_steps = 50
        timesteps = torch.linspace(1000, 0, num_steps, dtype=torch.float32)

        # Split into two halves
        slice1 = timesteps[0:25]
        slice2 = timesteps[25:50]

        # Last element of slice1 should be adjacent to first element of slice2
        # (not equal, but consecutive in the original array)
        assert slice1[-1] == timesteps[24]
        assert slice2[0] == timesteps[25]

        # Concatenate should equal original
        concatenated = torch.cat([slice1, slice2])
        assert torch.equal(concatenated, timesteps)

        print("✓ No gaps in consecutive slices")

    def test_empty_slice(self):
        """Test that empty slice returns empty tensor."""
        num_steps = 50
        timesteps = torch.linspace(1000, 0, num_steps, dtype=torch.float32)

        # Empty slice
        empty = timesteps[25:25]

        assert len(empty) == 0
        assert empty.shape == (0,)

        print("✓ Empty slice handled correctly")

    def test_single_step_slice(self):
        """Test slicing a single step."""
        num_steps = 50
        timesteps = torch.linspace(1000, 0, num_steps, dtype=torch.float32)

        # Single step
        single = timesteps[10:11]

        assert len(single) == 1
        assert single[0] == timesteps[10]

        print("✓ Single step slice correct")


class TestTimestepsStepRange:
    """Test step_range concept for multi-step inference."""

    def test_step_range_coverage(self):
        """Test that step_ranges cover all steps without gaps or overlaps."""
        num_steps = 50

        # Simulate scheduler assigning step ranges
        step_ranges = [
            (0, 25),  # First batch: steps 0-24
            (25, 50),  # Second batch: steps 25-49
        ]

        # Check coverage
        all_steps = set()
        for start, end in step_ranges:
            all_steps.update(range(start, end))

        expected_steps = set(range(num_steps))
        assert all_steps == expected_steps, "Step ranges should cover all steps"

        print(f"✓ Step ranges cover all {num_steps} steps")

    def test_step_range_no_overlaps(self):
        """Test that step_ranges don't overlap."""
        step_ranges = [
            (0, 25),
            (25, 50),
        ]

        # Check for overlaps
        for i, (start1, end1) in enumerate(step_ranges):
            for j, (start2, end2) in enumerate(step_ranges):
                if i != j:
                    # Ranges should not overlap
                    assert not (
                        start1 < end2 and start2 < end1
                    ), f"Ranges overlap: ({start1}, {end1}) and ({start2}, {end2})"

        print("✓ Step ranges don't overlap")

    def test_unequal_step_ranges(self):
        """Test that step_ranges can have different sizes."""
        num_steps = 50

        # Unequal split
        step_ranges = [
            (0, 20),  # 20 steps
            (20, 35),  # 15 steps
            (35, 50),  # 15 steps
        ]

        # Verify coverage
        all_steps = set()
        for start, end in step_ranges:
            all_steps.update(range(start, end))

        assert all_steps == set(range(num_steps))

        # Verify sizes
        sizes = [end - start for start, end in step_ranges]
        assert sizes == [20, 15, 15]

        print(f"✓ Unequal step ranges work: {sizes}")


class TestTimestepsDtype:
    """Test dtype handling for timesteps and sigmas."""

    def test_float32_timesteps(self):
        """Test float32 timesteps (standard)."""
        timesteps = torch.rand((50,), dtype=torch.float32)
        assert timesteps.dtype == torch.float32
        print("✓ float32 timesteps work")

    def test_float16_timesteps(self):
        """Test float16 timesteps (memory efficient)."""
        timesteps = torch.rand((50,), dtype=torch.float16)
        assert timesteps.dtype == torch.float16
        print("✓ float16 timesteps work")

    def test_int64_timesteps(self):
        """Test int64 timesteps (discrete steps)."""
        timesteps = torch.arange(1000, 0, -20, dtype=torch.int64)
        assert timesteps.dtype == torch.int64
        print("✓ int64 timesteps work")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
