"""
Test: LatentFuture metadata contract and serialization.

This test validates that LatentFuture properly stores and serializes metadata
about tensors without requiring actual tensor data.

Tests:
- Metadata contract (required fields)
- Tensor info structure (shape, dtype)
- Pickling/unpickling (for MP queue transfer)
"""

import pytest
import torch
import pickle
from tetriserve.light_ray.latent_future import LatentFuture, TensorInfo


class TestLatentFutureMetadata:
    """Test LatentFuture metadata contract."""

    @staticmethod
    def _info(shape, dtype):
        return TensorInfo(shape=shape, dtype=dtype, device="cpu")

    def test_latent_future_initialization(self):
        """Test that LatentFuture can be initialized with required fields."""
        sub_req_id = 123
        src_rank = 0
        tensors_info = {
            "latents": self._info((1, 4, 64, 64), torch.float16),
            "prompt_embeds": self._info((2, 333, 4096), torch.float16),
            "pooled_prompt_embeds": self._info((2, 2048), torch.float16),
            "timesteps": self._info((50,), torch.float32),
            "sigmas": self._info((51,), torch.float32),
        }

        future = LatentFuture(
            sub_req_id=sub_req_id,
            src_rank=src_rank,
            tensors_info=tensors_info,
        )

        # Verify fields
        assert future.sub_req_id == sub_req_id
        assert future.src_rank == src_rank
        assert future.tensors_info == tensors_info

        print("✓ LatentFuture initialized successfully")

    def test_tensors_info_structure(self):
        """Test that tensors_info contains correct shape and dtype information."""
        tensors_info = {
            "latents": self._info((2, 4, 128, 128), torch.float16),
            "prompt_embeds": self._info((4, 333, 4096), torch.float16),
        }

        future = LatentFuture(
            sub_req_id=456,
            src_rank=1,
            tensors_info=tensors_info,
        )

        # Verify latents info
        assert "latents" in future.tensors_info
        assert future.tensors_info["latents"].shape == (2, 4, 128, 128)
        assert future.tensors_info["latents"].dtype == torch.float16

        # Verify prompt_embeds info
        assert "prompt_embeds" in future.tensors_info
        assert future.tensors_info["prompt_embeds"].shape == (4, 333, 4096)
        assert future.tensors_info["prompt_embeds"].dtype == torch.float16

        print("✓ tensors_info structure validated")

    def test_all_required_tensor_names(self):
        """Test that all required tensor names are present in tensors_info."""
        required_tensors = [
            "latents",
            "prompt_embeds",
            "pooled_prompt_embeds",
            "timesteps",
            "sigmas",
        ]

        tensors_info = {
            "latents": self._info((1, 4, 64, 64), torch.float16),
            "prompt_embeds": self._info((2, 333, 4096), torch.float16),
            "pooled_prompt_embeds": self._info((2, 2048), torch.float16),
            "timesteps": self._info((50,), torch.float32),
            "sigmas": self._info((51,), torch.float32),
        }

        future = LatentFuture(
            sub_req_id=789,
            src_rank=2,
            tensors_info=tensors_info,
        )

        # Verify all required tensors are present
        for tensor_name in required_tensors:
            assert tensor_name in future.tensors_info, f"Missing required tensor: {tensor_name}"
            assert isinstance(future.tensors_info[tensor_name], TensorInfo)

        print(f"✓ All {len(required_tensors)} required tensors present")

    def test_shape_expectations(self):
        """Test that tensor shapes match expected dimensions."""
        tensors_info = {
            "latents": self._info((2, 4, 128, 128), torch.float16),  # (bs, 4, H/8, W/8)
            "prompt_embeds": self._info((4, 333, 4096), torch.float16),  # (bs*2, seq_len, hidden)
            "pooled_prompt_embeds": self._info((4, 2048), torch.float16),  # (bs*2, pooled_dim)
            "timesteps": self._info((50,), torch.float32),  # (num_steps,)
            "sigmas": self._info((51,), torch.float32),  # (num_steps+1,)
        }

        future = LatentFuture(
            sub_req_id=101,
            src_rank=0,
            tensors_info=tensors_info,
        )

        # Verify latents shape (4D: batch, channels, height, width)
        latents_shape = future.tensors_info["latents"].shape
        assert len(latents_shape) == 4, f"latents should be 4D, got {len(latents_shape)}D"
        assert latents_shape[1] == 4, f"latents should have 4 channels, got {latents_shape[1]}"

        # Verify prompt_embeds shape (3D: batch, seq_len, hidden_dim)
        prompt_shape = future.tensors_info["prompt_embeds"].shape
        assert len(prompt_shape) == 3, f"prompt_embeds should be 3D, got {len(prompt_shape)}D"

        # Verify pooled_prompt_embeds shape (2D: batch, pooled_dim)
        pooled_shape = future.tensors_info["pooled_prompt_embeds"].shape
        assert len(pooled_shape) == 2, f"pooled_prompt_embeds should be 2D, got {len(pooled_shape)}D"

        # Verify timesteps shape (1D)
        timesteps_shape = future.tensors_info["timesteps"].shape
        assert len(timesteps_shape) == 1, f"timesteps should be 1D, got {len(timesteps_shape)}D"

        # Verify sigmas shape (1D)
        sigmas_shape = future.tensors_info["sigmas"].shape
        assert len(sigmas_shape) == 1, f"sigmas should be 1D, got {len(sigmas_shape)}D"

        print("✓ All tensor shapes match expected dimensions")

    def test_dtype_expectations(self):
        """Test that tensor dtypes are appropriate."""
        tensors_info = {
            "latents": self._info((1, 4, 64, 64), torch.float16),
            "prompt_embeds": self._info((2, 333, 4096), torch.float16),
            "pooled_prompt_embeds": self._info((2, 2048), torch.float16),
            "timesteps": self._info((50,), torch.float32),
            "sigmas": self._info((51,), torch.float32),
        }

        future = LatentFuture(
            sub_req_id=202,
            src_rank=1,
            tensors_info=tensors_info,
        )

        # Verify latents, prompt_embeds, pooled_prompt_embeds use float16
        assert future.tensors_info["latents"].dtype == torch.float16
        assert future.tensors_info["prompt_embeds"].dtype == torch.float16
        assert future.tensors_info["pooled_prompt_embeds"].dtype == torch.float16

        # Verify timesteps and sigmas use float32
        assert future.tensors_info["timesteps"].dtype == torch.float32
        assert future.tensors_info["sigmas"].dtype == torch.float32

        print("✓ All tensor dtypes are appropriate")


class TestLatentFutureSerialization:
    """Test LatentFuture serialization for multiprocessing queue transfer."""

    def test_pickle_latent_future(self):
        """Test that LatentFuture can be pickled and unpickled."""
        original = LatentFuture(
            sub_req_id=999,
            src_rank=3,
            tensors_info={
                "latents": TensorInfo(shape=(1, 4, 64, 64), dtype=torch.float16, device="cpu"),
                "prompt_embeds": TensorInfo(shape=(2, 333, 4096), dtype=torch.float16, device="cpu"),
                "pooled_prompt_embeds": TensorInfo(shape=(2, 2048), dtype=torch.float16, device="cpu"),
                "timesteps": TensorInfo(shape=(50,), dtype=torch.float32, device="cpu"),
                "sigmas": TensorInfo(shape=(51,), dtype=torch.float32, device="cpu"),
            },
        )

        # Pickle
        pickled = pickle.dumps(original)

        # Unpickle
        restored = pickle.loads(pickled)

        # Verify fields match
        assert restored.sub_req_id == original.sub_req_id
        assert restored.src_rank == original.src_rank
        assert restored.tensors_info == original.tensors_info

        print("✓ LatentFuture can be pickled/unpickled")

    def test_pickle_preserves_dtype(self):
        """Test that pickling preserves torch.dtype objects correctly."""
        original = LatentFuture(
            sub_req_id=111,
            src_rank=0,
            tensors_info={
                "latents": TensorInfo(shape=(1, 4, 64, 64), dtype=torch.float16, device="cpu"),
                "timesteps": TensorInfo(shape=(50,), dtype=torch.float32, device="cpu"),
            },
        )

        # Pickle and unpickle
        restored = pickle.loads(pickle.dumps(original))

        # Verify dtypes are preserved correctly
        assert isinstance(restored.tensors_info["latents"].dtype, torch.dtype)
        assert restored.tensors_info["latents"].dtype == torch.float16
        assert isinstance(restored.tensors_info["timesteps"].dtype, torch.dtype)
        assert restored.tensors_info["timesteps"].dtype == torch.float32

        print("✓ Pickling preserves torch.dtype objects")

    def test_multiple_pickle_cycles(self):
        """Test that LatentFuture survives multiple pickle/unpickle cycles."""
        original = LatentFuture(
            sub_req_id=222,
            src_rank=1,
            tensors_info={
                "latents": TensorInfo(shape=(2, 4, 128, 128), dtype=torch.float16, device="cpu"),
                "prompt_embeds": TensorInfo(shape=(4, 333, 4096), dtype=torch.float16, device="cpu"),
            },
        )

        # Multiple cycles
        current = original
        for i in range(5):
            pickled = pickle.dumps(current)
            current = pickle.loads(pickled)

        # Verify still matches original
        assert current.sub_req_id == original.sub_req_id
        assert current.src_rank == original.src_rank
        assert current.tensors_info == original.tensors_info

        print("✓ LatentFuture survives 5 pickle/unpickle cycles")


class TestLatentFutureEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_tensors_info(self):
        """Test LatentFuture with empty tensors_info."""
        future = LatentFuture(
            sub_req_id=0,
            src_rank=0,
            tensors_info={},
        )

        assert future.tensors_info == {}
        print("✓ LatentFuture handles empty tensors_info")

    def test_single_tensor(self):
        """Test LatentFuture with only one tensor."""
        future = LatentFuture(
            sub_req_id=333,
            src_rank=2,
            tensors_info={
                "latents": TensorInfo(shape=(1, 4, 64, 64), dtype=torch.float16, device="cpu"),
            },
        )

        assert len(future.tensors_info) == 1
        assert "latents" in future.tensors_info
        print("✓ LatentFuture handles single tensor")

    def test_large_batch_size(self):
        """Test LatentFuture with large batch size."""
        batch_size = 16
        tensors_info = {
            "latents": TensorInfo(shape=(batch_size, 4, 256, 256), dtype=torch.float16, device="cpu"),
            "prompt_embeds": TensorInfo(shape=(batch_size * 2, 333, 4096), dtype=torch.float16, device="cpu"),
            "pooled_prompt_embeds": TensorInfo(shape=(batch_size * 2, 2048), dtype=torch.float16, device="cpu"),
            "timesteps": TensorInfo(shape=(50,), dtype=torch.float32, device="cpu"),
            "sigmas": TensorInfo(shape=(51,), dtype=torch.float32, device="cpu"),
        }

        future = LatentFuture(
            sub_req_id=444,
            src_rank=3,
            tensors_info=tensors_info,
        )

        assert future.tensors_info["latents"].shape[0] == batch_size
        assert future.tensors_info["prompt_embeds"].shape[0] == batch_size * 2
        print(f"✓ LatentFuture handles batch_size={batch_size}")

    def test_different_resolutions(self):
        """Test LatentFuture with different image resolutions."""
        resolutions = [
            (512, 512),
            (1024, 1024),
            (2048, 2048),
            (768, 1024),  # Non-square
        ]

        for height, width in resolutions:
            tensors_info = {
                "latents": TensorInfo(shape=(1, 4, height // 8, width // 8), dtype=torch.float16, device="cpu"),
            }

            future = LatentFuture(
                sub_req_id=555,
                src_rank=0,
                tensors_info=tensors_info,
            )

            expected_shape = (1, 4, height // 8, width // 8)
            assert future.tensors_info["latents"].shape == expected_shape

        print(f"✓ LatentFuture handles {len(resolutions)} different resolutions")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
