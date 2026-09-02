"""
Test: LatentManager save/retrieve on single rank.

This test validates that LatentManager can correctly save and retrieve tensors
on the same rank without distributed communication.

Tests:
- save_latent() stores tensors correctly
- LatentFuture.get() retrieves tensors from same rank
- Tensor values match (not just shapes)
"""

import pytest
import torch
import multiprocessing as mp
from tetriserve.light_ray.latent_manager import LatentManager
from tetriserve.light_ray.latent_future import LatentFuture, TensorInfo


@pytest.fixture
def sample_tensors():
    """Create sample tensors for testing."""
    return {
        "latents": torch.randn((1, 4, 64, 64), dtype=torch.float16),
        "prompt_embeds": torch.randn((2, 333, 4096), dtype=torch.float16),
        "pooled_prompt_embeds": torch.randn((2, 2048), dtype=torch.float16),
        "timesteps": torch.rand((50,), dtype=torch.float32),
        "sigmas": torch.rand((51,), dtype=torch.float32),
    }


class TestLatentManagerSingleRank:
    """Test LatentManager operations on a single rank (no distributed communication)."""

    def test_save_latent_stores_tensors(self, sample_tensors):
        """Test that save_latent() stores tensors in latent_store."""
        # Create a single-rank LatentManager
        request_queues = [mp.Queue()]
        manager = LatentManager(rank=0, request_queues=request_queues)

        sub_req_id = 123

        # Save tensors
        future = manager.save_latent(
            sub_req_id=sub_req_id,
            tensors=sample_tensors,
        )

        # Verify LatentFuture was returned
        assert isinstance(future, LatentFuture)
        assert future.sub_req_id == sub_req_id
        assert future.src_rank == 0

        # Verify tensors were stored
        assert sub_req_id in manager.latent_store
        stored_tensors = manager.latent_store[sub_req_id]["tensors"]

        # Verify all tensor names are present
        for tensor_name in sample_tensors.keys():
            assert tensor_name in stored_tensors, f"Missing tensor: {tensor_name}"

        # Cleanup
        manager.stop_sender_thread()
        print("✓ save_latent() stores tensors correctly")

    def test_get_local_tensor_retrieves_tensors(self, sample_tensors):
        """Test that get_local_tensor() retrieves stored tensors."""
        request_queues = [mp.Queue()]
        manager = LatentManager(rank=0, request_queues=request_queues)

        sub_req_id = 456

        # Save tensors
        manager.save_latent(sub_req_id=sub_req_id, tensors=sample_tensors)

        # Retrieve tensors
        retrieved = manager.get_local_tensor(sub_req_id)

        # Verify retrieval
        assert retrieved is not None
        assert isinstance(retrieved, dict)
        assert len(retrieved) == len(sample_tensors)

        # Verify all tensors present
        for tensor_name in sample_tensors.keys():
            assert tensor_name in retrieved

        # Cleanup
        manager.stop_sender_thread()
        print("✓ get_local_tensor() retrieves tensors correctly")

    def test_tensor_values_match(self, sample_tensors):
        """Test that retrieved tensor values match original values."""
        request_queues = [mp.Queue()]
        manager = LatentManager(rank=0, request_queues=request_queues)

        sub_req_id = 789

        # Save tensors
        manager.save_latent(sub_req_id=sub_req_id, tensors=sample_tensors)

        # Retrieve tensors
        retrieved = manager.get_local_tensor(sub_req_id)

        # Verify values match
        for tensor_name, original_tensor in sample_tensors.items():
            retrieved_tensor = retrieved[tensor_name]

            # Check shape
            assert (
                retrieved_tensor.shape == original_tensor.shape
            ), f"{tensor_name}: shape mismatch {retrieved_tensor.shape} != {original_tensor.shape}"

            # Check dtype
            assert (
                retrieved_tensor.dtype == original_tensor.dtype
            ), f"{tensor_name}: dtype mismatch {retrieved_tensor.dtype} != {original_tensor.dtype}"

            # Check values (should be identical since no transfer)
            assert torch.equal(retrieved_tensor, original_tensor), f"{tensor_name}: values don't match"

        # Cleanup
        manager.stop_sender_thread()
        print("✓ Retrieved tensor values match original values")

    def test_latent_future_metadata(self, sample_tensors):
        """Test that LatentFuture contains correct metadata."""
        request_queues = [mp.Queue()]
        manager = LatentManager(rank=0, request_queues=request_queues)

        sub_req_id = 101

        # Save tensors
        future = manager.save_latent(sub_req_id=sub_req_id, tensors=sample_tensors)

        # Verify metadata
        assert future.sub_req_id == sub_req_id
        assert future.src_rank == 0
        assert isinstance(future.tensors_info, dict)

        # Verify tensors_info contains correct information
        for tensor_name, tensor in sample_tensors.items():
            assert tensor_name in future.tensors_info
            info = future.tensors_info[tensor_name]

            assert isinstance(info, TensorInfo)
            assert tuple(info.shape) == tuple(tensor.shape)
            assert info.dtype == tensor.dtype

        # Cleanup
        manager.stop_sender_thread()
        print("✓ LatentFuture metadata is correct")

    def test_multiple_saves_different_ids(self, sample_tensors):
        """Test saving multiple tensors with different sub_req_ids."""
        request_queues = [mp.Queue()]
        manager = LatentManager(rank=0, request_queues=request_queues)

        num_saves = 5
        sub_req_ids = list(range(100, 100 + num_saves))

        # Save multiple tensors
        for sub_req_id in sub_req_ids:
            manager.save_latent(sub_req_id=sub_req_id, tensors=sample_tensors)

        # Verify all are stored
        for sub_req_id in sub_req_ids:
            assert sub_req_id in manager.latent_store
            retrieved = manager.get_local_tensor(sub_req_id)
            assert retrieved is not None
            assert len(retrieved) == len(sample_tensors)

        # Cleanup
        manager.stop_sender_thread()
        print(f"✓ Multiple saves work correctly ({num_saves} saves)")

    def test_get_nonexistent_tensor(self):
        """Test that get_local_tensor() returns None for non-existent sub_req_id."""
        request_queues = [mp.Queue()]
        manager = LatentManager(rank=0, request_queues=request_queues)

        # Try to get non-existent tensor
        retrieved = manager.get_local_tensor(sub_req_id=999)

        assert retrieved is None

        # Cleanup
        manager.stop_sender_thread()
        print("✓ get_local_tensor() returns None for non-existent ID")

    def test_save_without_tensors(self):
        """Test save_latent() with save_tensors=False (metadata only)."""
        request_queues = [mp.Queue()]
        manager = LatentManager(rank=0, request_queues=request_queues)

        sub_req_id = 202
        tensors = {
            "latents": torch.randn((1, 4, 64, 64), dtype=torch.float16),
        }

        # Save without storing tensors
        future = manager.save_latent(
            sub_req_id=sub_req_id,
            tensors=tensors,
            save_tensors=False,  # Don't store tensors
        )

        # Verify LatentFuture was returned with metadata
        assert isinstance(future, LatentFuture)
        assert future.sub_req_id == sub_req_id
        assert "latents" in future.tensors_info

        # Verify tensors were NOT stored
        assert sub_req_id not in manager.latent_store

        # Cleanup
        manager.stop_sender_thread()
        print("✓ save_latent(save_tensors=False) works correctly")

    def test_tensor_shapes_and_dtypes(self):
        """Test various tensor shapes and dtypes."""
        request_queues = [mp.Queue()]
        manager = LatentManager(rank=0, request_queues=request_queues)

        test_cases = [
            # (shape, dtype)
            ((1, 4, 64, 64), torch.float16),
            ((2, 4, 128, 128), torch.float16),
            ((4, 4, 256, 256), torch.float32),
            ((1, 333, 4096), torch.float16),
            ((50,), torch.float32),
        ]

        for idx, (shape, dtype) in enumerate(test_cases):
            sub_req_id = 300 + idx
            tensors = {
                "test_tensor": torch.randn(shape, dtype=dtype),
            }

            # Save and retrieve
            manager.save_latent(sub_req_id=sub_req_id, tensors=tensors)
            retrieved = manager.get_local_tensor(sub_req_id)

            # Verify
            assert retrieved is not None
            if isinstance(retrieved, dict):
                assert "test_tensor" in retrieved
                tensor = retrieved["test_tensor"]
            else:
                assert isinstance(retrieved, torch.Tensor)
                tensor = retrieved
            assert tensor.shape == shape
            assert tensor.dtype == dtype

        # Cleanup
        manager.stop_sender_thread()
        print(f"✓ Various tensor shapes and dtypes work ({len(test_cases)} cases)")


class TestLatentManagerCPU:
    """Test LatentManager with CPU tensors (no CUDA required)."""

    def test_cpu_tensors(self):
        """Test save/retrieve with CPU tensors."""
        request_queues = [mp.Queue()]
        manager = LatentManager(rank=0, request_queues=request_queues)

        sub_req_id = 500
        tensors = {
            "latents": torch.randn((1, 4, 64, 64), dtype=torch.float16, device="cpu"),
            "timesteps": torch.rand((50,), dtype=torch.float32, device="cpu"),
        }

        # Save tensors
        future = manager.save_latent(sub_req_id=sub_req_id, tensors=tensors)

        # Retrieve tensors
        retrieved = manager.get_local_tensor(sub_req_id)

        # Verify
        assert retrieved is not None
        for tensor_name, original_tensor in tensors.items():
            retrieved_tensor = retrieved[tensor_name]
            assert retrieved_tensor.device == original_tensor.device
            assert torch.equal(retrieved_tensor, original_tensor)

        # Cleanup
        manager.stop_sender_thread()
        print("✓ CPU tensors work correctly")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
