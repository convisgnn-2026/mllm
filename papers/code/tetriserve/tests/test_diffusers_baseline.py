"""
Test: Generate baseline reference images using pure diffusers SD3 pipeline.

This test creates reference images that will be used to validate TetriServe output.
We test multiple configurations to ensure comprehensive coverage.

Configurations:
- Resolutions: 512x512, 1024x1024, 2048x2048
- Fixed seed (42) for reproducibility
- Same prompts as used in TetriServe tests

Output: Reference images saved to tests/fixtures/diffusers_baseline/
"""

import os
import torch
import pytest
from pathlib import Path
from typing import List, Tuple

# Check if diffusers is available
try:
    from diffusers import StableDiffusion3Pipeline

    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False

# Skip all tests in this file if diffusers not available
pytestmark = pytest.mark.skipif(not DIFFUSERS_AVAILABLE, reason="diffusers not installed")


# Test configurations
SEED = 42
MODEL_ID = "stabilityai/stable-diffusion-3-medium-diffusers"
PROMPT = "a photo of an astronaut riding a horse on mars"
NUM_INFERENCE_STEPS = 28  # SD3 default
GUIDANCE_SCALE = 7.0

# Test resolutions: (height, width)
RESOLUTIONS = [
    (512, 512),
    (1024, 1024),
    (2048, 2048),
]

# Output directory for baseline images
BASELINE_DIR = Path(__file__).parent / "fixtures" / "diffusers_baseline"


def get_baseline_image_path(height: int, width: int, steps: int) -> Path:
    """Get the path for a baseline image file."""
    return BASELINE_DIR / f"sd3_baseline_{height}x{width}_steps{steps}_seed{SEED}.png"


@pytest.fixture(scope="module")
def sd3_pipeline():
    """Load SD3 pipeline once for all tests."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    print(f"\nLoading SD3 pipeline from {MODEL_ID}...")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
    )
    pipe = pipe.to("cuda")
    print("SD3 pipeline loaded successfully")

    yield pipe

    # Cleanup
    del pipe
    torch.cuda.empty_cache()


@pytest.mark.parametrize("height,width", RESOLUTIONS)
def test_generate_baseline_image(sd3_pipeline, height: int, width: int):
    """
    Generate a baseline reference image for a specific resolution.

    This test:
    1. Generates an image using pure diffusers SD3
    2. Saves it to tests/fixtures/diffusers_baseline/
    3. Can be used as reference for comparing TetriServe output
    """
    # Create output directory if it doesn't exist
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)

    output_path = get_baseline_image_path(height, width, NUM_INFERENCE_STEPS)

    # Skip if image already exists (to save time)
    if output_path.exists():
        print(f"\n✓ Baseline image already exists: {output_path}")
        assert output_path.stat().st_size > 0, "Existing baseline image is empty"
        return

    print(f"\n{'='*60}")
    print(f"Generating baseline: {height}x{width}, {NUM_INFERENCE_STEPS} steps")
    print(f"{'='*60}")

    # Generate image
    generator = torch.Generator(device="cuda").manual_seed(SEED)

    output = sd3_pipeline(
        prompt=PROMPT,
        height=height,
        width=width,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        generator=generator,
    )

    image = output.images[0]

    # Save image
    image.save(output_path)
    print(f"✓ Saved baseline image: {output_path}")
    print(f"  Image size: {image.size}")
    print(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")

    # Verify image was saved correctly
    assert output_path.exists(), "Failed to save baseline image"
    assert output_path.stat().st_size > 0, "Baseline image is empty"


def test_list_baseline_images():
    """List all generated baseline images for verification."""
    if not BASELINE_DIR.exists():
        pytest.skip("No baseline images generated yet")

    baseline_images = sorted(BASELINE_DIR.glob("*.png"))

    print(f"\n{'='*60}")
    print(f"Baseline Images Summary")
    print(f"{'='*60}")
    print(f"Directory: {BASELINE_DIR}")
    print(f"Total images: {len(baseline_images)}")
    print()

    for img_path in baseline_images:
        size_kb = img_path.stat().st_size / 1024
        print(f"  {img_path.name} ({size_kb:.1f} KB)")

    assert len(baseline_images) > 0, "No baseline images found"


@pytest.mark.parametrize("height,width", RESOLUTIONS)
def test_baseline_image_exists(height: int, width: int):
    """Verify that baseline images have been generated."""
    output_path = get_baseline_image_path(height, width, NUM_INFERENCE_STEPS)

    if not output_path.exists():
        pytest.skip(f"Baseline image not generated yet: {output_path.name}")

    # Verify file is not empty
    assert output_path.stat().st_size > 0, f"Baseline image is empty: {output_path}"
    print(f"✓ Baseline exists: {output_path.name} ({output_path.stat().st_size / 1024:.1f} KB)")


def test_verify_reproducibility(sd3_pipeline):
    """
    Verify that the same seed produces identical images.

    This test generates the same image twice and verifies they are identical.
    This ensures our baseline images are reproducible.
    """
    height, width = 512, 512  # Use smallest resolution for speed

    print(f"\nGenerating two images with same seed to verify reproducibility...")

    # Generate first image
    generator1 = torch.Generator(device="cuda").manual_seed(SEED)
    output1 = sd3_pipeline(
        prompt=PROMPT,
        height=height,
        width=width,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        generator=generator1,
    )

    # Generate second image with same seed
    generator2 = torch.Generator(device="cuda").manual_seed(SEED)
    output2 = sd3_pipeline(
        prompt=PROMPT,
        height=height,
        width=width,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        generator=generator2,
    )

    # Convert images to tensors for comparison
    import numpy as np

    img1_array = np.array(output1.images[0])
    img2_array = np.array(output2.images[0])

    # Check if arrays are identical
    assert img1_array.shape == img2_array.shape, "Image shapes don't match"

    # Calculate difference
    max_diff = np.abs(img1_array.astype(float) - img2_array.astype(float)).max()

    print(f"✓ Image shapes match: {img1_array.shape}")
    print(f"✓ Maximum pixel difference: {max_diff}")

    # Images should be identical (or very close due to floating point)
    assert max_diff < 1.0, f"Images differ too much (max diff: {max_diff})"
    print("✓ Reproducibility verified: Same seed produces identical images")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
