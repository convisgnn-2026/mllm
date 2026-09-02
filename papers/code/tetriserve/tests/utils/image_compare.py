"""
Image comparison utilities for validating TetriServe output against diffusers baseline.

Provides PSNR and SSIM metrics for comparing generated images.

Thresholds for "identical" images:
- PSNR > 30 dB (good quality match)
- SSIM > 0.95 (high structural similarity)
"""

import numpy as np
import torch
from typing import Union, Tuple, Optional
from pathlib import Path

try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


def calculate_psnr(img1: np.ndarray, img2: np.ndarray, max_value: float = 255.0) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR) between two images.

    PSNR measures the ratio between the maximum possible signal power and
    the power of corrupting noise. Higher is better.

    Args:
        img1: First image as numpy array
        img2: Second image as numpy array (must have same shape as img1)
        max_value: Maximum possible pixel value (255 for uint8, 1.0 for float)

    Returns:
        PSNR value in dB. Higher is better.
        - PSNR > 40 dB: Excellent quality (nearly identical)
        - PSNR > 30 dB: Good quality (acceptable match)
        - PSNR > 20 dB: Acceptable quality
        - PSNR < 20 dB: Poor quality

    Raises:
        ValueError: If images have different shapes
    """
    if img1.shape != img2.shape:
        raise ValueError(f"Images must have same shape. Got {img1.shape} and {img2.shape}")

    # Convert to float for accurate computation
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    # Calculate MSE (Mean Squared Error)
    mse = np.mean((img1 - img2) ** 2)

    # Handle perfect match (avoid division by zero)
    if mse == 0:
        return float("inf")

    # Calculate PSNR
    psnr = 20 * np.log10(max_value / np.sqrt(mse))
    return float(psnr)


def calculate_ssim(
    img1: np.ndarray,
    img2: np.ndarray,
    max_value: float = 255.0,
    window_size: int = 11,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    """
    Calculate Structural Similarity Index (SSIM) between two images.

    SSIM measures the perceived quality difference between two images.
    It considers luminance, contrast, and structure. Range: [-1, 1], higher is better.

    Args:
        img1: First image as numpy array
        img2: Second image as numpy array (must have same shape as img1)
        max_value: Maximum possible pixel value (255 for uint8, 1.0 for float)
        window_size: Size of Gaussian window (must be odd)
        k1: Constant for luminance stability (default: 0.01)
        k2: Constant for contrast stability (default: 0.03)

    Returns:
        SSIM value in range [-1, 1]. Higher is better.
        - SSIM > 0.99: Nearly identical
        - SSIM > 0.95: Very similar (our threshold)
        - SSIM > 0.90: Similar
        - SSIM < 0.90: Different

    Raises:
        ValueError: If images have different shapes
    """
    if img1.shape != img2.shape:
        raise ValueError(f"Images must have same shape. Got {img1.shape} and {img2.shape}")

    # Convert to float
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    # Handle multi-channel images
    if len(img1.shape) == 3:
        # Calculate SSIM for each channel and average
        ssim_channels = []
        for i in range(img1.shape[2]):
            ssim_ch = _calculate_ssim_single_channel(img1[:, :, i], img2[:, :, i], max_value, window_size, k1, k2)
            ssim_channels.append(ssim_ch)
        return float(np.mean(ssim_channels))
    else:
        return float(_calculate_ssim_single_channel(img1, img2, max_value, window_size, k1, k2))


def _calculate_ssim_single_channel(
    img1: np.ndarray, img2: np.ndarray, max_value: float, window_size: int, k1: float, k2: float
) -> float:
    """Calculate SSIM for a single channel (grayscale) image."""
    from scipy.ndimage import uniform_filter

    c1 = (k1 * max_value) ** 2
    c2 = (k2 * max_value) ** 2

    # Compute means
    mu1 = uniform_filter(img1, size=window_size)
    mu2 = uniform_filter(img2, size=window_size)

    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2

    # Compute variances and covariance
    sigma1_sq = uniform_filter(img1**2, size=window_size) - mu1_sq
    sigma2_sq = uniform_filter(img2**2, size=window_size) - mu2_sq
    sigma12 = uniform_filter(img1 * img2, size=window_size) - mu1_mu2

    # SSIM formula
    numerator = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
    denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)

    ssim_map = numerator / denominator
    return float(np.mean(ssim_map))


def compare_images(
    img1: Union[np.ndarray, Image.Image, str, Path],
    img2: Union[np.ndarray, Image.Image, str, Path],
    threshold_psnr: float = 30.0,
    threshold_ssim: float = 0.95,
) -> Tuple[bool, dict]:
    """
    Compare two images and determine if they match based on PSNR and SSIM thresholds.

    Threshold guidelines (empirically determined from TetriServe testing):
      - SP=1 vs diffusers baseline: PSNR>=25, SSIM>=0.90
        (flash-attn vs PyTorch attention causes minor numerical differences)
      - SP=2 vs SP=1: PSNR>=20, SSIM>=0.85
        (bfloat16 all-to-all adds per-layer noise over 20+ steps)
      - SP=4 vs SP=1: PSNR>=18, SSIM>=0.80
        (more all-to-all rounds amplify numerical noise)
      - Multistep (latent transfer) vs single-round: PSNR>=18, SSIM>=0.80
        (latent serialization adds additional numerical variance)

    Args:
        img1: First image (numpy array, PIL Image, or file path)
        img2: Second image (numpy array, PIL Image, or file path)
        threshold_psnr: Minimum PSNR for match (default: 30.0 dB)
        threshold_ssim: Minimum SSIM for match (default: 0.95)

    Returns:
        Tuple of (match: bool, metrics: dict)
        - match: True if both PSNR and SSIM meet thresholds
        - metrics: Dictionary with 'psnr', 'ssim', 'passed_psnr', 'passed_ssim'
    """
    # Convert inputs to numpy arrays
    arr1 = _to_numpy(img1)
    arr2 = _to_numpy(img2)

    # Ensure same shape
    if arr1.shape != arr2.shape:
        raise ValueError(f"Images have different shapes: {arr1.shape} vs {arr2.shape}")

    # Calculate metrics
    psnr = calculate_psnr(arr1, arr2, max_value=255.0)
    ssim = calculate_ssim(arr1, arr2, max_value=255.0)

    # Check thresholds
    passed_psnr = psnr >= threshold_psnr
    passed_ssim = ssim >= threshold_ssim
    match = passed_psnr and passed_ssim

    metrics = {
        "psnr": psnr,
        "ssim": ssim,
        "threshold_psnr": threshold_psnr,
        "threshold_ssim": threshold_ssim,
        "passed_psnr": passed_psnr,
        "passed_ssim": passed_ssim,
    }

    return match, metrics


def _to_numpy(img: Union[np.ndarray, Image.Image, str, Path]) -> np.ndarray:
    """Convert various image formats to numpy array."""
    if isinstance(img, np.ndarray):
        return img

    if not PIL_AVAILABLE:
        raise ImportError("PIL is required for loading images from files")

    if isinstance(img, Image.Image):
        return np.array(img)

    if isinstance(img, (str, Path)):
        pil_img = Image.open(img)
        return np.array(pil_img)

    raise TypeError(f"Unsupported image type: {type(img)}")


def print_comparison_results(metrics: dict, label: str = "Image Comparison") -> None:
    """
    Print formatted comparison results.

    Args:
        metrics: Dictionary returned by compare_images()
        label: Label for the comparison
    """
    print(f"\n{'='*60}")
    print(f"{label}")
    print(f"{'='*60}")
    print(f"PSNR:  {metrics['psnr']:.2f} dB  (threshold: {metrics['threshold_psnr']:.2f} dB)")
    print(f"       {'✓ PASS' if metrics['passed_psnr'] else '✗ FAIL'}")
    print(f"SSIM:  {metrics['ssim']:.4f}     (threshold: {metrics['threshold_ssim']:.4f})")
    print(f"       {'✓ PASS' if metrics['passed_ssim'] else '✗ FAIL'}")
    print(f"{'='*60}")

    if metrics["passed_psnr"] and metrics["passed_ssim"]:
        print("✓ Overall: PASS - Images match within thresholds")
    else:
        print("✗ Overall: FAIL - Images do not match")
    print()


def assert_images_match(
    img1: Union[np.ndarray, Image.Image, str, Path],
    img2: Union[np.ndarray, Image.Image, str, Path],
    threshold_psnr: float = 30.0,
    threshold_ssim: float = 0.95,
    label: str = "Image Comparison",
) -> dict:
    """
    Assert that two images match based on PSNR and SSIM thresholds.

    Raises AssertionError if images don't match.
    Returns metrics dictionary if images match.

    Args:
        img1: First image
        img2: Second image
        threshold_psnr: Minimum PSNR for match (default: 30.0 dB)
        threshold_ssim: Minimum SSIM for match (default: 0.95)
        label: Label for error message

    Returns:
        metrics: Dictionary with comparison metrics

    Raises:
        AssertionError: If images don't match thresholds
    """
    match, metrics = compare_images(img1, img2, threshold_psnr, threshold_ssim)

    print_comparison_results(metrics, label)

    if not match:
        error_msgs = []
        if not metrics["passed_psnr"]:
            error_msgs.append(f"PSNR {metrics['psnr']:.2f} < {threshold_psnr:.2f}")
        if not metrics["passed_ssim"]:
            error_msgs.append(f"SSIM {metrics['ssim']:.4f} < {threshold_ssim:.4f}")

        raise AssertionError(f"{label}: Images do not match - {', '.join(error_msgs)}")

    return metrics


if __name__ == "__main__":
    # Simple test
    print("Testing image comparison utilities...")

    # Test with identical images
    img1 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    img2 = img1.copy()

    psnr = calculate_psnr(img1, img2)
    ssim = calculate_ssim(img1, img2)

    print(f"Identical images - PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}")
    assert psnr == float("inf"), "PSNR should be infinite for identical images"
    assert ssim > 0.99, "SSIM should be > 0.99 for identical images"

    # Test with slightly different images
    img3 = img1.copy()
    noise = np.random.randint(-5, 5, img3.shape, dtype=np.int16)
    img3 = np.clip(img3.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    psnr = calculate_psnr(img1, img3)
    ssim = calculate_ssim(img1, img3)

    print(f"Slightly different images - PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}")

    print("\n✓ All tests passed!")
