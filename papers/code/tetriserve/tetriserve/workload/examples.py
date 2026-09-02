#!/usr/bin/env python3
"""
Examples of using the clean workload package.

Run this file to see all examples in action.
"""

import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


def example_1_simple_workload():
    """Example 1: Create a simple workload with defaults."""
    print("\n" + "=" * 60)
    print("Example 1: Simple Workload")
    print("=" * 60)

    from tetriserve.workload import (
        WorkloadBuilder,
        PoissonProcess,
        FixedSLOPolicy,
    )

    # Create workload with builder pattern
    workload = (
        WorkloadBuilder()
        .with_model("sd3")
        .with_arrival_process(PoissonProcess(rate=1.0))  # 1 req/s
        .with_duration(10)  # 10 seconds
        .with_num_steps(20)
        .with_slo_policy(FixedSLOPolicy(5.0))  # 5 second SLO
        .build()
    )

    print(f"Created: {workload}")
    print(f"Requests: {len(workload)}")
    print(f"Model: {workload.get_model_name()}")
    print(f"Mean rate: {workload.mean_rate:.2f} req/s")
    print(f"CV: {workload.cv:.2f}")

    # Show first request
    if workload.requests:
        req = workload.requests[0]
        print(f"\nFirst request:")
        print(f"  Resolution: {req.height}x{req.width}")
        print(f"  Batch size: {req.batch_size}")
        print(f"  Steps: {req.num_inference_steps}")
        print(f"  SLO: {req.slo}s")


def example_2_custom_distributions():
    """Example 2: Custom shape and batch distributions."""
    print("\n" + "=" * 60)
    print("Example 2: Custom Distributions")
    print("=" * 60)

    from tetriserve.workload import (
        WorkloadBuilder,
        PoissonProcess,
        ShapeDistribution,
        BatchSizeDistribution,
    )

    # Use predefined distributions
    workload = (
        WorkloadBuilder()
        .with_model("flux")
        .with_arrival_process(PoissonProcess(rate=0.5))
        .with_duration(20)
        .with_shape_distribution(ShapeDistribution.long_seqlen_high())  # Favor large images
        .with_batch_distribution(BatchSizeDistribution.single_batch(1))
        .build()
    )

    print(f"Created: {workload}")

    # Count shapes
    from collections import Counter

    shapes = Counter((req.height, req.width) for req in workload.requests)
    print(f"\nShape distribution:")
    for shape, count in sorted(shapes.items(), reverse=True):
        pct = 100.0 * count / len(workload)
        print(f"  {shape[0]}x{shape[1]}: {count} ({pct:.1f}%)")


def example_3_gamma_process():
    """Example 3: Gamma process with high CV (bursty arrivals)."""
    print("\n" + "=" * 60)
    print("Example 3: Bursty Arrivals (Gamma Process)")
    print("=" * 60)

    from tetriserve.workload import (
        WorkloadBuilder,
        GammaProcess,
        ShapeDistribution,
    )

    # High CV = more bursty
    workload = (
        WorkloadBuilder()
        .with_model("sd3")
        .with_arrival_process(GammaProcess(rate=1.0, cv=2.0))  # CV=2 is very bursty
        .with_duration(30)
        .with_shape_distribution(ShapeDistribution.evenly())
        .build()
    )

    print(f"Created: {workload}")
    print(f"CV: {workload.cv:.2f} (high CV = bursty)")

    # Show arrival intervals
    if len(workload.arrival_times) > 1:
        intervals = workload.arrival_times[1:] - workload.arrival_times[:-1]
        print(f"\nInter-arrival times:")
        print(f"  Mean: {intervals.mean():.3f}s")
        print(f"  Std: {intervals.std():.3f}s")
        print(f"  Min: {intervals.min():.3f}s")
        print(f"  Max: {intervals.max():.3f}s")


def example_4_baseline_slo():
    """Example 4: SLO based on performance model."""
    print("\n" + "=" * 60)
    print("Example 4: Baseline SLO from Performance Model")
    print("=" * 60)

    from tetriserve.workload import (
        WorkloadBuilder,
        PoissonProcess,
        BaselineSLOPolicy,
    )

    try:
        # Load baseline from scaling efficiency data
        slo_policy = BaselineSLOPolicy.from_file(
            model_name="sd3",
            steps=20,
            gpu_count=1,
            multiplier=1.5,  # 1.5x baseline = 50% slack
        )

        workload = (
            WorkloadBuilder()
            .with_model("sd3")
            .with_arrival_process(PoissonProcess(rate=1.0))
            .with_duration(20)
            .with_num_steps(20)
            .with_slo_policy(slo_policy)
            .build()
        )

        print(f"Created: {workload}")

        # Show SLOs for different shapes
        print(f"\nSLOs per shape (1.5x baseline):")
        shapes_seen = set()
        for req in workload.requests:
            shape = (req.height, req.width)
            if shape not in shapes_seen:
                print(f"  {shape[0]}x{shape[1]}: {req.slo:.2f}s")
                shapes_seen.add(shape)

    except FileNotFoundError as e:
        print(f"⚠ Baseline data not found: {e}")
        print("  (Run scaling efficiency benchmark first)")


def example_5_custom_slo():
    """Example 5: Custom SLO from configuration file."""
    print("\n" + "=" * 60)
    print("Example 5: Custom SLO Policy")
    print("=" * 60)

    from tetriserve.workload import (
        WorkloadBuilder,
        PoissonProcess,
        load_slo_policy,
    )

    try:
        # Load SLO policy from config file
        slo_policy = load_slo_policy(
            config_name="custom_1_1.0x_slo",
            model_name="sd3",
            steps=20,
        )

        workload = (
            WorkloadBuilder()
            .with_model("sd3")
            .with_arrival_process(PoissonProcess(rate=1.0))
            .with_duration(15)
            .with_num_steps(20)
            .with_slo_policy(slo_policy)
            .build()
        )

        print(f"Created: {workload}")
        print(f"SLO policy loaded from config file")

    except FileNotFoundError as e:
        print(f"⚠ SLO config not found: {e}")


def example_6_num_requests():
    """Example 6: Generate exact number of requests."""
    print("\n" + "=" * 60)
    print("Example 6: Exact Number of Requests")
    print("=" * 60)

    from tetriserve.workload import (
        WorkloadBuilder,
        PoissonProcess,
        ShapeDistribution,
        FixedSLOPolicy,
    )

    # Generate exactly 50 requests (instead of time-based)
    workload = (
        WorkloadBuilder()
        .with_model("flux")
        .with_arrival_process(PoissonProcess(rate=2.0))
        .with_num_requests(50)  # Exactly 50 requests
        .with_shape_distribution(ShapeDistribution.evenly())
        .with_slo_policy(FixedSLOPolicy(8.0))
        .build()
    )

    print(f"Created: {workload}")
    print(f"Requested: 50 requests")
    print(f"Generated: {len(workload)} requests")
    print(f"Rate: {workload.mean_rate:.2f} req/s")

    # Show duration was calculated
    duration = workload.arrival_times[-1] - workload.arrival_times[0]
    print(f"Total duration: {duration:.2f}s")
    print(f"\n✓ Use with_num_requests() when you need exact count!")


def example_7_statistics():
    """Example 7: Compute statistics (after execution)."""
    print("\n" + "=" * 60)
    print("Example 7: Compute Statistics")
    print("=" * 60)

    from tetriserve.workload import (
        create_simple_workload,
        compute_statistics,
        print_statistics,
    )
    import numpy as np

    # Create workload
    workload = create_simple_workload(model_name="sd3", rate=1.0, duration=10, num_steps=20, seed=42)

    # Simulate execution (normally done by runtime)
    # For demo, just set fake completion times
    for i, req in enumerate(workload.requests):
        req.submit_time = workload.arrival_times[i]
        # Fake latency: 2-4 seconds
        req.completion_time = req.submit_time + np.random.uniform(2.0, 4.0)

    # Compute statistics
    stats = compute_statistics(workload, warmup_seconds=0)  # No warmup for demo

    # Print statistics
    print_statistics(stats)


def example_8_no_caching():
    """Example 8: Demonstrate no caching behavior."""
    print("\n" + "=" * 60)
    print("Example 8: No Hidden Caching")
    print("=" * 60)

    from tetriserve.workload import (
        WorkloadBuilder,
        PoissonProcess,
        BaselineSLOPolicy,
    )

    print("Creating two workloads with same config...")

    try:
        # Create first workload
        policy1 = BaselineSLOPolicy.from_file("sd3", steps=20, multiplier=1.0)
        workload1 = (
            WorkloadBuilder()
            .with_model("sd3")
            .with_arrival_process(PoissonProcess(rate=1.0))
            .with_duration(10)
            .with_slo_policy(policy1)
            .build()
        )

        # Create second workload (loads data again, no caching!)
        policy2 = BaselineSLOPolicy.from_file("sd3", steps=20, multiplier=1.0)
        workload2 = (
            WorkloadBuilder()
            .with_model("sd3")
            .with_arrival_process(PoissonProcess(rate=1.0))
            .with_duration(10)
            .with_slo_policy(policy2)
            .build()
        )

        print(f"✓ Both workloads created successfully")
        print(f"  Workload 1: {len(workload1)} requests")
        print(f"  Workload 2: {len(workload2)} requests")
        print(f"\n✓ No hidden global state!")
        print(f"✓ No caching!")
        print(f"✓ Each workload is independent!")

    except FileNotFoundError:
        print("⚠ Baseline data not available (this is OK for demo)")


def example_9_validation():
    """Example 9: Configuration validation."""
    print("\n" + "=" * 60)
    print("Example 9: Validation Catches Errors")
    print("=" * 60)

    from tetriserve.workload import (
        WorkloadBuilder,
        PoissonProcess,
        ShapeDistribution,
    )

    print("Testing validation...")

    # Test 1: Negative rate
    try:
        process = PoissonProcess(rate=-1.0)
        print("✗ Should have caught negative rate!")
    except ValueError as e:
        print(f"✓ Caught invalid rate: {e}")

    # Test 2: Invalid probabilities
    try:
        dist = ShapeDistribution(
            shapes=((256, 256), (512, 512)),
            probabilities=(0.5, 0.3),  # Doesn't sum to 1.0!
        )
        print("✗ Should have caught invalid probabilities!")
    except ValueError as e:
        print(f"✓ Caught invalid probabilities: {e}")

    # Test 3: Missing required parameter
    try:
        workload = (
            WorkloadBuilder()
            .with_model("sd3")
            # Missing arrival_process!
            .with_duration(10)
            .build()
        )
        print("✗ Should have caught missing parameter!")
    except ValueError as e:
        print(f"✓ Caught missing parameter: {e}")

    print("\n✓ Validation works correctly!")


def example_10_visualization():
    """Example 10: Visualize workload characteristics."""
    print("\n" + "=" * 60)
    print("Example 10: Workload Visualization")
    print("=" * 60)

    from tetriserve.workload import (
        WorkloadBuilder,
        GammaProcess,
        ShapeDistribution,
        FixedSLOPolicy,
        print_workload_summary,
    )

    try:
        from tetriserve.workload import (
            plot_arrival_pattern,
            plot_request_shapes,
            plot_burstiness_analysis,
        )

        # Create bursty workload for visualization
        workload = (
            WorkloadBuilder()
            .with_model("flux")
            .with_arrival_process(GammaProcess(rate=1.0, cv=1.5))  # Bursty!
            .with_duration(30)
            .with_shape_distribution(ShapeDistribution.evenly())
            .with_slo_policy(FixedSLOPolicy(8.0))
            .build()
        )

        print(f"Created workload: {len(workload)} requests")

        # Print text summary
        print("\nText Summary:")
        print_workload_summary(workload)

        print("\n✓ Visualization functions available!")
        print("  - plot_arrival_pattern(): Shows burstiness")
        print("  - plot_request_shapes(): Shows shape distribution")
        print("  - plot_burstiness_analysis(): Detailed analysis")
        print("  - plot_all(): Creates all plots")
        print("\nTo see plots, run:")
        print("  python tetriserve/workload/visualize_example.py")
        print("  or")
        print("  python tetriserve/workload/quick_viz.py")

    except ImportError:
        print("⚠ Matplotlib not installed (visualization requires matplotlib)")
        print("  Install with: pip install matplotlib scipy")
        print("  Or just use the workload without visualization!")


def main():
    """Run all examples."""
    print("=" * 60)
    print("Clean Workload Package - Examples")
    print("=" * 60)
    print("\nThese examples demonstrate the new clean API.")
    print("No global state, no caching, just simple code!")

    example_1_simple_workload()
    example_2_custom_distributions()
    example_3_gamma_process()
    example_4_baseline_slo()
    example_5_custom_slo()
    example_6_num_requests()
    example_7_statistics()
    example_8_no_caching()
    example_9_validation()
    example_10_visualization()

    print("\n" + "=" * 60)
    print("All Examples Complete!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("  ✓ No global state")
    print("  ✓ No hidden caching")
    print("  ✓ Everything is explicit")
    print("  ✓ Easy to understand")
    print("  ✓ Easy to test")
    print("  ✓ Type-safe")
    print("  ✓ Visualization included!")
    print("\nNext Steps:")
    print("  • Check out README.md for more details")
    print("  • Run visualize_example.py to see plots")
    print("  • Run quick_viz.py for interactive demos")


if __name__ == "__main__":
    main()
