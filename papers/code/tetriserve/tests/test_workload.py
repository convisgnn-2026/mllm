from tetriserve.workload import (
    WorkloadBuilder,
    PoissonProcess,
    GammaProcess,
    ShapeDistribution,
    load_slo_policy,
    print_workload_summary,
    plot_all,
)
from tetriserve.workload.distribution import BatchSizeDistribution

for distribution, name in [
    (ShapeDistribution.evenly(), "evenly"),
    (ShapeDistribution.short_seqlen_high(), "short_seqlen_high"),
    (ShapeDistribution.long_seqlen_high(), "long_seqlen_high"),
]:
    workload = (
        WorkloadBuilder()
        .with_model("flux")
        .with_arrival_process(PoissonProcess(rate=1.0))
        .with_num_requests(300)
        .with_shape_distribution(distribution)
        .with_batch_distribution(BatchSizeDistribution.single_batch(1))
        .with_slo_policy(
            load_slo_policy(
                "custom_1_1.0x_slo_H100_flux",
                "flux",
                steps=20,
                config_dir="benchmark/serving_benchmark/slo_configs",
                baseline_dir="log/benchmark/scaling_efficiency",
            )
        )
        .with_num_steps(20)
        .build()
    )
    print_workload_summary(workload)
    plot_all(workload, output_path=f"test_workload_{name}.png", show=False)
