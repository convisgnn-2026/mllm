# baseline vs dp exhaustive (medium)

uv run --no-sync python benchmark/serving_benchmark/json_merger.py -i benchmark/serving_benchmark/results/benchmark_flux_H100_tetris_medium/benchmark_results.json benchmark/serving_benchmark/results/benchmark_flux_H100_baseline_medium/benchmark_results.json -o benchmark/serving_benchmark/results/benchmark_flux_H100_tetris_medium_vs_baseline_medium_merged.json

# Post-process baseline_medium results
uv run --no-sync python benchmark/serving_benchmark/analyze_slo_from_config.py \
    benchmark/serving_benchmark/results/benchmark_flux_H100_baseline_medium/benchmark_results.json \
    -o benchmark/serving_benchmark/results/benchmark_flux_H100_baseline_medium/benchmark_results_all_slos.json \
    --slo-configs custom_1_1.0x_slo_H100_flux \
                  custom_1_1.1x_slo_H100_flux \
                  custom_1_1.2x_slo_H100_flux \
                  custom_1_1.3x_slo_H100_flux \
                  custom_1_1.4x_slo_H100_flux \
                  custom_1_1.5x_slo_H100_flux

# plot the results for baseline medium
uv run --no-sync python -m benchmark.serving_benchmark.plot_exp.main \
    benchmark/serving_benchmark/results/benchmark_flux_H100_baseline_medium/benchmark_results_all_slos.json \
    --output-dir benchmark/serving_benchmark/results/benchmark_flux_H100_baseline_medium/plots \
    --dpi 600 \
    --no-reference-line \
    --plot-types all \
    --figsize 14 10
