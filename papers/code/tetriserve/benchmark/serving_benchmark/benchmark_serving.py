#!/usr/bin/env python3
"""
FluidServe Serving Benchmark - Clean Implementation

Usage:
    python benchmark/serving_benchmark/benchmark_serving.py --config config.yaml
"""

import asyncio
import argparse
import json
import socket
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
import yaml
import sys
import os

sys.path.append(str(Path(__file__).parent.parent.parent))

from tetriserve.config.args import ServerArgs
from tetriserve.xfuser.config import (
    EngineConfig,
    ModelConfig,
    RuntimeConfig,
    ParallelConfig,
    SequenceParallelConfig,
    DataParallelConfig,
    PipeFusionParallelConfig,
    TensorParallelConfig,
    InputConfig,
)
from tetriserve.xfuser.config.config import FastAttnConfig
from tetriserve.engine.runtime import CServeRuntime
from tetriserve.utils import set_all_seeds
from tetriserve import SUPPORTED_MODEL_DICT
from tetriserve.xfuser.logger import init_logger

# Use new workload package
from tetriserve.workload import (
    WorkloadBuilder,
    PoissonProcess,
    GammaProcess,
    ShapeDistribution,
    BatchSizeDistribution,
    load_slo_policy,
    compute_statistics,
    get_per_request_info,
    load_workload,
)

logger = init_logger(__name__)


def get_free_port() -> int:
    """Get a free port by binding to port 0 and letting the OS assign one."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


def load_yaml_config(config_file: str) -> Dict[str, Any]:
    """Load and normalize YAML configuration."""
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    # Convert single values to lists
    list_params = [
        "models",
        "schedule_logics",
        "gpu_nums",
        "distributions",
        "rates",
        "cvs",
        "steps",
        "shape_dists",
        "output_types",
        "slo_configs",
    ]
    for param in list_params:
        if param in config and not isinstance(config[param], list):
            config[param] = [config[param]]

    return config


def create_runtime(
    model_name: str,
    schedule_logic: str,
    port: int,
    gpu_num: int,
    fake_engine: bool,
    perf_model_dir: str,
    enable_tracer: bool = False,
    log_requests: bool = False,
    skip_latent_transfer: bool = False,
) -> CServeRuntime:
    """Create runtime with specified configuration."""
    server_args = ServerArgs(
        nnodes=1,
        nproc_per_node=gpu_num,
        master_addr="127.0.0.1",
        master_port=port,
        schedule_logic=schedule_logic,
        fake_engine=fake_engine,
        skip_latent_transfer=skip_latent_transfer,
        enable_tracer=enable_tracer,
        log_requests=log_requests,
        perf_model_dir=perf_model_dir,
        engine_config=EngineConfig(
            model_config=ModelConfig(model=SUPPORTED_MODEL_DICT[model_name]),
            runtime_config=RuntimeConfig(
                warmup_steps=0,
                use_cuda_graph=False,
                use_parallel_vae=False,
                use_profiler=False,
                use_torch_compile=False,
            ),
            parallel_config=ParallelConfig(
                dp_config=DataParallelConfig(),
                sp_config=SequenceParallelConfig(ulysses_degree=gpu_num, ring_degree=1),
                tp_config=TensorParallelConfig(),
                pp_config=PipeFusionParallelConfig(),
                dit_parallel_size=gpu_num,
                vae_parallel_size=0,
            ),
            fast_attn_config=FastAttnConfig(),
        ),
    )
    return CServeRuntime(server_args)


async def run_workload(runtime: CServeRuntime, workload, start_time: float):
    """Execute workload on runtime."""
    tasks = []

    for i, req in enumerate(workload.requests):
        arrival_time = workload.arrival_times[i]
        send_time = start_time + arrival_time

        async def execute_request(send_time, req):
            now = asyncio.get_event_loop().time()
            if send_time > now:
                await asyncio.sleep(send_time - now)

            # Convert request to InputConfig
            input_config = InputConfig(
                prompt=req.prompt,
                batch_size=req.batch_size,
                height=req.height,
                width=req.width,
                num_inference_steps=req.num_inference_steps,
                output_type=req.output_type,
            )

            request_start = asyncio.get_event_loop().time()
            await runtime.generate(input_config, req.req_idx, latency_threshold=req.slo)
            request_end = asyncio.get_event_loop().time()

            req.submit_time = request_start - start_time
            req.completion_time = request_end - start_time

        task = asyncio.create_task(execute_request(send_time, req))
        tasks.append(task)

    await asyncio.gather(*tasks)


async def run_experiment(exp_config: Dict[str, Any], args: argparse.Namespace, output_dir: Path) -> Dict[str, Any]:
    """Run single experiment."""
    logger.info(
        f"Running: {exp_config['model_name']} / {exp_config['schedule_logic']} "
        f"/ GPU={exp_config['gpu_num']} / Rate={exp_config['rate']}"
    )

    # Create arrival process
    if exp_config["distribution"] == "poisson":
        arrival_process = PoissonProcess(exp_config["rate"])
    else:
        arrival_process = GammaProcess(exp_config["rate"], exp_config["cv"])

    # Get shape distribution
    shape_dists = {
        "evenly": ShapeDistribution.evenly(),
        "short_seqlen_high": ShapeDistribution.short_seqlen_high(),
        "long_seqlen_high": ShapeDistribution.long_seqlen_high(),
        "long_seqlen_high_power": ShapeDistribution.long_seqlen_high_power(),
    }
    shape_dist = shape_dists.get(exp_config["shape_dist"], ShapeDistribution.evenly())

    # Check if workload should be loaded from file
    workload_file = None
    if "workload_file" in exp_config and exp_config["workload_file"]:
        workload_file = exp_config["workload_file"]
        workload_path = Path(workload_file)
        if not workload_path.exists():
            logger.warning(f"Workload file not found: {workload_path}, falling back to building workload")
            workload_file = None

    if workload_file:
        logger.info(f"Loading workload from file: {workload_file}")
        workload = load_workload(workload_file)

        # Update model name in requests to match exp_config (in case it differs)
        for req in workload.requests:
            req.model_name = exp_config["model_name"]

        # assert False, "TODO: SLO Settings"
        # Apply SLO policy if provided (workload file may not have SLOs)
        slo_policy = load_slo_policy(
            config_name=exp_config["slo_config"],
            model_name=exp_config["model_name"],
            steps=exp_config["steps"],
            config_dir="benchmark/serving_benchmark/slo_configs",
        )
        if slo_policy is not None:
            from tetriserve.workload.slo import apply_slo_policy

            workload = apply_slo_policy(workload, slo_policy)
    else:
        # Load SLO policy
        slo_policy = load_slo_policy(
            config_name=exp_config["slo_config"],
            model_name=exp_config["model_name"],
            steps=exp_config["steps"],
            config_dir="benchmark/serving_benchmark/slo_configs",
        )

        # Build workload
        builder = (
            WorkloadBuilder()
            .with_model(exp_config["model_name"])
            .with_arrival_process(arrival_process)
            .with_shape_distribution(shape_dist)
            .with_batch_distribution(BatchSizeDistribution.single_batch(1))
            .with_num_steps(exp_config["steps"])
            .with_output_type(exp_config["output_type"])
            .with_slo_policy(slo_policy)
            .with_seed(exp_config["seed"])
        )

        # Set duration or num_requests
        if "duration" in exp_config:
            builder.with_duration(exp_config["duration"])
        else:
            builder.with_num_requests(exp_config["request_num"])

        workload = builder.build()

    # Create runtime
    runtime = create_runtime(
        exp_config["model_name"],
        exp_config["schedule_logic"],
        exp_config["port"],
        exp_config["gpu_num"],
        args.fake_engine,
        args.perf_model_dir,
        enable_tracer=exp_config.get("enable_tracer", False),
        log_requests=args.log_requests,
        skip_latent_transfer=args.skip_latent_transfer,
    )

    try:
        # Initialize
        await runtime.init_zmq_router()

        # Warmup (skip if fake_engine or explicitly disabled)
        enable_warmup = exp_config.get("enable_warmup", True)  # Default: True
        if not args.fake_engine and enable_warmup:
            logger.info("Running warmup...")
            await runtime.warmup_test_all_combinations(if_continous=True)
        elif not enable_warmup:
            logger.info("Warmup disabled by configuration")

        # Execute workload
        loop = asyncio.get_event_loop()
        start_time = loop.time()
        await run_workload(runtime, workload, start_time)

        # Compute statistics
        stats = compute_statistics(workload, warmup_seconds=5.0)

        # Get per-request information for detailed analysis
        per_request_info = get_per_request_info(workload, warmup_seconds=5.0)

        # Export trace if tracer was enabled
        trace_file = None
        logger.info(f"Enable tracer: {exp_config.get('enable_tracer', False)}")
        if exp_config.get("enable_tracer", False):
            config_key = get_config_key(exp_config)
            # Create trace directory under output_dir
            trace_dir = output_dir / "traces"
            trace_dir.mkdir(parents=True, exist_ok=True)
            # Generate unique trace filename based on config
            trace_filename = config_key.replace("|", "_").replace("=", "-") + ".json"
            trace_file = str(trace_dir / trace_filename)
            runtime.export_trace(trace_file)
            logger.info(f"Trace saved to: {trace_file}")

        # Format result
        result = {
            "config": exp_config,
            "stats": {
                # Basic metrics
                "num_requests": stats.num_requests,
                "throughput": stats.throughput,
                "request_rate": stats.request_rate,
                # Latency metrics
                "latency_mean": stats.latency_mean,
                "latency_std": stats.latency_std,
                "latency_median": stats.latency_median,
                "latency_min": stats.latency_min,
                "latency_max": stats.latency_max,
                "latency_p50": stats.latency_p50,
                "latency_p90": stats.latency_p90,
                "latency_p95": stats.latency_p95,
                "latency_p99": stats.latency_p99,
                "latency": stats.latencies.tolist(),  # Full array for detailed analysis
                # SLO metrics
                "slo_meet_ratio": stats.slo_meet_ratio,
                "slo_violations": stats.slo_violations,
                # Shape distribution (useful metadata)
                "shape_distribution": {f"{h}x{w}": count for (h, w), count in stats.shape_distribution.items()},
                # Per-request information for detailed plotting (raw data calculation)
                "per_request_info": per_request_info,
                "request_starts": stats.start_times.tolist(),
                "request_finishes": stats.completion_times.tolist(),
            },
            "timestamp": datetime.now().isoformat(),
        }

        # Add trace file path to result
        if trace_file:
            result["trace_file"] = trace_file

        return result

    finally:
        # Cleanup
        try:
            await runtime._send_done_packet()
        except:
            pass
        await asyncio.sleep(2)  # Cleanup delay


def get_config_key(exp_config: Dict[str, Any]) -> str:
    """Generate unique key for an experiment configuration."""
    # Core parameters that uniquely identify an experiment
    core_keys = [
        "model_name",
        "schedule_logic",
        "gpu_num",
        "distribution",
        "rate",
        "cv",
        "steps",
        "shape_dist",
        "output_type",
        "duration",
        "request_num",
        "seed",
        "slo_config",
    ]

    key_parts = []
    for k in sorted(core_keys):
        if k in exp_config:
            v = exp_config[k]
            key_parts.append(f"{k}={v}")

    return "|".join(key_parts)


def load_existing_results(results_file: Path) -> Dict[str, Dict]:
    """Load existing results for resume functionality."""
    if not results_file.exists():
        return {}

    try:
        with open(results_file, "r") as f:
            data = json.load(f)

        # Build lookup dict: config_key -> result
        existing = {}
        for result in data:
            if "config" in result and not result.get("error"):
                # Only load successful experiments
                key = get_config_key(result["config"])
                existing[key] = result

        logger.info(f"Loaded {len(existing)} existing successful experiments")
        return existing

    except Exception as e:
        logger.warning(f"Could not load existing results: {e}")
        return {}


async def run_all_experiments(config: Dict[str, Any], args: argparse.Namespace):
    """Run all experiment combinations."""
    results = []

    # Setup output directory
    output_dir = Path(config.get("output_dir", "benchmark/serving/results"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Auto-generate file paths from output_dir
    console_log_file = output_dir / "console.log"  # Full console output (replaces tee)
    structured_log_file = output_dir / "benchmark.log"  # Structured logging only
    results_file = output_dir / "benchmark_results.json"

    # Load existing results if resume mode
    existing_results = {}
    if config.get("resume", False):
        existing_results = load_existing_results(results_file)
        logger.info(f"Resume mode: will skip {len(existing_results)} completed experiments")

    # Setup structured logging only (console logging handled by subprocess+tee)
    if config.get("save_logs", True):
        import logging

        # Setup structured logging
        file_handler = logging.FileHandler(structured_log_file, mode="w")
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        file_handler.setLevel(logging.DEBUG)  # Capture all levels
        logging.getLogger().addHandler(file_handler)
        logger.info(f"📝 Structured logging: {structured_log_file}")

    # Use try/finally to ensure logging cleanup
    try:
        # Generate and run experiments
        # First, calculate total number of experiments
        total_experiments = (
            len(config["models"])
            * len(config["gpu_nums"])
            * len(config["output_types"])
            * len(config["distributions"])
            * len(config["cvs"])
            * len(config["steps"])
            * len(config["shape_dists"])
            * len(config["slo_configs"])
            * len(config["schedule_logics"])
            * len(config["rates"])
        )

        current_experiment = 0
        completed_count = 0
        skipped_count = 0

        for model in config["models"]:
            for gpu_num in config["gpu_nums"]:
                for output_type in config["output_types"]:
                    for distribution in config["distributions"]:
                        for cv in config["cvs"]:
                            for steps in config["steps"]:
                                for shape_dist in config["shape_dists"]:
                                    for slo_config in config["slo_configs"]:
                                        for schedule_logic in config["schedule_logics"]:
                                            for rate in config["rates"]:
                                                current_experiment += 1

                                                exp_config = {
                                                    "model_name": model,
                                                    "schedule_logic": schedule_logic,
                                                    "gpu_num": gpu_num,
                                                    "distribution": distribution,
                                                    "rate": rate,
                                                    "cv": cv,
                                                    "steps": steps,
                                                    "shape_dist": shape_dist,
                                                    "output_type": output_type,
                                                    "seed": config.get("seed", 42),
                                                    "port": get_free_port(),
                                                    "slo_config": slo_config,
                                                    "enable_tracer": config.get("enable_tracer", False),
                                                    "enable_warmup": config.get("enable_warmup", True),
                                                }

                                                logger.info(
                                                    f"{current_experiment}/{total_experiments}: Exp  config: {exp_config}"
                                                )

                                                # Add duration or request_num
                                                if "duration" in config:
                                                    exp_config["duration"] = config["duration"]
                                                elif "request_nums" in config:
                                                    request_nums_list = (
                                                        config["request_nums"]
                                                        if isinstance(config["request_nums"], list)
                                                        else [config["request_nums"]]
                                                    )
                                                    exp_config["request_num"] = request_nums_list[0]

                                                # Add workload_file if specified in config
                                                if "workload_file" in config:
                                                    exp_config["workload_file"] = config["workload_file"]

                                                # Check if already completed (resume mode)
                                                config_key = get_config_key(exp_config)
                                                if config_key in existing_results:
                                                    skipped_count += 1
                                                    logger.info(
                                                        f"⏭️  Skipping {current_experiment}/{total_experiments}: "
                                                        f"{model}/{schedule_logic} (already completed)"
                                                    )
                                                    continue

                                                # Run experiment
                                                logger.info(
                                                    f"▶️  Running {current_experiment}/{total_experiments}: "
                                                    f"{model}/{schedule_logic}/gpu={gpu_num}/rate={rate}"
                                                )

                                                try:
                                                    result = await run_experiment(exp_config, args, output_dir)
                                                    results.append(result)
                                                    completed_count += 1
                                                    logger.info(f"✓ Completed: {completed_count} new experiments")
                                                except Exception as e:
                                                    logger.error(f"✗ Failed: {e}")
                                                    results.append(
                                                        {
                                                            "config": exp_config,
                                                            "error": str(e),
                                                            "timestamp": datetime.now().isoformat(),
                                                        }
                                                    )

                                                # Save results after each experiment
                                                if existing_results:
                                                    all_results = list(existing_results.values()) + results
                                                else:
                                                    all_results = results

                                                with open(results_file, "w") as f:
                                                    json.dump(all_results, f, indent=2)

        # Merge with existing results if resume mode
        if existing_results:
            all_results = list(existing_results.values()) + results
            logger.info(f"Merging: {len(existing_results)} existing + {len(results)} new = {len(all_results)} total")
        else:
            all_results = results

        # Save results
        with open(results_file, "w") as f:
            json.dump(all_results, f, indent=2)

        logger.info(f"")
        logger.info(f"{'='*70}")
        logger.info(f"Benchmark Complete!")
        logger.info(f"{'='*70}")
        logger.info(f"Total experiments: {total_experiments}")
        logger.info(f"  • Completed (new): {completed_count}")
        logger.info(f"  • Skipped (resume): {skipped_count}")
        logger.info(f"  • Results saved to: {results_file}")
        if config.get("save_logs", True):
            logger.info(f"  • Structured log: {structured_log_file}")
        logger.info(f"{'='*70}")

    finally:
        # Cleanup (no logging cleanup needed with subprocess approach)
        pass


def main_with_tee():
    """Main entry point with tee functionality using subprocess."""
    import subprocess
    import sys

    # Get command line args
    parser = argparse.ArgumentParser(description="FluidServe Serving Benchmark")
    parser.add_argument("--config", type=str, required=True, help="YAML config file")
    parser.add_argument("--output-dir", type=str, help="Output directory (overrides config file setting)")
    parser.add_argument("--fake-engine", action="store_true", help="Enable fake engine (overrides config file setting)")
    parser.add_argument("--skip-latent-transfer", action="store_true", help="Skip NCCL latent transfers, use dummy tensors")
    parser.add_argument(
        "--no-console-log",
        action="store_true",
        help="Disable full console output logging (equivalent to not using tee)",
    )
    args = parser.parse_args()

    if not args.no_console_log:
        # Determine output directory for console log
        config = load_yaml_config(args.config)
        if args.output_dir:
            output_dir = os.path.abspath(args.output_dir)
        else:
            output_dir = config.get("output_dir")
            if not output_dir:
                raise ValueError("output_dir is not set")
            output_dir = os.path.abspath(output_dir)

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        console_log_file = os.path.join(output_dir, "console.log")

        # Build command to re-run with --no-console-log and use shell tee
        cmd = [sys.executable, "-m", "benchmark.serving_benchmark.benchmark_serving"]
        cmd.extend(["--config", args.config])
        if args.output_dir:
            cmd.extend(["--output-dir", args.output_dir])
        if args.fake_engine:
            cmd.append("--fake-engine")
        if args.skip_latent_transfer:
            cmd.append("--skip-latent-transfer")
        cmd.append("--no-console-log")

        # Use shell with tee for proper logging
        shell_cmd = f"{' '.join(cmd)} 2>&1 | tee '{console_log_file}'"
        print(f"Running with console logging: {console_log_file}")

        # Execute with shell
        result = subprocess.run(shell_cmd, shell=True)
        return result.returncode

    # Fall through to regular main
    return main_without_tee()


def main_without_tee():
    """Main entry point without tee functionality."""
    parser = argparse.ArgumentParser(description="FluidServe Serving Benchmark")
    parser.add_argument("--config", type=str, required=True, help="YAML config file")
    parser.add_argument("--output-dir", type=str, help="Output directory (overrides config file setting)")
    parser.add_argument("--fake-engine", action="store_true", help="Enable fake engine (overrides config file setting)")
    parser.add_argument("--skip-latent-transfer", action="store_true", help="Skip NCCL latent transfers, use dummy tensors")
    parser.add_argument(
        "--no-console-log",
        action="store_true",
        help="Disable full console output logging (equivalent to not using tee)",
    )
    args = parser.parse_args()

    # Load config from YAML (everything configured there)
    config = load_yaml_config(args.config)
    logger.info(f"Loaded config from {args.config}")

    # Auto-detect and set GPU_NAME if not already set
    if not os.environ.get("GPU_NAME"):
        try:
            from tetriserve.utils import get_gpu_name

            gpu_name = get_gpu_name()
            os.environ["GPU_NAME"] = gpu_name
            logger.info(f"Auto-detected GPU_NAME: {gpu_name}")
        except Exception as e:
            logger.warning(f"Failed to auto-detect GPU_NAME: {e}. Please set GPU_NAME environment variable if needed.")

    # Handle output_dir priority: command line > config file > auto-generate
    if args.output_dir:
        # Command line argument takes highest priority
        output_dir = os.path.abspath(args.output_dir)
        config["output_dir"] = output_dir
        logger.info(f"Using command line output_dir: {output_dir}")
    else:
        # Fall back to config file or auto-generate
        output_dir = config.get("output_dir")
        if not output_dir:
            # Auto-generate based on config filename
            config_name = Path(args.config).stem
            output_dir = f"benchmark/serving_benchmark/results/{config_name}"
            config["output_dir"] = output_dir
            logger.info(f"Auto-generated output_dir: {output_dir}")
        elif not os.path.isabs(output_dir):
            # Convert relative path to absolute
            output_dir = os.path.abspath(output_dir)
            config["output_dir"] = output_dir
            logger.info(f"Using config file output_dir (made absolute): {output_dir}")
        else:
            logger.info(f"Using config file output_dir: {output_dir}")

    # Handle fake_engine override: command line > config file
    if args.fake_engine:
        # Command line --fake-engine takes priority
        config["fake_engine"] = True
        logger.info("Using command line: fake_engine=True")
    else:
        # Use config file setting or default
        if "fake_engine" in config:
            logger.info(f"Using config file: fake_engine={config['fake_engine']}")
        else:
            config["fake_engine"] = False
            logger.info("Using default: fake_engine=False")

    # Get settings from YAML (with defaults)
    args.fake_engine = config.get("fake_engine", False)
    args.skip_latent_transfer = config.get("skip_latent_transfer", False)
    args.log_requests = config.get("log_requests", False)
    args.perf_model_dir = config.get("perf_model_dir", "log/benchmark/scaling_efficiency")

    # Log configuration
    logger.info(f"Configuration:")
    logger.info(f"  Models: {config['models']}")
    logger.info(f"  Schedulers: {config['schedule_logics']}")
    logger.info(f"  GPU nums: {config['gpu_nums']}")
    logger.info(f"  Enable tracer: {config.get('enable_tracer', False)}")
    logger.info(f"  Fake engine: {args.fake_engine}")
    logger.info(f"  Output dir: {config.get('output_dir', 'benchmark/serving/results')}")
    logger.info(f"  GPU_NAME: {os.environ.get('GPU_NAME', 'Not set')}")

    # Set seed
    set_all_seeds(config.get("seed", 42))

    # Run experiments
    asyncio.run(run_all_experiments(config, args))


def main():
    """Main entry point - delegates to tee or non-tee version."""
    return main_with_tee()


if __name__ == "__main__":
    main()
