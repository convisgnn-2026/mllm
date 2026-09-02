import asyncio
import os

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
)
from tetriserve.xfuser.config.config import FastAttnConfig
from tetriserve.engine.runtime import CServeRuntime
from tetriserve.xfuser.config import InputConfig
from tetriserve.workload import (
    Workload,
    GammaProcess,
    populate_requests,
    apply_slo_policy,
    BaselineSLOPolicy,
    ShapeDistribution,
    BatchSizeDistribution,
    STEPS_LIST,
    RATE_LIST,
    SHAPE_LIST,
    BS_DISTRIBUTION,
)
from tetriserve.utils import set_all_seeds
from tetriserve import SUPPORTED_MODEL_DICT

from tetriserve.xfuser.logger import init_logger

logger = init_logger(__name__)

START_SERVER = False


async def run_workload(
    runtime: CServeRuntime, workload: Workload, start_server=False, end_server=False, warmup=True, trace_params=None
):
    global START_SERVER
    if START_SERVER or start_server:
        await runtime.init_zmq_router()
        START_SERVER = False

    # warmup
    if warmup:
        tasks = []
        for i in range(4):

            input_config = InputConfig(
                prompt="a photo of a cat",
                batch_size=1,
                height=2048,
                width=2048,
                num_inference_steps=4,
                output_type="pil",
            )
            task = asyncio.create_task(runtime.generate(input_config, 1037 + i * 1037, latency_threshold=float("inf")))
            tasks.append(task)
        await asyncio.gather(*tasks)

    loop = asyncio.get_event_loop()

    async def run():
        tasks = []
        start_time = loop.time()
        for i in range(len(workload.arrivals)):
            arrival_time = workload.arrivals[i]
            request = workload.requests[i]
            send_time = start_time + arrival_time

            async def scheduled_request(send_time, request, arrival_time):
                now = loop.time()
                if send_time > now:
                    await asyncio.sleep(send_time - now)
                else:
                    logger.info("Request is too early")

                input_dict = request.data
                latency_threshold = input_dict["latency_threshold"]
                del input_dict["latency_threshold"]
                input_config = InputConfig(**input_dict)
                request_output = await runtime.generate(
                    input_config=input_config, request_id=request.req_idx, latency_threshold=latency_threshold
                )

                completion_time = loop.time()

                request.submit_time = send_time - start_time
                request.completion_time = completion_time - start_time

                return request.completion_time

            task = asyncio.create_task(scheduled_request(send_time, request, arrival_time))
            tasks.append(task)

        results = await asyncio.gather(*tasks)
        return results

    if runtime.engine.enable_tracer:
        runtime.reset_tracer()

    results = await run()

    # Create detailed trace filename
    if trace_params:
        import datetime

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Extract model name (get the last part after slash, remove "diffusers" suffix if present)
        model_short = trace_params["model_name"].split("/")[-1].replace("-diffusers", "")

        if not os.path.exists("log/examples"):
            os.makedirs("log/examples")

        # Create detailed filename
        trace_filename = (
            f"log/examples/trace_{timestamp}_{trace_params['schedule_logic']}_"
            f"{model_short}_gpu{trace_params['gpu_num']}_"
            f"rate{trace_params['rate']}_cv{trace_params['cv']}_"
            f"{trace_params['dist_name']}_dur{trace_params['duration']}s_"
            f"steps{trace_params['steps']}_"
            f"{'fake' if trace_params['fake_engine'] else 'real'}.json"
        )
    else:
        trace_filename = f"log/examples/trace_{runtime.engine.scheduler.schedule_logic}.json"

    runtime.export_trace(trace_file=trace_filename)

    if end_server:
        await runtime._send_done_packet()

    return results


def get_runtime(
    model_name,
    schedule_logic,
    port,
    gpu_num,
    enable_tracer=False,
    log_requests=False,
    fake_engine=True,
):
    server_args = ServerArgs(
        nnodes=1,
        nproc_per_node=gpu_num,
        master_addr="127.0.0.1",
        master_port=port,
        schedule_logic=schedule_logic,
        fake_engine=fake_engine,
        enable_tracer=enable_tracer,
        log_requests=log_requests,
        perf_model_dir="log/benchmark/scaling_efficiency",
        engine_config=EngineConfig(
            model_config=ModelConfig(
                model=model_name,
            ),
            runtime_config=RuntimeConfig(
                warmup_steps=0,
                use_cuda_graph=False,
                use_parallel_vae=False,
                use_profiler=False,
                use_torch_compile=False,
                use_onediff=False,
            ),
            parallel_config=ParallelConfig(
                dp_config=DataParallelConfig(),
                sp_config=SequenceParallelConfig(
                    ulysses_degree=gpu_num,
                    ring_degree=1,
                ),
                tp_config=TensorParallelConfig(),
                pp_config=PipeFusionParallelConfig(),
                dit_parallel_size=gpu_num,
                vae_parallel_size=0,
            ),
            fast_attn_config=FastAttnConfig(),
        ),
    )

    runtime = CServeRuntime(server_args)

    return runtime


def run_server(
    schedule_logic,
    model_name,
    port,
    gpu_num,
    rate,
    cv,
    dist_name,
    duration,
    enable_tracer=False,
    steps=4,
    log_requests=False,
    debug=False,
    fake_engine=True,
    start_server=False,
    end_server=False,
):
    workload = GammaProcess(rate, cv).generate_workload("", start=0, duration=duration, seed=42)

    # Map distribution name to ShapeDistribution
    distribution_map = {
        "evenly": ShapeDistribution.evenly(),
        "short_seqlen_high": ShapeDistribution.short_seqlen_high(),
        "long_seqlen_high": ShapeDistribution.long_seqlen_high(),
        "poisson": ShapeDistribution.poisson(),
    }
    shape_dist = distribution_map.get(dist_name, ShapeDistribution.evenly())

    print(workload)

    populate_requests(
        workload,
        shape_distribution=shape_dist,
        num_inference_steps=steps,
        output_type="latent",
        seed=42,
        batch_distribution=BatchSizeDistribution.single_batch(1),
    )

    # Apply SLO policy
    slo_policy = BaselineSLOPolicy.from_file(model_name=model_name.split("/")[-1], steps=steps)
    apply_slo_policy(workload, slo_policy)

    runtime = get_runtime(model_name, schedule_logic, port, gpu_num, enable_tracer, log_requests, fake_engine)

    warmup = not fake_engine

    # Create trace parameters for detailed filename
    trace_params = (
        {
            "schedule_logic": schedule_logic,
            "model_name": model_name,
            "gpu_num": gpu_num,
            "rate": rate,
            "cv": cv,
            "dist_name": dist_name,
            "duration": duration,
            "steps": steps,
            "fake_engine": fake_engine,
        }
        if enable_tracer
        else None
    )

    completion_times = asyncio.run(
        run_workload(
            runtime=runtime,
            workload=workload,
            start_server=start_server,
            end_server=end_server,
            warmup=warmup,
            trace_params=trace_params,
        )
    )


"""
python examples/run_server.py 2>&1 | tee log.log
"""

if __name__ == "__main__":

    set_all_seeds(42)

    from tetriserve.workload import STEPS

    run_server(
        # schedule_logic="dyn_slo_schedule_global",
        # schedule_logic="fcfs_optimal_shape",
        schedule_logic="fixed_sp_2",
        model_name=SUPPORTED_MODEL_DICT["sd3"],
        port=1037,
        gpu_num=4,
        rate=1.0,
        cv=1.0,
        dist_name="short_seqlen_high",
        duration=30,
        enable_tracer=True,
        steps=STEPS,
        log_requests=True,
        debug=True,
        fake_engine=False,
        start_server=True,
        end_server=True,
    )
