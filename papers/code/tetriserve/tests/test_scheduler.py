import time
from tetriserve.scheduler.dyn_step_scheduler import DynStepScheduler
from tetriserve.config.request import InputConfig
from tetriserve.scheduler.scheduler import Scheduler
from tetriserve.utils import get_current_time

# from tetriserve.workload_legacy import SLO_DICT


def test_optimal_shape():
    scheduler = DynStepScheduler(
        ranks=[0, 1, 2, 3],
        schedule_logic="fcfs_optimal_shape_schedule",
        e2e_scaling_efficiency_path="log/benchmark/scaling_efficiency/sd3/A40/e2e_scaling_efficiency.json",
    )
    bs = 4
    height = 1024
    width = 1024
    num_inference_steps = 50
    # latency_threshold = SLO_DICT[(bs, height, width)]
    latency_threshold = 1.1
    print(f"latency_threshold: {latency_threshold}")

    scheduler.add_request(
        request_id=2,
        sub_req_ids=list(range(-1, -bs - 1, -1)),
        input_config=InputConfig(batch_size=bs, height=height, width=width, num_inference_steps=num_inference_steps),
        arrival_time=get_current_time(),
        latency_threshold=latency_threshold,
    )

    unscheduled_reqs = scheduler.request_tracker.get_merged_requests()

    for unscheduled_req in unscheduled_reqs:
        print(
            f"optimal shape for req {unscheduled_req.req_id}: {scheduler.get_optimal_shape(current_world_size=4, unscheduled_req=unscheduled_req)}"
        )


def test_get_optimal_shape_basic():
    scheduler = DynStepScheduler(
        ranks=[0, 1, 2, 3, 4, 5, 6, 7],
        schedule_logic="dyn_slo_schedule_global_dp",
        e2e_scaling_efficiency_path="log/benchmark/scaling_efficiency/flux/H100_HBM3/e2e_scaling_efficiency.json",
    )

    # scheduler.add_request(request_id=1, sub_req_ids=[-1], input_config=InputConfig(batch_size=1, height=256, width=256, num_inference_steps=10), arrival_time=0, latency_threshold=3.6)
    for i in range(1000):
        scheduler.add_request(
            request_id=i,
            sub_req_ids=[-i],
            input_config=InputConfig(batch_size=1, height=2048, width=2048, num_inference_steps=20),
            arrival_time=0,
            latency_threshold=16.5,
        )

    schedule_start_time = time.time()
    ret_req_list = scheduler.schedule()
    schedule_end_time = time.time()
    schedule_execution_time = schedule_end_time - schedule_start_time
    print(f"Scheduling execution time: {schedule_execution_time:.4f} seconds")

    for req_block in ret_req_list:
        print(f"req_block: {req_block}")
    pass


if __name__ == "__main__":
    test_get_optimal_shape_basic()
    # test_optimal_shape()
