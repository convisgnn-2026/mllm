"""
This file is modified from:
https://github.com/hao-ai-lab/MuxServe/blob/main/muxserve/tracer.py
"""

import json
import time
import re

from tetriserve.xfuser.logger import init_logger

logger = init_logger(__name__)


class FluidServeTracer:

    def __init__(self, process_names):
        self.process_names = process_names
        self.init(process_names)

    def init(self, process_names):
        self.trace_events = []

        self.proc_to_pid = {}
        for pid, proc_name in enumerate(process_names):
            self.proc_to_pid[proc_name] = pid
            meta_event = {"name": "thread_name", "ph": "M", "pid": pid, "tid": 0, "args": {"name": f"{proc_name}"}}
            self.trace_events.append(meta_event)
            meta_event = {
                "name": "thread_sort_index",
                "ph": "M",
                "pid": pid,
                "tid": 0,
                "args": {
                    "sort_index": pid,
                },
            }
            self.trace_events.append(meta_event)

        self.event_holder = {}

    def reset(self):
        self.init(self.process_names)

    def add_trace_event(self, name, cat, proc_name, times, args=None):
        """
        Adds a trace event pair (Begin and End) to the trace list.

        Args:
            name (str): Event name.
            cat (str): Event category.
            proc_name (str): Process name.
            times (list): List containing [start_time, end_time].
            args (dict, optional): Additional arguments to add to the Begin event. Defaults to None.
        """
        for ph, ts in zip(["B", "E"], times):
            event = {
                "name": name,
                "cat": cat,
                "ph": ph,
                "pid": self.proc_to_pid[proc_name],
                "tid": 0,  # Assuming single thread per process for simplicity
                "ts": ts * 1e6,  # Convert seconds to microseconds
            }
            # Add custom arguments only to the Begin event
            if ph == "B" and args:
                event["args"] = args
            self.trace_events.append(event)

    def add_event(self, name, cat, proc_name, ts, start=True, **kwargs):
        """
        Adds a start or end point for an event. Stores custom kwargs on start.

        Args:
            name (str): Event name.
            cat (str): Event category.
            proc_name (str): Process name.
            ts (float): Timestamp (in seconds).
            start (bool, optional): True if this is the start time, False otherwise. Defaults to True.
            **kwargs: Arbitrary keyword arguments to store as 'args' in the trace event (used only when start=True).
        """
        if start:
            # Store timestamp and any additional args passed via kwargs
            self.event_holder[proc_name] = (ts, kwargs)
            return

        # Pop the start timestamp and its associated args
        st_tick, start_args = self.event_holder.pop(proc_name, (None, None))
        if st_tick is None:
            logger.warning(f"End event '{name}' for process '{proc_name}' called without a corresponding start event.")
            return

        times = [st_tick, ts]
        # Pass the stored arguments to add_trace_event
        self.add_trace_event(name, cat, proc_name, times, args=start_args)

    def export(self, trace_file):
        logger.info(f"Export execution trace to {trace_file}")
        trace_data = {"traceEvents": self.trace_events, "displayTimeUnit": "ms"}
        with open(f"{trace_file}", "w") as f:
            json.dump(trace_data, f)

    def export_readable_json(self, output_file):
        """
        Export trace events in a simplified, readable JSON format with parsed fields.
        Treats each B/E pair as a distinct event, even with identical names.

        Args:
            output_file (str): Path to output file
        """
        logger.info(f"Export readable trace data to {output_file}")

        start_events = {}  # Key: f"{cat}_{name}_{pid}", Value: partially filled event data dict
        completed_events = []  # List to store fully processed event dicts

        # Process beginning events ('B')
        for event in self.trace_events:
            if event.get("ph") == "B":
                name = event.get("name")
                cat = event.get("cat")
                pid = event.get("pid")
                ts = event.get("ts") / 1e6  # Convert timestamp to seconds
                args = event.get("args", {})  # Get custom arguments

                # Use a unique key for each potential event instance
                key = f"{cat}_{name}_{pid}"

                # Initialize the event data dictionary
                event_data = {
                    "name": name,
                    "start_time": ts,
                    "end_time": None,
                    "duration": None,
                    "gpus": [],
                    # Add any custom args found
                    **args,
                }

                # Attempt to parse common fields from the name string
                # (Keep existing parsing logic)
                try:
                    # Parse and extract fields from the name
                    req_id_match = re.search(r"(?:ReqId|Request-ID):\s*\[(.*?)\]", name)
                    if req_id_match:
                        req_id_text = req_id_match.group(1)
                        try:
                            if "," in req_id_text:
                                event_data["req_id"] = [int(x.strip()) for x in req_id_text.split(",")]
                            else:
                                event_data["req_id"] = int(req_id_text)
                        except ValueError:
                            event_data["req_id"] = req_id_text

                    bs_match = re.search(r"BS:\s*\[(.*?)\]", name)
                    if bs_match:
                        bs_text = bs_match.group(1)
                        try:
                            if "," in bs_text:
                                event_data["bs"] = [int(x.strip()) for x in bs_text.split(",")]
                            else:
                                event_data["bs"] = int(bs_text)
                        except ValueError:
                            event_data["bs"] = bs_text

                    steps_match = re.search(r"Steps:\s*\[(.*?)\]", name)
                    if steps_match:
                        steps_text = steps_match.group(1)
                        try:
                            if "," in steps_text:
                                event_data["steps"] = [int(x.strip()) for x in steps_text.split(",")]
                            else:
                                event_data["steps"] = int(steps_text)
                        except ValueError:
                            event_data["steps"] = steps_text

                    seqlen_match = re.search(r"Seq-Len:\s*\[(.*?)\]", name)
                    if seqlen_match:
                        seqlen_text = seqlen_match.group(1)
                        try:
                            if "," in seqlen_text:
                                event_data["seqlen"] = [int(x.strip()) for x in seqlen_text.split(",")]
                            else:
                                event_data["seqlen"] = int(seqlen_text)
                        except ValueError:
                            event_data["seqlen"] = seqlen_text

                except Exception as e:
                    logger.warning(f"Failed to parse event name: {name}, error: {str(e)}")

                # Extract GPU number from process name if applicable
                for proc_name, proc_pid in self.proc_to_pid.items():
                    if proc_pid == pid and "Worker" in proc_name:  # Assuming worker processes are named like "Worker-N"
                        gpu_match = re.search(r"Worker-(\d+)", proc_name)
                        if gpu_match:
                            gpu_num = int(gpu_match.group(1))
                            if gpu_num not in event_data["gpus"]:  # Avoid duplicates if B event is somehow duplicated
                                event_data["gpus"].append(gpu_num)

                # Store the initialized event data, waiting for its 'E' event
                start_events[key] = event_data

        # Process ending events ('E') and finalize event data
        for event in self.trace_events:
            if event.get("ph") == "E":
                name = event.get("name")
                cat = event.get("cat")
                pid = event.get("pid")
                ts = event.get("ts") / 1e6  # Convert timestamp to seconds

                # Find the corresponding start event
                key = f"{cat}_{name}_{pid}"
                if key in start_events:
                    # Retrieve the partially filled event data
                    event_data = start_events.pop(key)  # Remove from start_events as it's now complete

                    # Set end time and calculate duration
                    event_data["end_time"] = ts
                    if event_data["start_time"] is not None:
                        event_data["duration"] = event_data["end_time"] - event_data["start_time"]

                    # Add the completed event data to our list
                    completed_events.append(event_data)
                # else: Handle cases where an 'E' event doesn't have a corresponding 'B' if necessary

        # Sort the completed events by starting time
        completed_events.sort(key=lambda x: x["start_time"])

        # Sort GPU lists within each event for consistent output
        for event in completed_events:
            if "gpus" in event:
                event["gpus"].sort()

        # Write the list of completed events to the file
        with open(output_file, "w") as f:
            json.dump(completed_events, f, indent=2)

        return completed_events


if __name__ == "__main__":
    tracer = FluidServeTracer(["gpu_0", "gpu_1", "gpu_2", "gpu_3"])

    for i in range(10):
        # some workload
        time.sleep(0.01)

        start = time.time()
        tracer.add_event(
            f"ReqId:[{i}], Steps:[1], BS:[1], Seq-Len:[16384]",
            "inference",
            f"gpu_{i % 4}",
            start,
            start=True,
            estimated_time=0.01,
        )
        time.sleep(0.02)
        start = time.time()
        tracer.add_event(
            f"ReqId:[{i}], Steps:[1], BS:[1], Seq-Len:[16384]",
            "inference",
            f"gpu_{(i + 1) % 4}",
            start,
            start=True,
            estimated_time=0.2,
        )

        end = time.time()
        tracer.add_event(
            f"ReqId:[{i}], Steps:[1], BS:[1], Seq-Len:[16384]", "inference", f"gpu_{i % 4}", end, start=False
        )
        time.sleep(0.2)
        end = time.time()
        tracer.add_event(
            f"ReqId:[{i}], Steps:[1], BS:[1], Seq-Len:[16384]", "inference", f"gpu_{(i + 1) % 4}", end, start=False
        )

    tracer.export("trace.json")
    tracer.export_readable_json("trace_readable.json")
