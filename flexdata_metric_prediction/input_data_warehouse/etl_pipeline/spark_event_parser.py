"""
Spark Event Log Parser

This script parses Spark event log files and extracts per-query-run metrics.

EXECUTION MODEL (from spark_job_runner.py):
- 1 warmup pass: run_type="warmup", run_id="warmup_q{N}"
- 3 measurement passes: run_type="measurement", run_id="q{N}_run{M}"
- 125s timeout triggers job cancellation

INPUT:
- Spark event log file (JSON lines format)

OUTPUT:
- CSV file with per-query-run metrics
- Columns: run_id, run_type, status, execution metrics, I/O metrics, memory metrics, config

KEY DESIGN:
1. Single-pass streaming parse
2. Fingerprint tracking via job properties (run.id, run.type)
3. Hierarchical aggregation: Task → Stage → Run
4. Handle stage retries (aggregate latest attempt only)
"""

import json
import csv
from collections import defaultdict


def parse_event_log(log_file, output_csv, exclude_warmup=False):
    """
    Parse Spark event log and produce per-query-run metrics CSV.

    Args:
        log_file: Path to Spark event log file
        output_csv: Path to output CSV file
        exclude_warmup: If True, exclude runs with run_type="warmup"
    """
    # State tracking
    jobs = {}  # job_id -> {run_id, run_type, status, config, ...}
    stages = {}  # (stage_id, attempt_id) -> {run_id, metrics, status, ...}
    stage_to_run = {}  # stage_id -> run_id (map from JobStart)
    sql_executions = {}  # execution_id -> {description, duration_ms, ...}

    with open(log_file) as f:
        for line in f:
            event = json.loads(line.strip())
            event_type = event.get("Event", "")

            # === SQL Execution tracking (optional metadata) ===
            if "SQLExecutionStart" in event_type:
                exec_id = event.get("executionId")
                if exec_id:
                    sql_executions[exec_id] = {
                        "description": event.get("description", ""),
                        "start_time": event.get("time"),
                    }

            elif "SQLExecutionEnd" in event_type:
                exec_id = event.get("executionId")
                if exec_id in sql_executions:
                    end_time = event.get("time")
                    start_time = sql_executions[exec_id].get("start_time")
                    if start_time and end_time:
                        sql_executions[exec_id]["duration_ms"] = end_time - start_time

            # === Job tracking (fingerprint extraction) ===
            elif event_type == "SparkListenerJobStart":
                job_id = event["Job ID"]
                props = event.get("Properties", {})
                run_id = props.get("run.id")

                if run_id:  # Only track fingerprinted queries
                    exec_id = props.get("spark.sql.execution.id")
                    jobs[job_id] = {
                        "run_id": run_id,
                        "run_type": props.get("run.type", "measurement"),
                        "execution_id": int(exec_id) if exec_id else None,
                        "status": "success",  # Assume success until proven otherwise
                        # Spark configuration
                        "driver_memory": props.get("spark.driver.memory"),
                        "executor_memory": props.get("spark.executor.memory"),
                        "executor_cores": props.get("spark.executor.cores"),
                        "executor_instances": props.get("spark.executor.instances"),
                    }

                    # Map stages to run_id (key insight: JobStart has "Stage IDs")
                    for stage_id in event.get("Stage IDs", []):
                        stage_to_run[stage_id] = run_id

            elif event_type == "SparkListenerJobEnd":
                job_id = event["Job ID"]
                if job_id in jobs:
                    result = event.get("Job Result", {}).get("Result")
                    if result != "JobSucceeded":
                        # Check if failure was due to SparkContext shutdown (not a real failure)
                        exception_msg = (
                            event.get("Job Result", {})
                            .get("Exception", {})
                            .get("Message", "")
                        )
                        if "SparkContext was shut down" not in exception_msg:
                            jobs[job_id]["status"] = "failed"

            # === Stage tracking ===
            elif event_type == "SparkListenerStageSubmitted":
                stage_info = event["Stage Info"]
                stage_id = stage_info["Stage ID"]
                attempt_id = stage_info.get("Stage Attempt ID", 0)

                if stage_id in stage_to_run:
                    stages[(stage_id, attempt_id)] = {
                        "run_id": stage_to_run[stage_id],
                        "num_tasks": stage_info.get("Number of Tasks", 0),
                        "status": "success",
                        "metrics": {
                            "executor_run_time": 0,
                            "executor_cpu_time": 0,
                            "jvm_gc_time": 0,
                            "memory_bytes_spilled": 0,
                            "disk_bytes_spilled": 0,
                            "input_bytes": 0,
                            "output_bytes": 0,
                            "shuffle_read_bytes": 0,
                            "shuffle_write_bytes": 0,
                            "executor_deserialize_time": 0,
                            "result_serialization_time": 0,
                            "shuffle_write_time": 0,
                            "shuffle_fetch_wait_time": 0,
                            "peak_execution_memory": 0,
                            "jvm_heap_memory": 0,
                            "jvm_off_heap_memory": 0,
                            "on_heap_execution_memory": 0,
                            "on_heap_storage_memory": 0,
                            "direct_pool_memory": 0,
                        },
                    }

            elif event_type == "SparkListenerStageCompleted":
                stage_info = event["Stage Info"]
                stage_id = stage_info["Stage ID"]
                attempt_id = stage_info.get("Stage Attempt ID", 0)
                stage_key = (stage_id, attempt_id)

                if stage_key in stages:
                    failure_reason = stage_info.get("Failure Reason", "")
                    # Check if failure was due to SparkContext shutdown (not a real failure)
                    if (
                        failure_reason
                        and "SparkContext was shut down" not in failure_reason
                    ):
                        stages[stage_key]["status"] = "failed"

            # === Task metrics aggregation ===
            elif event_type == "SparkListenerTaskEnd":
                stage_id = event.get("Stage ID")
                attempt_id = event.get("Stage Attempt ID", 0)
                stage_key = (stage_id, attempt_id)

                if stage_key not in stages:
                    continue

                tm = event.get("Task Metrics", {})
                tem = event.get("Task Executor Metrics", {})
                m = stages[stage_key]["metrics"]

                # Cumulative metrics (SUM)
                m["executor_run_time"] += tm.get("Executor Run Time", 0)
                m["executor_cpu_time"] += tm.get("Executor CPU Time", 0)
                m["jvm_gc_time"] += tm.get("JVM GC Time", 0)
                m["memory_bytes_spilled"] += tm.get("Memory Bytes Spilled", 0)
                m["disk_bytes_spilled"] += tm.get("Disk Bytes Spilled", 0)
                m["executor_deserialize_time"] += tm.get("Executor Deserialize Time", 0)
                m["result_serialization_time"] += tm.get("Result Serialization Time", 0)

                # I/O metrics (SUM)
                m["input_bytes"] += tm.get("Input Metrics", {}).get("Bytes Read", 0)
                m["output_bytes"] += tm.get("Output Metrics", {}).get(
                    "Bytes Written", 0
                )

                # Shuffle metrics (SUM)
                sr = tm.get("Shuffle Read Metrics", {})
                m["shuffle_read_bytes"] += sr.get("Remote Bytes Read", 0) + sr.get(
                    "Local Bytes Read", 0
                )
                m["shuffle_fetch_wait_time"] += sr.get("Fetch Wait Time", 0)

                sw = tm.get("Shuffle Write Metrics", {})
                m["shuffle_write_bytes"] += sw.get("Shuffle Bytes Written", 0)
                m["shuffle_write_time"] += sw.get("Shuffle Write Time", 0)

                # Memory metrics (MAX - peak snapshots)
                m["peak_execution_memory"] = max(
                    m["peak_execution_memory"], tm.get("Peak Execution Memory", 0)
                )
                m["jvm_heap_memory"] = max(
                    m["jvm_heap_memory"], tem.get("JVMHeapMemory", 0)
                )
                m["jvm_off_heap_memory"] = max(
                    m["jvm_off_heap_memory"], tem.get("JVMOffHeapMemory", 0)
                )
                m["on_heap_execution_memory"] = max(
                    m["on_heap_execution_memory"], tem.get("OnHeapExecutionMemory", 0)
                )
                m["on_heap_storage_memory"] = max(
                    m["on_heap_storage_memory"], tem.get("OnHeapStorageMemory", 0)
                )
                m["direct_pool_memory"] = max(
                    m["direct_pool_memory"], tem.get("DirectPoolMemory", 0)
                )

    # === Rollup: Aggregate stages to run_id (latest attempts only) ===
    latest_attempts = {}  # stage_id -> max_attempt_id
    for stage_id, attempt_id in stages.keys():
        latest_attempts[stage_id] = max(latest_attempts.get(stage_id, 0), attempt_id)

    rollup = defaultdict(
        lambda: {
            "run_type": "measurement",
            "status": "success",
            "num_jobs": 0,
            "num_stages": 0,
            "num_tasks": 0,
            "num_stage_retries": 0,
            "execution_id": None,
            "query_description": "",
            "wall_clock_duration_ms": None,
            "driver_memory": None,
            "executor_memory": None,
            "executor_cores": None,
            "executor_instances": None,
            "executor_run_time": 0,
            "executor_cpu_time": 0,
            "jvm_gc_time": 0,
            "memory_bytes_spilled": 0,
            "disk_bytes_spilled": 0,
            "input_bytes": 0,
            "output_bytes": 0,
            "shuffle_read_bytes": 0,
            "shuffle_write_bytes": 0,
            "executor_deserialize_time": 0,
            "result_serialization_time": 0,
            "shuffle_write_time": 0,
            "shuffle_fetch_wait_time": 0,
            "peak_execution_memory": 0,
            "jvm_heap_memory": 0,
            "jvm_off_heap_memory": 0,
            "on_heap_execution_memory": 0,
            "on_heap_storage_memory": 0,
            "direct_pool_memory": 0,
        }
    )

    # Aggregate job-level metadata
    for job_data in jobs.values():
        run_id = job_data["run_id"]

        if exclude_warmup and job_data["run_type"] == "warmup":
            continue

        rollup[run_id]["num_jobs"] += 1
        rollup[run_id]["run_type"] = job_data["run_type"]

        if job_data["status"] == "failed":
            rollup[run_id]["status"] = "failed"

        # Link to SQL execution metadata
        exec_id = job_data.get("execution_id")
        if exec_id and exec_id in sql_executions:
            rollup[run_id]["execution_id"] = exec_id
            rollup[run_id]["query_description"] = sql_executions[exec_id].get(
                "description", ""
            )
            rollup[run_id]["wall_clock_duration_ms"] = sql_executions[exec_id].get(
                "duration_ms"
            )

        # Capture Spark configuration (first job only)
        if rollup[run_id]["driver_memory"] is None:
            rollup[run_id]["driver_memory"] = job_data.get("driver_memory")
            rollup[run_id]["executor_memory"] = job_data.get("executor_memory")
            rollup[run_id]["executor_cores"] = job_data.get("executor_cores")
            rollup[run_id]["executor_instances"] = job_data.get("executor_instances")

    # Aggregate stage-level metrics (latest attempts only)
    for stage_id, max_attempt in latest_attempts.items():
        stage_key = (stage_id, max_attempt)
        if stage_key not in stages:
            continue

        stage = stages[stage_key]
        run_id = stage["run_id"]

        if exclude_warmup and rollup[run_id]["run_type"] == "warmup":
            continue

        rollup[run_id]["num_stages"] += 1
        rollup[run_id]["num_tasks"] += stage["num_tasks"]

        if max_attempt > 0:
            rollup[run_id]["num_stage_retries"] += 1

        if stage["status"] == "failed":
            rollup[run_id]["status"] = "failed"

        # Aggregate metrics (cumulative metrics use +=, peak metrics use max)
        m = stage["metrics"]
        r = rollup[run_id]

        # Cumulative metrics
        for key in [
            "executor_run_time",
            "executor_cpu_time",
            "jvm_gc_time",
            "memory_bytes_spilled",
            "disk_bytes_spilled",
            "executor_deserialize_time",
            "result_serialization_time",
            "input_bytes",
            "output_bytes",
            "shuffle_read_bytes",
            "shuffle_write_bytes",
            "shuffle_write_time",
            "shuffle_fetch_wait_time",
        ]:
            r[key] += m[key]

        # Peak metrics
        for key in [
            "peak_execution_memory",
            "jvm_heap_memory",
            "jvm_off_heap_memory",
            "on_heap_execution_memory",
            "on_heap_storage_memory",
            "direct_pool_memory",
        ]:
            r[key] = max(r[key], m[key])

    # === Write CSV output ===
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "run_id",
                "run_type",
                "status",
                "num_jobs",
                "num_stages",
                "num_tasks",
                "num_stage_retries",
                "execution_id",
                "query_description",
                "wall_clock_duration_ms",
                # Execution metrics
                "executor_run_time_ms",
                "executor_cpu_time_ns",
                "jvm_gc_time_ms",
                "memory_bytes_spilled",
                "disk_bytes_spilled",
                "peak_execution_memory",
                # I/O metrics
                "input_bytes",
                "output_bytes",
                "shuffle_read_bytes",
                "shuffle_write_bytes",
                # Timing metrics
                "executor_deserialize_time_ms",
                "result_serialization_time_ms",
                "shuffle_write_time_ns",
                "shuffle_fetch_wait_time_ms",
                # Memory metrics
                "jvm_heap_memory",
                "jvm_off_heap_memory",
                "on_heap_execution_memory",
                "on_heap_storage_memory",
                "direct_pool_memory",
                # Configuration
                "driver_memory",
                "executor_memory",
                "executor_cores",
                "executor_instances",
            ]
        )

        for run_id in sorted(rollup.keys()):
            r = rollup[run_id]
            writer.writerow(
                [
                    run_id,
                    r["run_type"],
                    r["status"],
                    r["num_jobs"],
                    r["num_stages"],
                    r["num_tasks"],
                    r["num_stage_retries"],
                    r["execution_id"] or "",
                    r["query_description"],
                    r["wall_clock_duration_ms"] or "",
                    # Execution metrics
                    r["executor_run_time"],
                    r["executor_cpu_time"],
                    r["jvm_gc_time"],
                    r["memory_bytes_spilled"],
                    r["disk_bytes_spilled"],
                    r["peak_execution_memory"],
                    # I/O metrics
                    r["input_bytes"],
                    r["output_bytes"],
                    r["shuffle_read_bytes"],
                    r["shuffle_write_bytes"],
                    # Timing metrics
                    r["executor_deserialize_time"],
                    r["result_serialization_time"],
                    r["shuffle_write_time"],
                    r["shuffle_fetch_wait_time"],
                    # Memory metrics
                    r["jvm_heap_memory"],
                    r["jvm_off_heap_memory"],
                    r["on_heap_execution_memory"],
                    r["on_heap_storage_memory"],
                    r["direct_pool_memory"],
                    # Configuration
                    r["driver_memory"] or "",
                    r["executor_memory"] or "",
                    r["executor_cores"] or "",
                    r["executor_instances"] or "",
                ]
            )

    print(f"Parsed {len(rollup)} query runs")
    if exclude_warmup:
        print("  (warmup runs excluded)")
    print(f"Output: {output_csv}")


def main():
    """Command-line interface for standalone usage."""
    import sys

    if len(sys.argv) < 3:
        print(
            "Usage: python spark_event_parser.py <event_log_file> <output_csv> [--exclude-warmup]"
        )
        sys.exit(1)

    event_log = sys.argv[1]
    output_csv = sys.argv[2]
    exclude_warmup = "--exclude-warmup" in sys.argv

    parse_event_log(event_log, output_csv, exclude_warmup=exclude_warmup)


if __name__ == "__main__":
    main()
