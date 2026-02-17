"""
Spark ETL Script

Processes Spark event logs and extracts per-query-run metrics.
Uses spark_event_parser.py as the core parsing engine.
Standardizes units: time -> seconds, size -> MB.
"""

from pathlib import Path
from typing import Dict, Any
import pandas as pd
import spark_event_parser


# Unit conversion factors (source_unit -> multiplier to get base unit)
UNIT_CONVERSIONS = {
    # Time units (base: seconds)
    "ns": 1e-9,
    "us": 1e-6,
    "ms": 1e-3,
    "s": 1.0,
    "m": 60.0,
    "h": 3600.0,
    "d": 86400.0,
    # Size units (base: MB)
    "B": 1 / (1024**2),
    "kB": 1 / 1024,
    "KB": 1 / 1024,
    "MB": 1.0,
    "GB": 1024.0,
    "TB": 1024**2,
}

# Metric type mapping: metric_name -> (native_unit, target_unit_type)
# Native units from spark_event_parser.py output
SPARK_METRIC_UNITS = {
    # Time metrics in milliseconds
    "wall_clock_duration_ms": ("ms", "time"),
    "executor_run_time_ms": ("ms", "time"),
    "jvm_gc_time_ms": ("ms", "time"),
    "executor_deserialize_time_ms": ("ms", "time"),
    "result_serialization_time_ms": ("ms", "time"),
    "shuffle_fetch_wait_time_ms": ("ms", "time"),
    # Time metrics in nanoseconds
    "executor_cpu_time_ns": ("ns", "time"),
    "shuffle_write_time_ns": ("ns", "time"),
    # Size metrics in bytes
    "memory_bytes_spilled": ("B", "size"),
    "disk_bytes_spilled": ("B", "size"),
    "peak_execution_memory": ("B", "size"),
    "input_bytes": ("B", "size"),
    "output_bytes": ("B", "size"),
    "shuffle_read_bytes": ("B", "size"),
    "shuffle_write_bytes": ("B", "size"),
    "jvm_heap_memory": ("B", "size"),
    "jvm_off_heap_memory": ("B", "size"),
    "on_heap_execution_memory": ("B", "size"),
    "on_heap_storage_memory": ("B", "size"),
    "direct_pool_memory": ("B", "size"),
}

# Column renames: native_name -> unit_agnostic_name
# Remove unit suffixes (_ms, _ns, _bytes) after standardization
COLUMN_RENAMES = {
    # Time metrics - remove _ms suffix
    "wall_clock_duration_ms": "wall_clock_duration",
    "executor_run_time_ms": "executor_run_time",
    "jvm_gc_time_ms": "jvm_gc_time",
    "executor_deserialize_time_ms": "executor_deserialize_time",
    "result_serialization_time_ms": "result_serialization_time",
    "shuffle_fetch_wait_time_ms": "shuffle_fetch_wait_time",
    # Time metrics - remove _ns suffix
    "executor_cpu_time_ns": "executor_cpu_time",
    "shuffle_write_time_ns": "shuffle_write_time",
    # Size metrics - remove _bytes suffix
    "memory_bytes_spilled": "memory_spilled",
    "disk_bytes_spilled": "disk_spilled",
    "input_bytes": "input_data_size",
    "output_bytes": "output_data_size",
    "shuffle_read_bytes": "shuffle_read_size",
    "shuffle_write_bytes": "shuffle_write_size",
}


def convert_to_standard_unit(value: float, source_unit: str, target_unit: str) -> float:
    """Convert value from source_unit to target_unit using UNIT_CONVERSIONS.

    Args:
        value: Numeric value to convert
        source_unit: Source unit (e.g., 'ms', 'B')
        target_unit: Target unit (e.g., 's', 'MB')

    Returns:
        Converted value

    Raises:
        ValueError: If source_unit or target_unit is not recognized
    """
    if not source_unit or source_unit == target_unit:
        return value

    if source_unit not in UNIT_CONVERSIONS:
        raise ValueError(
            f"Unknown source unit '{source_unit}'. " f"Supported units: {', '.join(sorted(UNIT_CONVERSIONS.keys()))}"
        )

    if target_unit not in UNIT_CONVERSIONS:
        raise ValueError(
            f"Unknown target unit '{target_unit}'. " f"Supported units: {', '.join(sorted(UNIT_CONVERSIONS.keys()))}"
        )

    # Convert: source -> base -> target
    base_value = value * UNIT_CONVERSIONS[source_unit]
    return base_value / UNIT_CONVERSIONS[target_unit]


def standardize_spark_metrics(df: pd.DataFrame, spark_config: Dict[str, Any]) -> pd.DataFrame:
    """Standardize Spark metric units according to configuration.

    Args:
        df: DataFrame with Spark metrics
        spark_config: Spark configuration with standard_units settings

    Returns:
        DataFrame with standardized units
    """
    if df.empty:
        return df

    # Get standard units from config
    standard_units = spark_config.get("standard_units", {})
    time_unit = standard_units.get("time", "s")
    size_unit = standard_units.get("size", "MB")

    print(f"\nStandardizing Spark metrics: time -> {time_unit}, size -> {size_unit}")

    conversions_applied = 0

    # Convert each metric based on its type
    for metric_name, (native_unit, metric_type) in SPARK_METRIC_UNITS.items():
        if metric_name not in df.columns:
            continue

        # Determine target unit based on metric type
        target_unit = time_unit if metric_type == "time" else size_unit

        # Skip if already in target unit
        if native_unit == target_unit:
            continue

        # Apply conversion to non-null values
        mask = df[metric_name].notna()
        if mask.any():
            df.loc[mask, metric_name] = df.loc[mask, metric_name].apply(
                lambda x: convert_to_standard_unit(x, native_unit, target_unit)
            )
            conversions_applied += 1

    if conversions_applied > 0:
        print(f"  ✓ Converted {conversions_applied} metric column(s)")

    return df


def process_spark_engine(
    data_dir: str,
    num_queries: int = 1000,
    num_runs: int = 3,
    exclude_warmup: bool = True,
    spark_config: Dict[str, Any] = None,
) -> pd.DataFrame:
    """
    Process all Spark event logs for a Spark engine.

    Args:
        data_dir: Directory containing Spark event log files
        num_queries: Total number of queries (default 1000)
        num_runs: Number of runs per query (default 3)
        exclude_warmup: If True, exclude warmup runs from output
        spark_config: Spark configuration dict with standard_units settings

    Returns:
        DataFrame with columns: query, run, and extracted metrics (standardized units)
    """
    data_path = Path(data_dir)

    # Find all event log files (files without extension in Spark directories)
    event_logs = [f for f in data_path.iterdir() if f.is_file() and not f.name.startswith(".")]

    if not event_logs:
        print(f"WARNING: No event logs found in {data_dir}")
        return pd.DataFrame()

    print(f"\n{'='*70}")
    print(f"Spark Engine: {data_path.name}")
    print(f"{'='*70}")
    print(f"Found {len(event_logs)} event log file(s)")

    # Parse all event logs and combine results
    all_dfs = []

    for event_log in sorted(event_logs):
        print(f"\nProcessing: {event_log.name}")

        # Use spark_event_parser to extract metrics
        # Parse to a temporary CSV file with unique naming
        import tempfile
        import os

        # Use process ID and counter for unique temp file names
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".csv", prefix=f"spark_etl_{os.getpid()}_")
        os.close(tmp_fd)  # Close file descriptor immediately

        try:
            spark_event_parser.parse_event_log(str(event_log), tmp_path, exclude_warmup=exclude_warmup)

            # Read the generated CSV
            df = pd.read_csv(tmp_path)

            if len(df) > 0:
                all_dfs.append(df)
                print(f"  Extracted {len(df)} query runs")
            else:
                print("   No query runs found in this log")

        except Exception as e:
            print(f"  Error processing {event_log.name}: {e}")
            # Continue processing other files even if one fails
        finally:
            # Clean up temp file - guaranteed to run even on exception
            if os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError as e:
                    print(f"  Warning: Could not delete temp file {tmp_path}: {e}")

    if not all_dfs:
        print("\n No data extracted from any event logs")
        return pd.DataFrame()

    # Combine all DataFrames and remove duplicates (same run_id may appear in multiple event logs)
    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df = combined_df.drop_duplicates(subset=["run_id"], keep="first")

    # Extract query and run numbers from run_id (format: q{N}_run{M})
    # Filter out rows where extraction failed (invalid run_id format)
    combined_df["query"] = pd.to_numeric(combined_df["run_id"].str.extract(r"q(\d+)_")[0], errors="coerce")
    combined_df["run"] = pd.to_numeric(combined_df["run_id"].str.extract(r"run(\d+)")[0], errors="coerce")

    # Drop rows with invalid run_id format (couldn't extract query or run) - usually the warmup runs
    before_filter = len(combined_df)
    combined_df = combined_df.dropna(subset=["query", "run"])
    if len(combined_df) < before_filter:
        print(f"  Dropped {before_filter - len(combined_df)} rows with invalid run_id format")

    # Now safe to convert to int
    combined_df["query"] = combined_df["query"].astype(int)
    combined_df["run"] = combined_df["run"].astype(int)

    # Create complete grid of expected query/run combinations
    all_queries = list(range(1, num_queries + 1))
    all_runs = list(range(1, num_runs + 1))

    # Build expected combinations
    import itertools

    expected_combinations = pd.DataFrame(list(itertools.product(all_queries, all_runs)), columns=["query", "run"])

    # Merge to identify missing queries
    merged_df = expected_combinations.merge(combined_df, on=["query", "run"], how="left")

    # Convert failed queries to empty rows (keep only query, run, status, run_id, but null out all metrics)
    if "status" in merged_df.columns:
        failed_mask = merged_df["status"] == "failed"
        # For failed queries, null out all columns except query, run, status, and run_id
        cols_to_null = [col for col in merged_df.columns if col not in ["query", "run", "status", "run_id"]]
        merged_df.loc[failed_mask, cols_to_null] = None

        # Mark as empty/failed in run_id for identification (status remains "failed")
        merged_df.loc[failed_mask, "run_id"] = "EMPTY/FAILED"

    # For completely missing queries, mark as empty
    missing_mask = merged_df["run_id"].isna()
    merged_df.loc[missing_mask, "run_id"] = "EMPTY/FAILED"

    # Reorder columns: query, run first, then rest
    cols = ["query", "run"] + [col for col in merged_df.columns if col not in ["query", "run"]]
    merged_df = merged_df[cols]

    # Sort by query and run
    merged_df = merged_df.sort_values(["query", "run"]).reset_index(drop=True)

    # Standardize units if config provided
    if spark_config is not None:
        merged_df = standardize_spark_metrics(merged_df, spark_config)

        # Rename columns to be unit-agnostic (remove _ms, _ns, _bytes suffixes)
        columns_to_rename = {old: new for old, new in COLUMN_RENAMES.items() if old in merged_df.columns}
        if columns_to_rename:
            merged_df = merged_df.rename(columns=columns_to_rename)
            print(f"  ✓ Renamed {len(columns_to_rename)} column(s) to unit-agnostic names")

    # Calculate statistics with validation
    total_expected = num_queries * num_runs
    successful = len(merged_df[merged_df["status"] == "success"]) if "status" in merged_df.columns else 0
    failed = len(merged_df[merged_df["run_id"] == "EMPTY/FAILED"])

    print(f"\n{'='*70}")
    print(f"Summary for {data_path.name}:")
    print(f"{'='*70}")
    print(f"Total expected runs: {total_expected}")
    print(f"Total processed: {len(merged_df)}")
    print(f"Successful runs: {successful}")
    print(f"Failed/Missing runs: {failed}")

    # Data integrity check
    total_accounted = successful + failed
    if total_accounted != total_expected:
        print(f"⚠ WARNING: Mismatch! Expected {total_expected}, got {total_accounted}")
        print(f"  Difference: {total_expected - total_accounted}")
    else:
        print(f"✓ All queries accounted for")

    if total_expected > 0:
        print(f"\nSuccess rate: {successful/total_expected*100:.1f}%")
    print(f"{'='*70}\n")

    return merged_df


def save_metrics(df: pd.DataFrame, output_path: str) -> None:
    """Save metrics DataFrame to CSV."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"Saved metrics to: {output_file}")


if __name__ == "__main__":
    """Standalone testing interface."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python etl_spark.py <data_dir> [output_csv]")
        print("Example: python etl_spark.py ../spark-w4 output.csv")
        sys.exit(1)

    data_dir = sys.argv[1]
    output_csv = sys.argv[2] if len(sys.argv) > 2 else "spark_metrics.csv"

    df = process_spark_engine(data_dir, num_queries=2000, num_runs=3, exclude_warmup=True)

    if len(df) > 0:
        save_metrics(df, output_csv)
        print(f"\n Processing complete! Shape: {df.shape}")
    else:
        print("\n No data to save")
