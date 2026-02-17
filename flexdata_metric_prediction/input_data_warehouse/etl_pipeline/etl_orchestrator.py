"""
ETL Orchestrator

Coordinates ETL for Presto and Spark engines across multiple schemas (TPCH, TPCDS, etc.)
and produces per-schema, per-engine metric CSVs.

Output format: {schema}_{engine}.csv (e.g., tpch_presto-w1.csv, tpch_spark-w4.csv)
"""

import sys
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, List

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
import etl_presto
import etl_spark
import pandas as pd


def load_config(config_path: Path) -> Dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def validate_metrics(
    df: pd.DataFrame,
    time_columns: List[str],
    memory_columns: List[str],
    engine_type: str = "presto",  # "presto" or "spark"
) -> pd.DataFrame:
    """Validate metrics and report statistics. NO transformation applied.

    ETL outputs RAW values. Log transformation, imputation, and standardization
    are handled by MetricNormalizer at training time.

    This function only:
    - Reports success/failure counts
    - Warns about zeros/negatives in successful runs
    - Sets failed runs to NaN (they will be filtered in GNNDataset)

    Args:
        df: DataFrame with metrics
        time_columns: List of time metric column names
        memory_columns: List of memory metric column names
        engine_type: "presto" or "spark" to determine success column

    Returns:
        DataFrame with metrics in RAW space (NaN for failed runs)
    """
    df = df.copy()

    # Determine success mask based on engine type
    if engine_type == "presto":
        if "state" in df.columns:
            success_mask = df["state"] == "FINISHED"
        else:
            print(f"  ⚠ Warning: No 'state' column found, processing all rows")
            success_mask = pd.Series(True, index=df.index)
    else:  # spark
        if "status" in df.columns:
            success_mask = df["status"] == "success"
        elif "run_id" in df.columns:
            # Fallback: exclude EMPTY/FAILED
            success_mask = df["run_id"] != "EMPTY/FAILED"
        else:
            print(f"  ⚠ Warning: No 'status' or 'run_id' column found, processing all rows")
            success_mask = pd.Series(True, index=df.index)

    num_successful = success_mask.sum()
    num_failed = (~success_mask).sum()
    print(f"  Success: {num_successful} runs, Failed/Missing: {num_failed} runs")

    all_metric_cols = time_columns + memory_columns

    for col in all_metric_cols:
        if col not in df.columns:
            continue

        # Only look at successful runs for statistics
        successful_data = df.loc[success_mask, col]

        # Report zeros/negatives (will be handled by MetricNormalizer)
        zeros_in_success = (successful_data == 0).sum()
        negatives_in_success = (successful_data < 0).sum()

        if zeros_in_success > 0:
            print(f"  ℹ {col}: {zeros_in_success} zeros in successful runs")

        if negatives_in_success > 0:
            print(f"  ⚠ {col}: {negatives_in_success} negatives in successful runs")

        # Report value range for sanity check
        valid_data = successful_data[successful_data.notna() & (successful_data > 0)]
        if len(valid_data) > 0:
            print(
                f"  ℹ {col}: range [{valid_data.min():.2f}, {valid_data.max():.2f}], median={valid_data.median():.2f}"
            )

    return df


def process_schema_presto_engine(
    schema_name: str,
    engine_config: Dict,
    schema_config: Dict,
    presto_config: Dict,
    script_dir: Path,
    global_config: Dict,
) -> pd.DataFrame:
    """Process a single Presto engine for a specific schema.

    Args:
        schema_name: Schema name (e.g., 'tpch')
        engine_config: Engine configuration dict
        schema_config: Schema-level configuration
        presto_config: Presto-specific metrics config
        script_dir: Base directory for relative paths
        global_config: Global configuration

    Returns:
        DataFrame with schema, query, run, query_id, and metrics columns
    """
    engine_name = engine_config["name"]
    raw_metrics_base = script_dir / schema_config["raw_metrics_dir"]
    data_dir = raw_metrics_base / engine_config["data_dir"]

    print(f"\n{'='*70}")
    print(f"Processing: {schema_name} / {engine_name}")
    print(f"{'='*70}")
    print(f"Data directory: {data_dir}")

    # Process using etl_presto
    df = etl_presto.process_presto_engine(
        str(data_dir),
        presto_config["queryStats_fields"],
        num_queries=global_config["num_queries"],
        num_runs=global_config["num_runs"],
    )

    if len(df) == 0:
        print(f"⚠ No data extracted for {schema_name}/{engine_name}")
        return pd.DataFrame()

    # Add schema and query_id columns
    df.insert(0, "schema", schema_name)
    df["query_id"] = df.apply(lambda row: f"{schema_name}_q{int(row['query'])}_run{int(row['run'])}", axis=1)

    # Reorder: schema, query, run, query_id, then all other columns
    cols = ["schema", "query", "run", "query_id"] + [
        col for col in df.columns if col not in ["schema", "query", "run", "query_id"]
    ]
    df = df[cols]

    return df


def process_schema_spark_engine(
    schema_name: str,
    engine_config: Dict,
    schema_config: Dict,
    spark_config: Dict,
    script_dir: Path,
    global_config: Dict,
) -> pd.DataFrame:
    """Process a single Spark engine for a specific schema.

    Args:
        schema_name: Schema name (e.g., 'tpch')
        engine_config: Engine configuration dict
        schema_config: Schema-level configuration
        spark_config: Spark-specific configuration
        script_dir: Base directory for relative paths
        global_config: Global configuration

    Returns:
        DataFrame with schema, query, run, query_id, and metrics columns
    """
    engine_name = engine_config["name"]
    raw_metrics_base = script_dir / schema_config["raw_metrics_dir"]
    data_dir = raw_metrics_base / engine_config["data_dir"]

    print(f"\n{'='*70}")
    print(f"Processing: {schema_name} / {engine_name}")
    print(f"{'='*70}")
    print(f"Data directory: {data_dir}")

    # Process using etl_spark
    df = etl_spark.process_spark_engine(
        str(data_dir),
        num_queries=global_config["num_queries"],
        num_runs=global_config["num_runs"],
        exclude_warmup=True,
        spark_config=spark_config,
    )

    if len(df) == 0:
        print(f"⚠ No data extracted for {schema_name}/{engine_name}")
        return pd.DataFrame()

    # Add schema and query_id columns
    df.insert(0, "schema", schema_name)
    df["query_id"] = df.apply(lambda row: f"{schema_name}_q{int(row['query'])}_run{int(row['run'])}", axis=1)

    # Reorder: schema, query, run, query_id, then all other columns
    cols = ["schema", "query", "run", "query_id"] + [
        col for col in df.columns if col not in ["schema", "query", "run", "query_id"]
    ]
    df = df[cols]

    return df


def process_schema(
    schema_name: str,
    schema_config: Dict,
    config: Dict,
    script_dir: Path,
    output_dir: Path,
) -> Dict[str, pd.DataFrame]:
    """Process all engines for a specific schema.

    Args:
        schema_name: Schema name (e.g., 'tpch')
        schema_config: Schema configuration
        config: Full configuration dict
        script_dir: Base script directory
        output_dir: Output directory for processed metrics

    Returns:
        Dictionary mapping engine names to DataFrames
    """
    print(f"\n{'#'*70}")
    print(f"# PROCESSING SCHEMA: {schema_name.upper()}")
    print(f"{'#'*70}")

    results = {}
    global_config = config["global"]

    # Process Presto engines
    if "presto" in schema_config["engines"]:
        print(f"\n--- Presto Engines ---")
        presto_config = config["presto"]
        for engine_config in schema_config["engines"]["presto"]:
            df = process_schema_presto_engine(
                schema_name,
                engine_config,
                schema_config,
                presto_config,
                script_dir,
                global_config,
            )
            if len(df) > 0:
                engine_name = engine_config["name"]

                # Validate metrics (no transformation - raw values preserved)
                print(f"\n  Validating metrics for {engine_name}...")
                time_cols = ["elapsedTime", "queuedTime", "executionTime", "planningTime"]
                memory_cols = ["peakNodeTotalMemory", "peakUserMemoryReservation", "peakTotalMemoryReservation"]
                df = validate_metrics(df, time_cols, memory_cols, engine_type="presto")

                results[engine_name] = df

                # Save individual engine CSV
                output_file = output_dir / f"{schema_name}_{engine_name}.csv"
                df.to_csv(output_file, index=False)
                print(f"✓ Saved: {output_file}")

    # Process Spark engines
    if "spark" in schema_config["engines"]:
        print(f"\n--- Spark Engines ---")
        spark_config = config["spark"]
        for engine_config in schema_config["engines"]["spark"]:
            df = process_schema_spark_engine(
                schema_name,
                engine_config,
                schema_config,
                spark_config,
                script_dir,
                global_config,
            )
            if len(df) > 0:
                engine_name = engine_config["name"]

                # Validate metrics (no transformation - raw values preserved)
                print(f"\n  Validating metrics for {engine_name}...")
                time_cols = ["wall_clock_duration", "executor_run_time", "jvm_gc_time", "executor_cpu_time"]
                memory_cols = [
                    "on_heap_execution_memory",
                    "on_heap_storage_memory",
                    "peak_execution_memory",
                    "jvm_heap_memory",
                    "memory_spilled",
                ]
                df = validate_metrics(df, time_cols, memory_cols, engine_type="spark")

                results[engine_name] = df

                # Save individual engine CSV
                output_file = output_dir / f"{schema_name}_{engine_name}.csv"
                df.to_csv(output_file, index=False)
                print(f"✓ Saved: {output_file}")

    return results


def generate_schema_summary(
    schema_name: str,
    engine_dfs: Dict[str, pd.DataFrame],
    output_dir: Path,
) -> None:
    """Generate summary report for a schema.

    Args:
        schema_name: Schema name
        engine_dfs: Dictionary of engine_name -> DataFrame
        output_dir: Output directory
    """
    if not engine_dfs:
        print(f"\n⚠ No data for schema {schema_name}")
        return

    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append(f"SCHEMA: {schema_name.upper()} - SUMMARY REPORT")
    report_lines.append("=" * 70)

    for engine_name, df in engine_dfs.items():
        report_lines.append(f"\n--- {engine_name} ---")
        report_lines.append(f"Total runs: {len(df)}")
        report_lines.append(f"Unique queries: {df['query'].nunique()}")

        # Check for success/failure metrics
        if "state" in df.columns:  # Presto
            successful = len(df[df["state"] == "FINISHED"])
            # Failed includes: FAILED, MISSING, EMPTY/FAILED, PARSE_ERROR
            failed = len(df[df["state"] == "FAILED"])
            missing = len(df[df["state"] == "MISSING"])
            empty_failed = len(df[df["state"] == "EMPTY/FAILED"])
            parse_errors = len(df[df["state"] == "PARSE_ERROR"])

            # Other states (QUEUED, RUNNING, etc.)
            other_states = len(df[~df["state"].isin(["FINISHED", "FAILED", "MISSING", "EMPTY/FAILED", "PARSE_ERROR"])])

            total_failed = failed + missing + empty_failed + parse_errors

            report_lines.append(f"Successful: {successful}")
            report_lines.append(f"Failed: {failed}")
            report_lines.append(f"Missing: {missing}")
            report_lines.append(f"Empty/Failed: {empty_failed}")
            report_lines.append(f"Parse Errors: {parse_errors}")
            if other_states > 0:
                report_lines.append(f"Other States: {other_states}")
            report_lines.append(f"Total Failed/Missing/Errors: {total_failed}")
            if len(df) > 0:
                report_lines.append(f"Success rate: {successful/len(df)*100:.1f}%")

        if "status" in df.columns:  # Spark
            successful = len(df[df["status"] == "success"])
            # For Spark, all failed/missing queries have run_id="EMPTY/FAILED"
            # (etl_spark.py sets this for both status="failed" and missing queries)
            failed = len(df[df["run_id"] == "EMPTY/FAILED"]) if "run_id" in df.columns else 0
            report_lines.append(f"Successful: {successful}")
            report_lines.append(f"Failed/Missing: {failed}")
            if len(df) > 0:
                report_lines.append(f"Success rate: {successful/len(df)*100:.1f}%")

    report_lines.append("=" * 70)

    report_text = "\n".join(report_lines)
    print(report_text)

    # Save to file
    report_file = output_dir / f"{schema_name}_summary.txt"
    with open(report_file, "w") as f:
        f.write(report_text)
    print(f"\nSummary saved to: {report_file}")


def main(config_path: str = "ingestion_config.yaml") -> None:
    """Main orchestration function - processes all schemas and their engines."""
    script_dir = Path(__file__).parent
    config_file = script_dir / config_path

    print("=" * 70)
    print("ETL ORCHESTRATOR - Schema-Aware Processing")
    print("=" * 70)

    # 1. Load config
    config = load_config(config_file)
    print(f"\nLoaded configuration from: {config_file}")

    # 2. Setup output directory
    output_dir = script_dir / Path(config["global"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # 3. Process each schema
    schemas_to_process = config["global"]["schemas"]
    all_schema_results = {}

    for schema_name in schemas_to_process:
        if schema_name not in config["schemas"]:
            print(f"\n⚠ Warning: Schema '{schema_name}' not found in config, skipping")
            continue

        schema_config = config["schemas"][schema_name]
        engine_dfs = process_schema(
            schema_name,
            schema_config,
            config,
            script_dir,
            output_dir,
        )

        all_schema_results[schema_name] = engine_dfs

        # Generate summary for this schema
        generate_schema_summary(schema_name, engine_dfs, output_dir)

    # 4. Final summary
    print("\n" + "=" * 70)
    print("ETL ORCHESTRATOR - Complete!")
    print("=" * 70)
    print(f"\nProcessed {len(all_schema_results)} schema(s)")
    for schema_name, engine_dfs in all_schema_results.items():
        print(f"  {schema_name}: {len(engine_dfs)} engine(s)")
    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
