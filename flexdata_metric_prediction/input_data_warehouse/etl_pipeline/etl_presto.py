"""
Presto ETL Script

Extracts metrics from Presto JSON files and standardizes units.
[Default] All time metrics -> seconds, all size metrics -> MB.
"""

import json
import re
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import pandas as pd


UNIT_CONVERSIONS = {
    "ns": 1e-9,
    "us": 1e-6,
    "ms": 1e-3,
    "s": 1.0,
    "m": 60.0,
    "h": 3600.0,
    "d": 86400.0,
    "B": 1 / (1024**2),
    "kB": 1 / 1024,
    "KB": 1 / 1024,
    "MB": 1.0,
    "GB": 1024.0,
    "TB": 1024**2,
}


def parse_value_with_unit(value_str: Any) -> Tuple[Optional[float], str]:
    """Parse '21.71s' -> (21.71, 's'), handle non-strings gracefully.

    Raises:
        ValueError: If value_str cannot be parsed as a number with optional unit
    """
    if not isinstance(value_str, str):
        if isinstance(value_str, (int, float)):
            return value_str, ""
        raise ValueError(f"Cannot parse value of type {type(value_str).__name__}: {value_str}")

    match = re.match(r"^([0-9]*\.?[0-9]+)([a-zA-Z]*)$", value_str.strip())
    if match:
        return float(match.group(1)), match.group(2)

    try:
        return float(value_str), ""
    except (ValueError, TypeError) as e:
        raise ValueError(f"Cannot parse value '{value_str}' as a number with optional unit: {e}") from e


def convert_to_standard_unit(value: float, source_unit: str, target_unit: str) -> float:
    """Convert value from source_unit to target_unit using UNIT_CONVERSIONS.

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


def extract_metrics_from_json(json_path: Path, metrics_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract and standardize metrics from a single Presto JSON file."""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Check if file is essentially empty (failed query)
        if not data or "queryStats" not in data:
            return None

        query_stats = data["queryStats"]
        metrics = {}

        # Add top-level state
        metrics["state"] = data.get("state", "UNKNOWN")

        # Extract queryStats fields based on config
        for field_name, config in metrics_config.items():
            if not config.get("extract", False):
                continue

            if field_name not in query_stats:
                metrics[field_name] = None
                continue

            value_str = query_stats[field_name]

            try:
                value, source_unit = parse_value_with_unit(value_str)
            except ValueError as e:
                raise ValueError(f"Failed to parse field '{field_name}' in {json_path.name}: {e}") from e

            # Convert to standard unit if specified
            target_unit = config.get("standard_unit")
            if target_unit and source_unit:
                try:
                    value = convert_to_standard_unit(value, source_unit, target_unit)
                except ValueError as e:
                    raise ValueError(
                        f"Failed to convert field '{field_name}' from '{source_unit}' to '{target_unit}' "
                        f"in {json_path.name}: {e}"
                    ) from e

            metrics[field_name] = value

        return metrics

    except json.JSONDecodeError:
        return None  # Empty or corrupted file
    except ValueError:
        # Re-raise ValueError from parsing/conversion - these are bugs we need to fix
        raise
    except Exception as e:
        # Unexpected errors - log and treat as failed
        print(f"Unexpected error processing {json_path}: {type(e).__name__}: {e}")
        raise


def process_presto_engine(
    data_dir: str,
    metrics_config: Dict[str, Any],
    num_queries: int = 1000,
    num_runs: int = 3,
) -> pd.DataFrame:
    """
    Process all JSON files for a Presto engine.

    Args:
        data_dir: Directory containing JSON files (e.g., '../presto-w1')
        metrics_config: Dict of {field_name: {extract: bool, standard_unit: str}}
        num_queries: Total number of queries (default 1000)
        num_runs: Number of runs per query (default 3)

    Returns:
        DataFrame with columns: query, run, and extracted metrics

    """
    data_path = Path(data_dir)
    file_pattern = re.compile(r"^q(\d+)_run(\d+)\.json$")

    records = []
    missing_files = []
    empty_files = []
    parse_errors = []
    state_counts = {}  # Track all state values

    # Iterate through expected query-run combinations
    for query_num in range(1, num_queries + 1):
        for run_num in range(1, num_runs + 1):
            file_name = f"q{query_num}_run{run_num}.json"
            file_path = data_path / file_name

            record = {"query": query_num, "run": run_num}

            # Check if file exists
            if not file_path.exists():
                missing_files.append((query_num, run_num))
                # Add row with all None values
                for field_name in metrics_config:
                    if metrics_config[field_name].get("extract", False):
                        record[field_name] = None
                record["state"] = "MISSING"
            else:
                # Extract metrics
                try:
                    metrics = extract_metrics_from_json(file_path, metrics_config)

                    if metrics is None:
                        empty_files.append((query_num, run_num))
                        for field_name in metrics_config:
                            if metrics_config[field_name].get("extract", False):
                                record[field_name] = None
                        record["state"] = "EMPTY/FAILED"
                    else:
                        record.update(metrics)
                        # Track state distribution
                        state = metrics.get("state", "UNKNOWN")
                        state_counts[state] = state_counts.get(state, 0) + 1
                except (ValueError, Exception) as e:
                    # Parse/conversion error - this is a bug we need to fix
                    parse_errors.append((query_num, run_num, str(e)))
                    for field_name in metrics_config:
                        if metrics_config[field_name].get("extract", False):
                            record[field_name] = None
                    record["state"] = "PARSE_ERROR"

            records.append(record)

    # Print summary with comprehensive validation
    total_expected = num_queries * num_runs
    successful = state_counts.get("FINISHED", 0)
    failed = state_counts.get("FAILED", 0)

    print(f"\n{'='*60}")
    print(f"Presto Engine: {data_path.name}")
    print(f"{'='*60}")
    print(f"Total expected queries: {total_expected}")
    print(f"Total processed: {len(records)}")

    # State breakdown
    print(f"\nState Distribution:")
    print(f"  FINISHED (successful): {successful}")
    print(f"  FAILED: {failed}")
    print(f"  MISSING: {len(missing_files)}")
    print(f"  EMPTY/FAILED: {len(empty_files)}")
    print(f"  PARSE_ERROR: {len(parse_errors)}")

    # Show other states if any
    other_states = {k: v for k, v in state_counts.items() if k not in ["FINISHED", "FAILED"]}
    if other_states:
        print(f"\n  Other states:")
        for state, count in sorted(other_states.items()):
            print(f"    {state}: {count}")

    # Data integrity checks
    total_accounted = successful + failed + len(missing_files) + len(empty_files) + len(parse_errors)
    print(f"\nData Integrity:")
    print(f"  Total accounted for: {total_accounted}")
    if total_accounted != total_expected:
        print(f"  ⚠ WARNING: Mismatch! Expected {total_expected}, got {total_accounted}")
        print(f"  Difference: {total_expected - total_accounted}")
    else:
        print(f"  ✓ All queries accounted for")

    # Success rate
    if total_expected > 0:
        print(f"\nSuccess rate: {successful/total_expected*100:.1f}%")

    # Show problematic files
    if missing_files:
        print(f"\nMissing files (first 10): {missing_files[:10]}")
    if empty_files:
        print(f"Empty/Failed files (first 10): {empty_files[:10]}")
    if parse_errors:
        print(f"\n⚠ Parse errors (first 5):")
        for query_num, run_num, error in parse_errors[:5]:
            print(f"  q{query_num}_run{run_num}: {error}")
    print(f"{'='*60}\n")

    # Create DataFrame
    df = pd.DataFrame(records)
    return df


def save_metrics(df: pd.DataFrame, output_path: str) -> None:
    """Save metrics DataFrame to CSV."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"Saved metrics to: {output_file}")


if __name__ == "__main__":
    # Example usage
    presto_data_dir = "../raw_metrics/tpch/presto-w4"
    output_csv = "test_presto_w4_metrics.csv"

    # Define which metrics to extract and their standard units
    metrics_config = {
        "elapsedTime": {"extract": True, "standard_unit": "s"},
        "executionTime": {"extract": True, "standard_unit": "s"},
        "totalScheduledTime": {"extract": True, "standard_unit": "s"},
        "totalCpuTime": {"extract": True, "standard_unit": "s"},
        "peakUserMemoryReservation": {"extract": True, "standard_unit": "MB"},
        "peakTotalMemoryReservation": {"extract": True, "standard_unit": "MB"},
        "peakNodeTotalMemory": {"extract": True, "standard_unit": ""},
        "processedInputDataSize": {"extract": True, "standard_unit": "MB"},
        "rawInputDataSize": {"extract": True, "standard_unit": "MB"},
    }

    df = process_presto_engine(presto_data_dir, metrics_config, num_queries=1000, num_runs=3)
    if len(df) > 0:
        save_metrics(df, output_csv)
        print(f"\n Processing complete! Shape: {df.shape}")
    else:
        print("\n No data to save")
