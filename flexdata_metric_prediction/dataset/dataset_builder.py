"""
Dataset Builder

Simple class to build QueryDataset from CSV files, SQL queries, and query plans.
Handles multi-engine, multi-metric measurements.
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd

from flexdata_metric_prediction.dataset.query_datapoint import QueryDatapoint
from flexdata_metric_prediction.dataset.query_dataset import QueryDataset


class DatasetBuilder:
    """Build QueryDataset from CSV measurements, SQL, and plans."""

    def __init__(
        self,
        plans_dir: str | Path,
        queries_file: str | Path,
        metrics_dir: str | Path,
        schema: str = "tpch",
    ):
        """Initialize the dataset builder.

        Args:
            plans_dir: Directory containing query plan JSON files (e.g., query_1_plan.json)
            queries_file: Path to JSON file with SQL queries (list of strings)
            metrics_dir: Directory containing CSV files with metrics (e.g., tpch_presto-w1.csv)
            schema: Schema name (default: "tpch")
        """
        self.plans_dir = Path(plans_dir)
        self.queries_file = Path(queries_file)
        self.metrics_dir = Path(metrics_dir)
        self.schema = schema

    def load_sql_queries(self) -> List[str]:
        """Load SQL queries from JSON file.

        Returns:
            List of SQL query strings
        """
        with open(self.queries_file, "r") as f:
            queries = json.load(f)
        return queries

    def load_plan(self, query_num: int) -> dict:
        """Load query plan from JSON file with Substrait compatibility fixes.

        Args:
            query_num: Query number (1-indexed)

        Returns:
            Query plan as dict (will be converted to Substrait Plan by QueryDatapoint)
        """
        plan_path = self.plans_dir / f"query_{query_num}_plan.json"

        if not plan_path.exists():
            raise FileNotFoundError(f"Plan not found: {plan_path}")

        with open(plan_path, "r") as f:
            plan_text = f.read()

        # Fix Substrait compatibility: Remove deprecated field names
        # Old plans use "Urn" but new Substrait schema expects "Uri"
        # These fixes are applied BEFORE parsing to ensure compatibility
        plan_text = re.sub(r'"extensionUrns"', '"extensionUris"', plan_text)
        plan_text = re.sub(r'"extensionUrnAnchor"', '"extensionUriAnchor"', plan_text)
        plan_text = re.sub(r'"extensionUrnReference"', '"extensionUriReference"', plan_text)
        plan_text = re.sub(r'"urn"(\s*:\s*")', r'"uri"\1', plan_text)

        # Return as dict - QueryDatapoint will convert to Substrait Plan protobuf
        return json.loads(plan_text)

    def load_engine_metrics(self, engine_name: str, metric_cols: List[str] | None = None) -> pd.DataFrame:
        """Load metrics for a single engine from CSV.

        Args:
            engine_name: Engine name (e.g., "presto-w1")
            metric_cols: List of metric columns to extract. If None, auto-detects all numeric columns.

        Returns:
            DataFrame with 'query' column and metric columns
        """
        csv_path = self.metrics_dir / f"{self.schema}_{engine_name}.csv"

        if not csv_path.exists():
            logging.warning(f"Metrics file not found: {csv_path}")
            return pd.DataFrame()

        df = pd.read_csv(csv_path)

        # Auto-detect metric columns if not specified
        if metric_cols is None:
            exclude_cols = {"query", "run", "schema", "query_id", "state", "status", "run_id", "run_type"}
            metric_cols = [
                col for col in df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])
            ]

        # Aggregate by query (median across runs)
        if "query" in df.columns:
            agg_dict = {col: "median" for col in metric_cols}
            df = df.groupby("query").agg(agg_dict).reset_index()

        return df

    def build_dataset(
        self,
        engine_configs: Dict[str, List[str] | None],
        query_range: tuple[int, int] | None = None,
        dataset_name: str | None = None,
    ) -> QueryDataset:
        """Build a QueryDataset with multi-engine measurements.

        Args:
            engine_configs: Dict mapping engine_name -> list of metric columns (or None for auto-detect)
                Example: {"presto-w1": ["elapsedTime", "peakNodeTotalMemory"],
                         "spark-w1": None}  # Auto-detect all metrics
            query_range: Optional tuple (start, end) to limit queries (1-indexed, inclusive)
            dataset_name: Optional name for the dataset

        Returns:
            QueryDataset with QueryDatapoints containing SQL, plans, and multi-engine measurements
        """
        # Load SQL queries
        sql_queries = self.load_sql_queries()

        # Determine query range
        if query_range is None:
            start_query, end_query = 1, len(sql_queries)
        else:
            start_query, end_query = query_range

        # Load metrics for all engines
        engine_metrics = {}
        for engine_name, metric_cols in engine_configs.items():
            df = self.load_engine_metrics(engine_name, metric_cols)
            if not df.empty:
                engine_metrics[engine_name] = df
            else:
                logging.warning(f"No metrics loaded for engine: {engine_name}")

        if not engine_metrics:
            raise ValueError("No metrics loaded for any engine")

        # Build QueryDatapoints
        datapoints = []
        queries_with_data = 0

        for query_num in range(start_query, end_query + 1):
            # Get SQL
            sql = sql_queries[query_num - 1]  # 0-indexed in list

            # Load plan
            try:
                plan = self.load_plan(query_num)
            except FileNotFoundError:
                logging.warning(f"No plan found for query {query_num}, skipping")
                continue

            # Collect measurements from all engines
            measurements = {}
            has_measurements = False

            for engine_name, metrics_df in engine_metrics.items():
                # Find measurements for this query
                query_data = metrics_df[metrics_df["query"] == query_num]

                if not query_data.empty:
                    # Extract metrics for this engine
                    engine_measurements = {}
                    for col in query_data.columns:
                        if col != "query":
                            value = query_data[col].iloc[0]
                            if pd.notna(value):
                                engine_measurements[col] = float(value)

                    if engine_measurements:
                        measurements[engine_name] = engine_measurements
                        has_measurements = True

            # Create QueryDatapoint
            if has_measurements:
                query_id = f"{self.schema}_q{query_num}"
                datapoint = QueryDatapoint(
                    sql=sql,
                    id=query_id,
                    plan=plan,
                    measurements=measurements,
                )
                datapoints.append(datapoint)
                queries_with_data += 1
            else:
                logging.info(f"No measurements found for query {query_num}, skipping")

        # Create QueryDataset
        if dataset_name is None:
            dataset_name = f"{self.schema} dataset ({queries_with_data} queries)"

        dataset = QueryDataset(queries=datapoints, name=dataset_name)

        logging.info(f"Built dataset: {dataset}")
        logging.info(f"  Queries: {len(datapoints)}")
        logging.info(f"  Engines: {list(engine_metrics.keys())}")

        return dataset
