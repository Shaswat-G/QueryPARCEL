import json
import logging
from datetime import datetime
from typing import Any

import substrait.json
from substrait.proto import Plan

from flexdata_metric_prediction.tree.tree import Tree


class QueryDatapoint:
    """Stores a query in possibly multiple representations along with its basic properties."""

    def __init__(
        self,
        sql: str | None = None,
        id: str | None = None,
        plan: Any = None,
        tree: Tree | None = None,
        measurements: dict[str, float | dict[str, float]] | None = None,
        statistics: dict | None = None,
        query_res_empty: bool | None = None,
    ) -> None:
        """Constructor.

        Args:
            sql (str | None, optional): SQL statement. Defaults to None.
            id (str | None, optional): query ID. If None, a unique ID will be generated. Defaults to None.
            plan (Any, optional): Substrait plan. Defaults to None.
            tree (Tree | None, optional): A query's internal tree representation. Defaults to None.
            measurements (dict | None, optional): Dictionary mapping engine names to measurements.
                Can be either:
                - dict[engine, float]: Single metric (execution time) - backward compatible
                - dict[engine, dict[metric, float]]: Multi-metric per engine
                Examples:
                    {"presto_w1": 12.5, "spark_w1": 15.2}  # Single metric (time)
                    {"presto_w1": {"time": 12.5, "cpu": 45.3, "memory": 2048}, ...}  # Multi-metric
                Defaults to None.
            statistics (dict | None, optional): Basic statistics about the query. Defaults to None.
            query_res_empty (bool | None, optional): If query returns rows or not (bool). Defaults to None.

        Raises:
            AttributeError: Thrown if none of the possible representations was set for the query.
        """
        if sql is None and plan is None and tree is None:
            raise AttributeError("Either the SQL statement or a query plan must be passed to define a query datapoint.")

        self.plan = self._load_plan(plan)

        self.sql = sql
        self.tree = tree

        # Normalize measurements to always be dict[engine][metric] = value
        self.measurements = self._normalize_measurements(measurements or {})
        self.statistics = statistics or {}
        self.query_res_empty = query_res_empty

        if id is None:  # Generate Unique ID if not provided
            id = "query_" + datetime.now().strftime("%Y-%m-%d_%H:%M:%S.%f")
        self.id = id

    def init_tree(self, plan: Any = None, force: bool = False) -> None:
        """Create the query's tree representation from a decorated substrait plan.

        Args:
            plan (Any, optional): Optionally specify the substrait plan to be used. Defaults to None.
            force (bool, optional): If true, the tree representation is re-created even if the query already contains one. Defaults to False.

        Raises:
            RuntimeError: Thrown if no plan has been defined for the query.
        """
        if plan is None and self.plan is None:
            raise RuntimeError("A plan must first be defined to create its tree representation.")
        if plan is not None and self.plan is not None:
            logging.warning(
                "A plan has already been defined for this datapoint but a new plan was passed for tree creation. Using the new plan"
            )
            self.plan = self._load_plan(plan)

        if self.tree is not None and not force:
            logging.warning("Tree has already been computed for this query")
            return

        self.tree = Tree()
        self.tree.create_tree(self.plan)

    def append_measurements(
        self,
        measurements: dict[str, float | dict[str, float]],
        allow_overwrite: bool = False,
    ) -> None:
        """Append measurements to the query.

        Args:
            measurements (dict): Mapping between engines and corresponding measurements.
                Can be scalar (time only) or dict of metrics.
            allow_overwrite (bool, optional): Overwrite measurements for engines already stored for this query. Defaults to False.

        Raises:
            AttributeError: Throw if duplicate engine is found and `allow_overwrite` was not set.
        """
        # Normalize new measurements
        normalized_new = self._normalize_measurements(measurements)

        duplicate_engines = [engine for engine in normalized_new if engine in self.measurements]

        if 0 < len(duplicate_engines):
            if allow_overwrite:
                logging.info(f"Overwriting measurements for engines {duplicate_engines} for query: {self}")
            else:
                raise AttributeError(
                    f"Measurements for engine(s) {duplicate_engines} have already been included for query {self} and allow_overwrite was not set"
                )
        self.measurements.update(normalized_new)

    def append_stats(self, statistics: dict, allow_overwrite=False) -> None:
        """Append statistics to the query

        Args:
            statistics (dict): Dictionary containing statistics.
            allow_overwrite (bool, optional): Overwrite statistics for features already stored for this query. Defaults to False.

        Raises:
            AttributeError: Thrown is duplicate features are found and `allow_overwrite` was not set.
        """
        duplicate_features = [feature for feature in self.statistics if feature in statistics]
        if 0 < len(duplicate_features):
            if allow_overwrite:
                logging.info(f"Overwriting features {duplicate_features} for query: {self}")
            else:
                raise AttributeError(
                    f"Features {duplicate_features} have already been included for query {self} and allow_overwrite was not set"
                )
        self.statistics.update({feature: statistics[feature] for feature in statistics})

    def _normalize_measurements(self, measurements: dict) -> dict:
        """Normalize measurements to dict[engine][metric] = value format.

        Handles backward compatibility:
        - If measurements[engine] is a scalar, converts to {"time": value}
        - If measurements[engine] is already a dict, keeps as-is

        Args:
            measurements: Raw measurements dict

        Returns:
            Normalized measurements dict[engine][metric] = value
        """
        normalized = {}
        for engine, value in measurements.items():
            if isinstance(value, dict):
                # Already multi-metric format
                normalized[engine] = value
            else:
                # Single scalar value - assume it's execution time
                normalized[engine] = {"time": float(value)}
        return normalized

    def get_metric(self, engine: str, metric: str = "time") -> float | None:
        """Get a specific metric value for an engine.

        Args:
            engine: Engine name (e.g., "presto_w1")
            metric: Metric name (default: "time")

        Returns:
            Metric value or None if not found
        """
        return self.measurements.get(engine, {}).get(metric)

    def get_all_metrics(self, engine: str) -> dict[str, float]:
        """Get all metrics for a specific engine.

        Args:
            engine: Engine name

        Returns:
            Dict of metric_name -> value
        """
        return self.measurements.get(engine, {})

    def get_metric_names(self, engine: str | None = None) -> list[str]:
        """Get list of available metric names.

        Args:
            engine: Optional engine name. If None, returns union of all metrics across engines.

        Returns:
            List of metric names
        """
        if engine is not None:
            return list(self.measurements.get(engine, {}).keys())

        # Return union of all metrics across all engines
        all_metrics = set()
        for engine_metrics in self.measurements.values():
            all_metrics.update(engine_metrics.keys())
        return sorted(all_metrics)

    def _load_plan(self, plan):
        if isinstance(plan, str):
            ret = substrait.json.load_json(plan)
        elif isinstance(plan, bytes):
            ret = Plan()
            ret.ParseFromString(plan)
        elif isinstance(plan, dict):
            ret = substrait.json.parse_json(json.dumps(plan))
        else:
            ret = plan
        return ret

    def __str__(self) -> str:
        return self.id
