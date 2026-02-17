import logging
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd

from flexdata_metric_prediction.dataset.query_datapoint import QueryDatapoint


class QueryDataset:
    """
    Dataset containing a list of queries.
    In general, a dataset contains only queries that target the same schema (not enforced!).
    """

    def __init__(
        self,
        queries: List[QueryDatapoint],
        name: str | None = None,
    ) -> None:
        """Constructor.

        Args:
            queries (List[QueryDatapoint]): List of queries
            name (str | None, optional): Name of the dataset. Defaults to None.
        """
        self.queries = queries
        self.name = name
        if self.name is None:
            self.name = "Query dataset - " + datetime.now().strftime(
                "%Y-%m-%d_%H:%M:%S.%f"
            )
        self.id_to_query = {
            query.id: query for query in self.queries
        }  # TODO: inherit from dict correctly

    def get_query(self, query_id) -> QueryDatapoint:
        """Get a query by its ID.

        Args:
            query_id (_type_): Query ID

        Returns:
            QueryDatapoint: Query
        """
        return self.id_to_query[query_id]

    def get_measurements(
        self, skip_missing: bool = True, metric: str = "time", flatten: bool = False
    ) -> pd.DataFrame:
        """Get the measurements for each query in the dataset stored as a `pd.DataFrame`.
        The method ensures that each query contains either no measurements (if `skip_missing` is set) or measurements for the same set of engines.

        Args:
            skip_missing (bool, optional): If set, queries for which no measurements have been set are skipped. Defaults to True.
            metric (str, optional): Which metric to extract. Defaults to "time" for backward compatibility.
            flatten (bool, optional): If True, return all metrics in flat columns (engine_metric format).
                If False, return only the specified metric. Defaults to False.

        Raises:
            ValueError: Thrown if queries contain measurements for different sets of engines.
            AttributeError: Missing measurements for a query and `skip_missing=False`
            ValueError: None of the queries contained measurements.

        Returns:
            pd.DataFrame: Dataframe with a *query* column and:
                - If flatten=False: columns for each engine with the specified metric
                - If flatten=True: columns for each engine_metric combination
        """
        results = []
        engine_names = set()

        for query in self.queries:
            if 0 < len(query.measurements):
                engine_names.update(query.measurements.keys())
            elif skip_missing:
                logging.info(
                    f"Query {query.id} does not contain measurements. Skipping!"
                )
            else:
                raise AttributeError(
                    f"No measurement for query {str(query)}. Set skip_missing or provide measurements for each query"
                )

        engine_names = sorted(engine_names)
        if len(engine_names) == 0:
            raise ValueError("None of the queries contained measurements")

        if flatten:
            # Collect all unique metrics across all engines
            all_metrics = set()
            for query in self.queries:
                for engine in engine_names:
                    all_metrics.update(query.get_all_metrics(engine).keys())
            all_metrics = sorted(all_metrics)

            # Create column names: engine_metric
            columns = ["query"] + [
                f"{engine}_{metric}"
                for engine in engine_names
                for metric in all_metrics
            ]

            for query in self.queries:
                if len(query.measurements) > 0:
                    row = [query.id]
                    for engine in engine_names:
                        for metric_name in all_metrics:
                            value = query.get_metric(engine, metric_name)
                            row.append(value if value is not None else np.nan)
                    results.append(row)

            return pd.DataFrame(results, columns=columns)
        else:
            # Return single metric (backward compatible)
            for query in self.queries:
                if len(query.measurements) > 0:
                    row = [query.id] + [
                        (
                            query.get_metric(engine, metric)
                            if engine in query.measurements
                            else np.nan
                        )
                        for engine in engine_names
                    ]
                    results.append(row)

            return pd.DataFrame(results, columns=["query"] + engine_names)

    def get_statistics(self, skip_missing: bool = True) -> pd.DataFrame:
        """Get a dataframe containing the statistics for each query.
        The method ensures that queries in the dataset contain the same set of features.

        **NOTE**: These are not the features used for training.

        Args:
            skip_missing (bool, optional): Skip queries with missing statistics. Defaults to True.

        Raises:
            ValueError: All queries must contain statistics for the same set of features.
            AttributeError: A query without statistics is found and `skip_missing=False`
            ValueError: None of the queries contained statistics

        Returns:
            pd.DataFrame: Table containing query IDs under `query` and a column for each feature.
        """
        results = []
        feature_names = None
        for query in self.queries:
            if query.statistics is not None:
                if feature_names is None:
                    feature_names = query.statistics.keys()
                elif feature_names != query.statistics.keys():
                    raise ValueError(
                        "All queries must contain statistics for the same set of features."
                    )
            elif skip_missing:
                logging.info(f"Query {query.id} does not contain statistics. Skipping!")
            else:
                raise AttributeError(
                    f"No statistics for query {str(query)}. Set skip_missing or provide statistics for each query"
                )

        if feature_names is None:
            raise ValueError("None of the statistics contained statistics")

        for query in self.queries:
            if query.statistics is not None:
                results.append(
                    [query.id] + [query.statistics[stat] for stat in feature_names]
                )

        return pd.DataFrame(results, columns=["query"] + list(feature_names))

    def append_measurements_csv(
        self,
        measurements: pd.DataFrame,
        engine_names: List[str] | None = None,
        allow_overwrite: bool = False,
        multi_metric: bool = False,
    ) -> None:
        """Append each measurement from a CSV.

        Supports two CSV formats:
        1. Single metric (backward compatible): columns are query, engine1, engine2, ...
        2. Multi-metric: columns are query, engine1_metric1, engine1_metric2, engine2_metric1, ...

        Args:
            measurements (pd.DataFrame): Measurements dataframe
            engine_names (List[str] | None): List of engine names to be appended.
                If None and multi_metric=True, auto-detects from column names. Defaults to None.
            allow_overwrite (bool, optional): Overwrite measurement for engines already contained for the query. Defaults to False.
            multi_metric (bool, optional): If True, expect columns in engine_metric format. Defaults to False.
        """
        for query in self.queries:
            cur = measurements.loc[measurements["query"] == query.id]
            if cur.size == 0:
                logging.warning("No measurements for query: " + str(query))
                continue

            if multi_metric:
                # Parse engine_metric column names
                engine_metrics = {}
                for col in measurements.columns:
                    if col == "query" or col == "run":
                        continue
                    # Split on last underscore to handle engine names with underscores
                    parts = col.rsplit("_", 1)
                    if len(parts) == 2:
                        engine, metric = parts
                        if engine not in engine_metrics:
                            engine_metrics[engine] = {}
                        value = cur[col].values[0] if len(cur) > 0 else np.nan
                        if not pd.isna(value):
                            engine_metrics[engine][metric] = float(value)

                query.append_measurements(engine_metrics, allow_overwrite)
            else:
                # Single metric format (backward compatible)
                if engine_names is None:
                    raise ValueError(
                        "engine_names must be provided for single-metric format"
                    )

                single_metric_measurements = {}
                for engine in engine_names:
                    if engine in cur.columns:
                        value = cur[engine].values[0] if len(cur) > 0 else np.nan
                        if not pd.isna(value):
                            single_metric_measurements[engine] = float(value)

                query.append_measurements(single_metric_measurements, allow_overwrite)

    def append_statistics_csv(
        self,
        statistics: pd.DataFrame,
        feature_names: List[str],
        allow_overwrite: bool = False,
    ) -> None:
        """Append each feature from a CSV.

        Args:
            measurements (pd.DataFrame): Statistics dataframe
            engine_names (List[str]): List of feature names to be appended
            allow_overwrite (bool, optional): Overwrite features already contained for the query. Defaults to False.
        """
        for query in self.queries:
            cur = statistics.loc[statistics["query"] == query.id]
            if cur.size == 0:
                logging.warning("No statistics for query: " + str(query))
            query.append_measurements(
                {feature: cur[feature] for feature in feature_names}, allow_overwrite
            )

    def __str__(self) -> str:
        return str(self.name) + f" ({len(self.queries)} queries)"
