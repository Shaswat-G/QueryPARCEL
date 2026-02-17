import logging
import os
from typing import List

import numpy as np
import pandas as pd
import torch

from flexdata_metric_prediction.dataset.query_datapoint import QueryDatapoint
from flexdata_metric_prediction.encoder.encoder import Encoder


def read_exp_measurements(labels_path: str, metric_cols: list[str] | None = None) -> pd.DataFrame:
    """Read measurements of multiple queries on a given engine.

    Supports both single-metric (time only) and multi-metric formats:
    - Single metric: CSV with 'query', 'run', 'time' columns
    - Multi metric: CSV with 'query', 'run', and multiple metric columns

    If multiple measurements (runs) exist for a query, the median is taken.

    Args:
        labels_path: Path to the CSV file
        metric_cols: List of metric column names to extract. If None, extracts all numeric columns
            except 'query' and 'run'. For backward compatibility, defaults to ['time'] if only
            'time' column exists.

    Returns:
        pd.DataFrame: DataFrame with 'query' as index and metric columns
            Single metric format: columns ['time']
            Multi metric format: columns ['time', 'memory', 'cpu', ...] depending on input
    """
    labels_df = pd.read_csv(labels_path)

    # Identify metric columns
    if metric_cols is None:
        # Auto-detect: all numeric columns except query and run
        exclude_cols = {"query", "run", "schema", "query_id", "state", "status"}
        metric_cols = [
            col
            for col in labels_df.columns
            if col not in exclude_cols and pd.api.types.is_numeric_dtype(labels_df[col])
        ]

        # Backward compatibility: if only 'time' exists, use it
        if not metric_cols and "time" in labels_df.columns:
            metric_cols = ["time"]

    if not metric_cols:
        raise ValueError(f"No metric columns found in {labels_path}")

    # Aggregate by query (median across runs)
    agg_dict = {col: "median" for col in metric_cols}
    labels_df = labels_df.groupby("query").agg(agg_dict).reset_index()

    return labels_df.set_index("query")


def merge_engine_measurements(engine_mes: dict[str, pd.DataFrame], merge_strategy: str = "inner") -> pd.DataFrame:
    """Merge measurement dataframes for multiple engines.

    Supports both single-metric and multi-metric formats. For multi-metric inputs,
    each engine's metrics are prefixed with the engine name.

    Args:
        engine_mes: Dictionary mapping engine names to their measurement DataFrames
        merge_strategy: How to merge queries across engines:
            - "inner": Only include queries present in ALL engines (default, backward compatible)
            - "outer": Include all queries from any engine (NaN for missing measurements)

    Returns:
        pd.DataFrame: Merged dataframe with 'query' as index
            For single metric: columns are engine names (e.g., 'presto_w1', 'spark_w1')
            For multi metric: columns are engine_metric format (e.g., 'presto_w1_time',
                'presto_w1_memory', 'spark_w1_time', 'spark_w1_memory')
    """
    if not engine_mes:
        return pd.DataFrame()

    # Determine if we're in single-metric or multi-metric mode
    first_df = next(iter(engine_mes.values()))
    is_single_metric = len(first_df.columns) == 1 and first_df.columns[0] == "time"

    if is_single_metric:
        # Backward compatible mode: single 'time' metric
        if merge_strategy == "inner":
            # Find queries present in all engines
            queries = None
            for engine_df in engine_mes.values():
                cur_queries = engine_df.index.unique().tolist()
                queries = cur_queries if queries is None else [q for q in queries if q in cur_queries]
            queries = sorted(queries)
        else:
            # Union of all queries
            queries_set = set()
            for engine_df in engine_mes.values():
                queries_set.update(engine_df.index.unique())
            queries = sorted(queries_set)

        # Build merged DataFrame
        labels = pd.DataFrame(index=queries)
        labels.index.name = "query"

        for engine, engine_df in engine_mes.items():
            labels[engine] = engine_df.loc[engine_df.index.intersection(queries)]["time"]

        return labels

    else:
        # Multi-metric mode: prefix each metric with engine name
        merged_df = None

        for engine, engine_df in engine_mes.items():
            # Create a copy and prefix all columns with engine name
            engine_prefixed = engine_df.copy()
            engine_prefixed.columns = [f"{engine}_{col}" for col in engine_prefixed.columns]

            if merged_df is None:
                merged_df = engine_prefixed
            else:
                # Merge on query index
                how = "inner" if merge_strategy == "inner" else "outer"
                merged_df = merged_df.join(engine_prefixed, how=how)

        return merged_df


def create_datapoints(
    plans_dir: str, measurements: dict[str, str], encoder: Encoder | None = None
) -> List[QueryDatapoint]:
    """DEPRECATED"""
    mes_dfs = {engine: read_exp_measurements(eng_mes) for engine, eng_mes in measurements.items()}
    combined_df = merge_engine_measurements(mes_dfs)
    query_datapoints = []

    for query in combined_df["query"].tolist():
        # NOTE: a query with id quer_xx.sql is mapped to the plan plan_query_xx.json
        plan_name = "plan_" + query.split(".sql")[0]
        plan_path = os.path.join(plans_dir, plan_name)
        if not os.path.exists(plan_path):
            logging.warning("No plan found for query: " + query)
            continue
    return query_datapoints


# def createWeightedDataloader(data, batch_size, shuffle, weight_coef, normalizer):
#     ys = np.asarray([dp.y for dp in data])
#     if normalizer is not None:
#         ys = np.asarray([normalizer.inverse_transform(y.reshape(-1, 1)) for y in ys])
#     min_val, max_val = np.min(ys), np.max(ys)
#     weights = torch.tensor(
#         [((1 + weight_coef * (y / max_val)) / 2).item() for y in ys]
#     ).flatten()
#     sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
#     return DataLoader(
#         data, batch_size=batch_size, shuffle=False, sampler=sampler, drop_last=False
#     )


class LogNormalizer:
    """DEPRECATED"""

    def __init__(self) -> None:
        pass

    def fit(self, ys):
        pass

    def transform(self, ys):
        return np.log1p(ys)

    def inverse_transform(self, ys):
        return np.expm1(ys)


def normalizerTreeLabels(data, normalizer=None):
    """DEPRECATED"""
    ys = np.asarray([dp.y for dp in data])
    if normalizer is None:
        # normalizer = preprocessing.RobustScaler()
        # normalizer = preprocessing.MinMaxScaler(feature_range=(0.000001, 1))
        normalizer = LogNormalizer()
        normalizer.fit(ys.reshape(-1, 1))

    for tree, normed_label in zip(data, normalizer.transform(ys)):
        tree.y = torch.tensor(normed_label)
    # for tree, normed_label in zip(data, normalizer.transform(ys.reshape(-1, 1))):
    #     tree.y = torch.tensor(normed_label).reshape(-1, 1)
    return data, normalizer
