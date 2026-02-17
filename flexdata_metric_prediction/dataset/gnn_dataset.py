import logging
import random
from typing import Any, List, Tuple

import numpy as np

from flexdata_metric_prediction.dataset.query_dataset import QueryDataset
from flexdata_metric_prediction.encoder.encoder import Encoder


class GNNDataset:
    """Dataset containing torch_geometric graph objects created from `Tree` objects.
    A dataset is meant to mix queries targeting different schemas and represent a split during training.
    Each query is encoded the same way.
    """

    def __init__(
        self,
        datasets: List[QueryDataset],
        encoder: Encoder,
        skip_on_error: bool = True,
        log_to_clearml: bool = False,
        target_metrics: dict[str, list[str]] | None = None,
    ) -> None:
        """Constructor

        Args:
            datasets (List[QueryDataset]): List of query datasets to include
            encoder (Encoder): Encoder to be used for creating the GNN graphs
            skip_on_error (bool, optional): Skip queries that fail during encoding. Defaults to True.
            log_to_clearml (bool, optional): Also log to clearml. Defaults to False.
            target_metrics (dict[str, list[str]] | None): Metrics to use as targets per engine.
                Format: {"engine_name": ["metric1", "metric2", ...]}
                If None, defaults to {"all_engines": ["time"]} for backward compatibility.
                Example: {
                    "presto_w1": ["time", "cpu", "memory"],
                    "spark_w1": ["time", "executor_run_time_ms", "memory"]
                }
        """
        self.data = {}
        # NOTE: we take the largest set of engines as targets
        # and drop all queries that do not contain measurements for each
        self.engines = set()
        for dataset in datasets:
            for query in dataset.queries:
                if query.measurements is not None:
                    self.engines.update(query.measurements.keys())
        self.engines = sorted(self.engines)  # NOTE: from this point, the order of engines is fixed!

        # Set target metrics (backward compatible)
        if target_metrics is None:
            # Default: all engines use "time" metric only
            self.target_metrics = {engine: ["time"] for engine in self.engines}
        else:
            self.target_metrics = target_metrics
            # Validate that all engines have metrics specified
            for engine in self.engines:
                if engine not in self.target_metrics:
                    logging.warning(f"No target metrics specified for engine {engine}, defaulting to ['time']")
                    self.target_metrics[engine] = ["time"]

        self.translated_queries_mask = {}
        for dataset in datasets:
            self.data[dataset.name] = []
            self.translated_queries_mask[dataset.name] = []
            self._log(f"Processing dataset {str(dataset)}", log_to_clearml)
            self._process_dataset(dataset, encoder, skip_on_error, log_to_clearml)

    def _process_dataset(
        self,
        dataset: QueryDataset,
        encoder: Encoder,
        skip_on_error: bool = True,
        log_to_clearml=False,
    ):
        """Create a GNN data object for each query in the dataset.

        Args:
            datasets (List[QueryDataset]): List of query datasets to include
            encoder (Encoder): Encoder to be used for creating the GNN graphs
            skip_on_error (bool, optional): Skip queries that fail during encoding. Defaults to True.
            log_to_clearml (bool, optional): Also log to clearml. Defaults to False.
        """
        error_summary = {}
        for query in dataset.queries:
            if query.plan is None:
                self._log("No plan for query: " + str(query) + "... Skipping", log_to_clearml)
                self.translated_queries_mask[dataset.name].append(False)
                continue

            try:
                if query.tree is None:
                    query.init_tree()
            except Exception as e:
                self._handle_error(
                    f"Failure at creating the Tree: {str(query)}",
                    "Tree creation",
                    skip_on_error,
                    error_summary,
                    e,
                )
                self.translated_queries_mask[dataset.name].append(False)
                continue

            assert query.tree is not None and query.tree.root is not None

            if not all(engine in query.measurements for engine in self.engines):
                self._handle_error(
                    f"Missing measurement for query: {str(query)}",
                    "Missing measurement",
                    skip_on_error,
                    error_summary,
                    AttributeError(),
                )
                self.translated_queries_mask[dataset.name].append(False)
                continue

            try:
                encoder.encode_tree(query.tree.root, query.plan)
            except Exception as e:
                self._handle_error(
                    f"Failure at encoding query: {str(query)}",
                    "Encoding",
                    skip_on_error,
                    error_summary,
                    e,
                )
                self.translated_queries_mask[dataset.name].append(False)
                continue

            # Extract multi-metric labels per engine
            # Shape: (num_engines, num_metrics_per_engine)
            # For backward compatibility with single metric, this becomes (num_engines, 1)
            cur_labels = []
            for engine in self.engines:
                engine_metrics = []
                for metric in self.target_metrics[engine]:
                    value = query.get_metric(engine, metric)
                    engine_metrics.append(value if value is not None else np.nan)
                cur_labels.append(engine_metrics)

            # Flatten to 1D for compatibility with existing code
            # Labels are stored as: [engine1_metric1, engine1_metric2, ..., engine2_metric1, ...]
            flattened_labels = [metric for engine_metrics in cur_labels for metric in engine_metrics]

            try:
                cur_data = query.tree.get_gnn_data(np.asarray([flattened_labels]))
                # Store metadata about label structure
                cur_data.num_metrics_per_engine = [len(self.target_metrics[engine]) for engine in self.engines]
                cur_data.metric_names_per_engine = [self.target_metrics[engine] for engine in self.engines]
                # Store schema/dataset name and query id for filtering during evaluation
                cur_data.schema = dataset.name
                cur_data.query_id = query.id
            except Exception as e:
                self._handle_error(
                    "Failure at creating the GNN graph structure",
                    "GNN data",
                    skip_on_error,
                    error_summary,
                    e,
                )
                current_query_succeeds = False
                self.translated_queries_mask[dataset.name].append(current_query_succeeds)
                continue

            self.data[dataset.name].append(cur_data)
            self.translated_queries_mask[dataset.name].append(True)
        self._report_errors(error_summary)

    def get_splits(self, val_split: dict, test_split: dict, shuffle: bool = True) -> Tuple[Any, Any, Any]:
        """Split the dataset into multiple splits. Datapoints not in the validation or test split are automatically included in the training split.

        Args:
            val_split (dict): Dictionary mapping from dataset names to the number of datapoints to be included in the validation split.
            test_split (dict): Dictionary mapping from dataset names to the number of datapoints to be included in the test split.
            shuffle (bool, optional): Shuffle. Defaults to True.

        Returns:
            Tuple[Self]: train, val, test GNNDataset objects
        """
        train = []
        val = []
        test = []

        for dataset_name in self.data:
            if shuffle:
                random.shuffle(self.data[dataset_name])

            if test_split[dataset_name] == -1:
                cur_train = 0
                cur_val = 0
            else:
                cur_train = len(self.data[dataset_name]) - val_split[dataset_name] - test_split[dataset_name]
                cur_val = val_split[dataset_name]

            train += self.data[dataset_name][:cur_train]
            val += self.data[dataset_name][cur_train : cur_train + cur_val]
            test += self.data[dataset_name][cur_train + cur_val :]

        return train, val, test

    def get_label_index(self, engine: str, metric: str) -> int:
        """Get the flattened label index for a specific engine/metric pair.

        Labels are stored as: [engine1_metric1, engine1_metric2, ..., engine2_metric1, ...]
        This helper maps (engine, metric) → index for easier debugging and extraction.
        """
        idx = 0
        for eng in self.engines:
            if eng == engine:
                return idx + self.target_metrics[eng].index(metric)
            idx += len(self.target_metrics[eng])
        raise ValueError(f"Engine '{engine}' or metric '{metric}' not found")

    def get_label_mapping(self) -> dict[tuple[str, str], int]:
        """Return full mapping of (engine, metric) → label index."""
        mapping = {}
        idx = 0
        for engine in self.engines:
            for metric in self.target_metrics[engine]:
                mapping[(engine, metric)] = idx
                idx += 1
        return mapping

    def _handle_error(
        self,
        msg: str,
        category: str,
        skip_on_error: bool,
        error_summary: dict,
        error: BaseException | None = None,
    ):
        if skip_on_error:
            if category not in error_summary:
                error_summary[category] = 0
            error_summary[category] += 1
            logging.debug(msg + "... Skipping")
            logging.debug(error)
        else:
            raise RuntimeError(msg) from error

    def _report_errors(self, error_summary):
        if len(error_summary) != 0:
            logging.warning(error_summary)

    def _log(self, message, to_clearml):
        logging.info(message)
        if to_clearml:
            from clearml import Logger

            Logger.current_logger().report_text(message + "\n")
