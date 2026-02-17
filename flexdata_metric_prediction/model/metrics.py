import logging
from typing import Callable, List

import numpy as np
import pandas as pd
import scipy
import torch
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator

from flexdata_metric_prediction.model.loss import Loss


class Metrics:
    def __init__(
        self,
        training_loss: Loss | None,
        aux_loss: List[Loss],
        percentiles: dict | None = None,
        output_fn: Callable | None = None,
        normalizer: BaseEstimator | None = None,
        model_name: str = "",
        log_to_clearml: bool = False,
        step: int = 0,
    ) -> None:
        self.vals = {}
        self.aux_loss = aux_loss
        self.training_loss = training_loss
        self.normalizer = normalizer
        self.model_name = model_name
        self.log_to_clearml = log_to_clearml
        self.step = step

        if output_fn is None:
            output_fn = logging.info
        self.output_fn = output_fn

        self.epoch_predictions = {
            "train": [],
            "test": [],
            "val": [],
        }
        self.epoch_gts = {
            "train": [],
            "test": [],
            "val": [],
        }

        if percentiles is None:
            percentiles = {
                "median": np.median,
                "mean": np.mean,
                "hmean": scipy.stats.hmean,
                "p95": lambda d: sorted(d)[int(len(d) * 0.95)] if len(d) > 0 else np.float64(np.nan),
                "p99": lambda d: sorted(d)[int(len(d) * 0.99)] if len(d) > 0 else np.float64(np.nan),
            }
        self.percentiles = percentiles

        for loss in aux_loss:
            self.vals.update(
                {
                    self._to_key("train", loss.name): {agg_name: [] for agg_name in self.percentiles},
                    self._to_key("test", loss.name): {agg_name: [] for agg_name in self.percentiles},
                    self._to_key("val", loss.name): {agg_name: [] for agg_name in self.percentiles},
                }
            )

        self.vals.update(
            {
                self._to_key("train", "loss"): {agg_name: [] for agg_name in self.percentiles},
                self._to_key("test", "loss"): {agg_name: [] for agg_name in self.percentiles},
                self._to_key("val", "loss"): {agg_name: [] for agg_name in self.percentiles},
            }
        )

    def update(self, pred: torch.Tensor, true: torch.Tensor, stage: str) -> None:
        pred = pred.detach().cpu()
        true = true.detach().cpu()

        if self.normalizer:
            pred = torch.tensor(self.normalizer.inverse_transform(pred))  # type: ignore
            true = torch.tensor(self.normalizer.inverse_transform(true))  # type: ignore

        self.epoch_gts[stage] += true
        self.epoch_predictions[stage] += pred

    def compute_metric(self, metric_fn: Callable, aggr_fn: Callable, stage: str):
        preds = torch.stack(self.epoch_predictions[stage])
        gts = torch.stack(self.epoch_gts[stage])
        res = metric_fn(preds, gts).flatten()
        if isinstance(res, torch.Tensor):
            res = res.numpy()
        if all(res < 0) and aggr_fn == scipy.stats.hmean:
            res = np.abs(res)
        return aggr_fn(res)

    def _compute_report(self, metric_fn: Callable, stage: str, col_width: int):
        rets = []
        for f in self.percentiles:
            rets.append(f"{self.compute_metric(metric_fn, f, stage):<{col_width}.3f}")
        if self.log_to_clearml:
            from clearml import Logger

            logger = Logger.current_logger()
            logger.report_text(f"{str(rets)}\n")
        return str(rets)

    def _to_key(self, stage: str, mName: str):
        return stage + "_" + mName

    def finish_epoch(self, epoch: int):
        for stage in ["train", "test", "val"]:
            if 0 == len(self.epoch_predictions[stage]):
                continue

            for loss in self.aux_loss:
                for agg_key, agg_fn in self.percentiles.items():
                    cur_metric = self.compute_metric(metric_fn=loss, aggr_fn=agg_fn, stage=stage)

                    self.vals[self._to_key(stage, loss.name)][agg_key].append(cur_metric.item())

            if self.training_loss is not None:
                loss_val = self.compute_metric(
                    metric_fn=self.training_loss,
                    aggr_fn=self.percentiles["mean"],
                    stage=stage,
                )
                for agg_key in self.percentiles:
                    self.vals[self._to_key(stage, "loss")][agg_key].append(loss_val)
            if self.log_to_clearml:
                self._log_to_clearml(epoch, stage)

    def start_epoch(self):
        # NOTE: these are not saved over multiple epochs
        self.epoch_predictions = {
            "train": [],
            "test": [],
            "val": [],
        }
        self.epoch_gts = {
            "train": [],
            "test": [],
            "val": [],
        }

    def _log_to_clearml(self, epoch, stage):
        from clearml import Logger

        logger = Logger.current_logger()
        to_log = {}
        for loss in self.aux_loss:
            for p in self.percentiles:
                metric_key = stage + "_" + loss.name + "_" + p
                metric_last_val = self.vals[self._to_key(stage, loss.name)][p][-1]
                to_log[metric_key] = metric_last_val
                if self.model_name != "":
                    logger.report_scalar(
                        title=self.model_name + "_" + metric_key,
                        series=metric_key + "_step_" + str(self.step),
                        value=metric_last_val,
                        iteration=epoch,
                    )

    def log_best_to_clearml(
        self,
        epoch: int,
        stage: str = "val",
        minimize: bool = True,
    ):  # Not called
        to_log = {}
        aggregate = min if minimize else max
        for loss in self.aux_loss:
            for p in self.percentiles:
                best = aggregate(self.vals[self._to_key(stage, loss.name)][p])
                log_metric_key = "best_" + stage + "_" + loss.name + "_" + p
                to_log[log_metric_key] = best
                if self.model_name != "" and self.log_to_clearml:
                    from clearml import Logger

                    logger = Logger.current_logger()
                    logger.report_scalar(
                        title=self.model_name + "_" + log_metric_key,
                        series=log_metric_key + "_step_" + str(self.step),
                        value=best,
                        iteration=epoch,
                    )

    def report(self, epoch, stage):
        header = f"EPOCH {epoch} | Stage: {stage} | Model: {self.model_name}"

        report_vals = []
        for loss in self.aux_loss:
            metric_vals = [loss.name]
            for f in self.percentiles.values():
                metric_vals.append(self.compute_metric(loss, f, stage).item())
            report_vals.append(metric_vals)
        report_table = pd.DataFrame(report_vals, columns=["Metric"] + list(self.percentiles))
        report_table = report_table.round(3)

        self.output_fn(f"{header}\n{report_table}\n")

        if self.log_to_clearml:
            from clearml import Logger

            logger = Logger.current_logger()
            logger.report_table(
                title=f"Training at step {self.step}",
                series=f"Outputs {stage}",
                iteration=epoch,
                table_plot=report_table,
            )

    def is_best_epoch(self, epoch, metric="q_error", aggr="median", split="val", minimize=True):
        return epoch == self.best_epoch_nmbr(metric, aggr, split, minimize)

    def best_epoch_nmbr(self, metric="q_error", aggr="median", split="val", minimize=True):
        metric_vals = [val for val in self.vals[self._to_key(split, metric)][aggr] if not np.isnan(val)]

        if len(metric_vals) == 0:
            # We are at epoch 0
            return 0
        extremal_fn = min if minimize else max
        return self.vals[self._to_key(split, metric)][aggr].index(extremal_fn(metric_vals))

    def log_regression_plot_clearml(self, epoch, title, split="val"):
        from clearml import Logger

        increasing_order = sorted(
            range(len(self.epoch_gts[split])),
            key=lambda x: self.epoch_gts[split][x],
        )
        idxs_left = np.array(range(len(self.epoch_gts[split]))) - 0.2
        idxs_right = np.array(range(len(self.epoch_gts[split]))) + 0.2

        plt.figure(figsize=(12, 4))
        plt.bar(
            idxs_left,
            np.array(self.epoch_gts[split]).flatten()[increasing_order],
            width=0.5,
            label="True exec time",
        )
        plt.bar(
            idxs_right,
            np.array(self.epoch_predictions[split]).flatten()[increasing_order],
            width=0.5,
            label="Predicted exec time",
        )

        plt.title(f"Results on {split} set - epoch {epoch} - {title}")
        plt.legend()

        series_name = f"validation_results_{title}"
        if self.model_name != "":
            series_name = f"{self.model_name}_{series_name}"

        logger = Logger.current_logger()
        logger.report_matplotlib_figure(
            title=f"Validation_results_step {self.step}",
            series=series_name,
            iteration=epoch,
            figure=plt,
        )
        plt.close()

    def log_classification_plot(self, epoch, title, split="val", pred_argmin=False):
        from clearml import Logger

        logger = Logger.current_logger()
        true = torch.stack(self.epoch_gts[split])
        pred = torch.stack(self.epoch_predictions[split])
        single_engine_times = torch.sum(true, dim=0)
        if pred_argmin:
            pred_class = torch.argmin(pred, dim=1)
        else:
            pred_class = torch.argmax(pred, dim=1)
        predicted_total_runtime = 0
        for i in range(pred_class.shape[0]):
            predicted_total_runtime += true[i][pred_class[i]]

        best_single_engine = torch.min(single_engine_times)
        total_best = torch.sum(torch.min(true, dim=1).values)

        data = {
            "Predicted routing": predicted_total_runtime,
            "Best single engine": best_single_engine,
            "Overall best": total_best,
        }

        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        for i, dName in enumerate(data):
            dp = ax[0].bar([i], data[dName], width=0.5, label=dName)[0]
            height = dp.get_height()
            ax[0].text(
                dp.get_x() + dp.get_width() / 2,
                height * 0.9,
                f"{data[dName]:.2f}",
                ha="center",
                va="bottom",
                color="w",
            )
        ax[0].set_title(f"Absolute difference in total exec. time on {split} set")
        ax[0].set_xticks(range(len(data.keys())))
        ax[0].set_xticklabels(data.keys())

        # Also compute relative differences
        for dName in data:
            data[dName] = data[dName] / predicted_total_runtime

        for i, dName in enumerate(data):
            dp = ax[1].bar([i], data[dName], width=0.5, label=dName)[0]
            height = dp.get_height()
            ax[1].text(
                dp.get_x() + dp.get_width() / 2,
                height * 0.9,
                f"{data[dName]:.2f}",
                ha="center",
                va="bottom",
                color="w",
            )
        ax[1].set_title(f"Relative difference in total exec. time on {split} set")
        ax[1].set_xticks(range(len(data.keys())))
        ax[1].set_xticklabels(data.keys())
        plt.tight_layout()

        series_name = f"validation_results_{title}"
        if self.model_name != "":
            series_name = f"{self.model_name}_{series_name}"
        if self.log_to_clearml:
            logger = Logger.current_logger()
            logger.report_matplotlib_figure(
                title="Validation_results",
                series=series_name,
                iteration=epoch,
                figure=plt,
            )

        plt.close()


class AggregateMetrics(Metrics):
    def __init__(
        self,
        loss: Loss | None,
        metrics: List[Loss],
        metric_objs: List[Metrics],
        percentiles: dict | None = None,
        output_fn: Callable | None = None,
        normalizer: BaseEstimator | None = None,
        model_name="Aggregate_(mean)",
        log_to_clearml: bool = False,
        step: int = 0,
    ) -> None:
        super().__init__(
            loss,
            metrics,
            percentiles,
            output_fn,
            normalizer,
            model_name,
            log_to_clearml,
        )
        self.metric_objs = metric_objs

    def compute_metric(self, metric_fn, aggr_fn, stage):
        res = []
        for metric in self.metric_objs:
            res.append(metric.compute_metric(metric_fn, aggr_fn, stage))
        return np.mean(res)

    def finish_epoch(self, epoch):
        for stage in ["train", "test", "val"]:
            try:
                for loss in self.aux_loss:
                    for agg_key, agg_fn in self.percentiles.items():
                        self.vals[self._to_key(stage, loss.name)][agg_key].append(
                            self.compute_metric(metric_fn=loss, aggr_fn=agg_fn, stage=stage).item()
                        )
                for agg_key, agg_fn in self.percentiles.items():
                    self.vals[self._to_key(stage, "loss")][agg_key].append(
                        self.compute_metric(metric_fn=self.training_loss, aggr_fn=agg_fn, stage=stage)
                    )
                if self.log_to_clearml:
                    self._log_to_clearml(epoch, stage)
            except:  # noqa: E722
                continue


class MultiMetricTracker:
    """Track metrics at multiple granularities for multi-metric models.

    Computes Q-error (and other metrics) at:
    - Per engine-metric (e.g., presto-w1/time, spark-w4/memory)
    - Per engine (aggregate across metrics for one engine)
    - Per metric type (aggregate across engines for one metric)
    - Overall (global aggregate)

    Args:
        engines: List of engine names (e.g., ["presto-w1", "spark-w1"])
        metric_names: List of metric names (e.g., ["time", "memory"])
        metric_fns: Dict of metric functions to track (e.g., {"q_error": QLoss(aggr=None)})
        aggregations: Dict of aggregation functions (default: median, mean, p95)
        normalizer: Optional MetricNormalizer for inverse-transforming predictions
                    before computing metrics like Q-error (which expect raw values)
    """

    def __init__(
        self,
        engines: list[str],
        metric_names: list[str],
        metric_fns: dict[str, Loss] | None = None,
        aggregations: dict[str, Callable] | None = None,
        normalizer=None,  # Optional MetricNormalizer
    ) -> None:
        self.engines = engines
        self.metric_names = metric_names
        self.num_engines = len(engines)
        self.num_metrics = len(metric_names)
        self.normalizer = normalizer

        # Default metric functions
        if metric_fns is None:
            from flexdata_metric_prediction.model.loss import QLoss

            metric_fns = {"q_error": QLoss(aggr=None)}
        self.metric_fns = metric_fns

        # Default aggregations
        if aggregations is None:
            aggregations = {
                "median": lambda d: np.nanmedian(d),
                "mean": lambda d: np.nanmean(d),
                "p95": lambda d: np.nanpercentile(d, 95) if len(d) > 0 else np.nan,
            }
        self.aggregations = aggregations

        # Storage for predictions/ground truth per stage
        self._preds = {"train": [], "val": [], "test": []}
        self._trues = {"train": [], "val": [], "test": []}

        # History of computed metrics per epoch
        self.history = {stage: [] for stage in ["train", "val", "test"]}

    def update(self, pred: torch.Tensor, true: torch.Tensor, stage: str) -> None:
        """Accumulate predictions and ground truth for an epoch.

        Args:
            pred: [batch, num_engines * num_metrics] flattened predictions
            true: [batch, num_engines * num_metrics] flattened ground truth
            stage: "train", "val", or "test"
        """
        self._preds[stage].append(pred.detach().cpu())
        self._trues[stage].append(true.detach().cpu())

    def start_epoch(self) -> None:
        """Reset accumulators for new epoch."""
        for stage in self._preds:
            self._preds[stage] = []
            self._trues[stage] = []

    def _reshape_to_grid(self, tensor: torch.Tensor) -> torch.Tensor:
        """Reshape [batch, num_engines * num_metrics] → [batch, num_engines, num_metrics]."""
        batch_size = tensor.shape[0]
        return tensor.view(batch_size, self.num_engines, self.num_metrics)

    def _compute_raw_errors(self, pred: torch.Tensor, true: torch.Tensor, metric_name: str) -> torch.Tensor:
        """Compute per-sample errors with shape [batch, num_engines, num_metrics]."""
        metric_fn = self.metric_fns[metric_name]
        pred_grid = self._reshape_to_grid(pred)
        true_grid = self._reshape_to_grid(true)

        # Compute errors for each engine-metric pair
        errors = torch.zeros_like(pred_grid)
        for e in range(self.num_engines):
            for m in range(self.num_metrics):
                p = pred_grid[:, e, m].unsqueeze(1)
                t = true_grid[:, e, m].unsqueeze(1)
                err = metric_fn(p, t)
                if err.numel() == 1:
                    errors[:, e, m] = err.expand(pred_grid.shape[0])
                else:
                    errors[:, e, m] = err.squeeze()
        return errors

    def compute(self, stage: str) -> dict:
        """Compute all metrics for a stage.

        If a normalizer is provided, predictions and targets are inverse-transformed
        back to raw scale before computing metrics like Q-error.

        Returns:
            Dict with structure:
            {
                "engine_metric": {("presto-w1", "time"): {"q_error": {"median": 1.5, ...}}, ...},
                "engine": {"presto-w1": {"q_error": {"median": 1.6, ...}}, ...},
                "metric": {"time": {"q_error": {"median": 1.4, ...}}, ...},
                "overall": {"q_error": {"median": 1.5, ...}}
            }
        """
        if not self._preds[stage]:
            return {}

        pred = torch.cat(self._preds[stage], dim=0)
        true = torch.cat(self._trues[stage], dim=0)

        # Inverse transform if normalizer is provided (for Q-error in raw space)
        if self.normalizer is not None:
            pred = self.normalizer.inverse_transform_tensor(pred)
            true = self.normalizer.inverse_transform_tensor(true)

        results = {
            "engine_metric": {},
            "engine": {},
            "metric": {},
            "overall": {},
        }

        for metric_name in self.metric_fns:
            errors = self._compute_raw_errors(pred, true, metric_name)  # [batch, engines, metrics]
            errors_np = errors.numpy()

            # Per engine-metric
            for e, engine in enumerate(self.engines):
                for m, mname in enumerate(self.metric_names):
                    key = (engine, mname)
                    if key not in results["engine_metric"]:
                        results["engine_metric"][key] = {}
                    results["engine_metric"][key][metric_name] = {
                        agg_name: float(agg_fn(errors_np[:, e, m])) for agg_name, agg_fn in self.aggregations.items()
                    }

            # Per engine (aggregate across metrics)
            for e, engine in enumerate(self.engines):
                if engine not in results["engine"]:
                    results["engine"][engine] = {}
                engine_errors = errors_np[:, e, :].flatten()
                results["engine"][engine][metric_name] = {
                    agg_name: float(agg_fn(engine_errors)) for agg_name, agg_fn in self.aggregations.items()
                }

            # Per metric type (aggregate across engines)
            for m, mname in enumerate(self.metric_names):
                if mname not in results["metric"]:
                    results["metric"][mname] = {}
                metric_errors = errors_np[:, :, m].flatten()
                results["metric"][mname][metric_name] = {
                    agg_name: float(agg_fn(metric_errors)) for agg_name, agg_fn in self.aggregations.items()
                }

            # Overall
            all_errors = errors_np.flatten()
            results["overall"][metric_name] = {
                agg_name: float(agg_fn(all_errors)) for agg_name, agg_fn in self.aggregations.items()
            }

        return results

    def finish_epoch(self, epoch: int) -> dict:
        """Compute and store metrics for all stages at end of epoch."""
        epoch_results = {}
        for stage in ["train", "val", "test"]:
            if self._preds[stage]:
                epoch_results[stage] = self.compute(stage)
                self.history[stage].append(epoch_results[stage])
        return epoch_results

    def report(self, epoch: int, stage: str = "val") -> str:
        """Generate a concise report string."""
        if not self.history[stage]:
            return f"Epoch {epoch} | {stage}: No data"

        results = self.history[stage][-1]
        lines = [f"Epoch {epoch} | {stage}"]

        # Overall summary
        for metric_name, aggs in results["overall"].items():
            agg_str = ", ".join(f"{k}={v:.3f}" for k, v in aggs.items())
            lines.append(f"  Overall {metric_name}: {agg_str}")

        # Per metric type
        for mname in self.metric_names:
            if mname in results["metric"]:
                for metric_name, aggs in results["metric"][mname].items():
                    lines.append(f"  {mname} {metric_name}: median={aggs['median']:.3f}, mean={aggs['mean']:.3f}")

        # Per engine (compact)
        engine_medians = []
        for engine in self.engines:
            if engine in results["engine"]:
                med = results["engine"][engine].get("q_error", {}).get("median", float("nan"))
                engine_medians.append(f"{engine}={med:.2f}")
        if engine_medians:
            lines.append(f"  Per-engine q_error median: {', '.join(engine_medians)}")

        return "\n".join(lines)

    def get_best_epoch(
        self, metric_name: str = "q_error", agg: str = "median", stage: str = "val", level: str = "overall"
    ) -> int:
        """Get epoch number with best metric value.

        Args:
            level: "overall", "engine", or "metric"
        """
        if not self.history[stage]:
            return 0

        values = []
        for epoch_results in self.history[stage]:
            if level == "overall":
                val = epoch_results["overall"].get(metric_name, {}).get(agg, float("inf"))
            else:
                # For engine/metric level, average across all keys
                vals = [
                    epoch_results[level][k].get(metric_name, {}).get(agg, float("inf")) for k in epoch_results[level]
                ]
                val = np.mean(vals) if vals else float("inf")
            values.append(val)

        return int(np.argmin(values))

    def to_table(self, stage: str = "val", metric_name: str = "q_error") -> list[dict]:
        """Convert metrics to table format for easy viewing.

        Returns list of dicts, one per row (engine + overall), with columns:
        engine, time_median, time_mean, time_p95, memory_median, ..., combined_median, ...

        Args:
            stage: "train", "val", or "test"
            metric_name: Which metric to use (default: "q_error")

        Returns:
            List of row dicts suitable for pandas DataFrame or wandb.Table
        """
        if not self.history[stage]:
            return []

        results = self.history[stage][-1]
        rows = []

        # Row for each engine
        for engine in self.engines:
            row = {"engine": engine}

            # Per-metric columns (time, memory, etc.)
            for mname in self.metric_names:
                key = (engine, mname)
                if key in results["engine_metric"]:
                    for agg_name, value in results["engine_metric"][key].get(metric_name, {}).items():
                        row[f"{mname}_{agg_name}"] = value

            # Combined column (aggregate across metrics for this engine)
            if engine in results["engine"]:
                for agg_name, value in results["engine"][engine].get(metric_name, {}).items():
                    row[f"combined_{agg_name}"] = value

            rows.append(row)

        # Overall row
        overall_row = {"engine": "overall"}
        for mname in self.metric_names:
            if mname in results["metric"]:
                for agg_name, value in results["metric"][mname].get(metric_name, {}).items():
                    overall_row[f"{mname}_{agg_name}"] = value

        if results["overall"].get(metric_name):
            for agg_name, value in results["overall"][metric_name].items():
                overall_row[f"combined_{agg_name}"] = value

        rows.append(overall_row)

        return rows


class MultiMetrics:
    def __init__(
        self,
        metrics: List[Metrics],
        overall_metrics: Metrics,
        aggregate_metrics: AggregateMetrics,
    ) -> None:
        self.metrics = metrics
        self.overall_metrics = overall_metrics
        self.aggregate_metrics = aggregate_metrics

        self.epoch_predictions = {
            "train": [],
            "test": [],
            "val": [],
        }
        self.epoch_gts = {
            "train": [],
            "test": [],
            "val": [],
        }

    def update(self, pred, true, stage):
        # NOTE: pred will be a list
        for i, pred_model in enumerate(pred):
            self.metrics[i].update(pred_model, true[:, i].reshape(-1, 1), stage)

        self.overall_metrics.update(torch.stack(pred, dim=1).flatten(1), true, stage)

    def finish_epoch(self, epoch):
        self.aggregate_metrics.finish_epoch(epoch)
        for metric in self.metrics:
            metric.finish_epoch(epoch)
        self.overall_metrics.finish_epoch(epoch)

    def start_epoch(self):
        for metric in self.metrics:
            metric.start_epoch()
        self.overall_metrics.start_epoch()

    def report(self, epoch, stage):
        for metric in self.metrics:
            metric.report(epoch, stage)
        self.overall_metrics.report(epoch, stage)
        self.aggregate_metrics.report(epoch, stage)

    def is_best_epoch(self, epoch, metric="q_error", aggr="median", split="val", minimize=True):
        return self.overall_metrics.is_best_epoch(epoch, metric, aggr, split, minimize)

    def best_epoch_nmbr(self, metric="q_error", aggr="median", split="val", minimize=True):
        return self.overall_metrics.best_epoch_nmbr(metric, aggr, split, minimize)

    def best_aggr_epoch_nmbr(self, metric="q_error", aggr="median", split="val", minimize=True):
        return self.aggregate_metrics.best_epoch_nmbr(metric, aggr, split, minimize)

    def is_best_aggr_epoch(self, epoch, metric="q_error", aggr="median", split="val", minimize=True):
        return self.aggregate_metrics.is_best_epoch(epoch, metric, aggr, split, minimize)
