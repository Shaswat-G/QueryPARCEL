import logging

import torch
from scipy.stats import hmean
from sklearn.metrics import f1_score
from torch import nn


class Loss(nn.Module):
    _aggregate_functions = {
        "max": torch.max,
        "min": torch.min,
        "argmax": torch.argmax,
        "argmin": torch.argmin,
        "hmean": hmean,
        "median": torch.median,
        "mean": torch.nanmean,
        "sum": torch.nansum,
    }

    def __init__(self, name: str, aggr: str, allow_nans: bool = False) -> None:
        super().__init__()
        self.name = name
        self.aggr = self._aggregate_functions.get(aggr, lambda d: d)
        self.allow_nans = allow_nans

    def nan_operation(self, tensor, op: str, dimension: int | None = None, **kwargs):  # noqa: C901
        def _mask(x: torch.Tensor, fill: float) -> torch.Tensor:
            return torch.where(torch.isnan(x), torch.full_like(x, fill), x)

        if not self.allow_nans and torch.isnan(tensor).any():
            raise ValueError("Encountered a NaN value in the tensor, while allow_nans is False")

        if dimension is not None and tensor.ndim < dimension:
            raise ValueError(
                f"The tensor doesn't have enough dimensions to perform an operation along axis {dimension}"
            )

        if op in ["min", "argmin"]:
            tensor = _mask(tensor, float("inf"))
        elif op in ["max", "argmax"]:
            tensor = _mask(tensor, -float("inf"))

        if op in ["min", "argmin", "max", "argmax", "sum"]:
            res = self._aggregate_functions[op](tensor, dim=dimension)
        elif op == "clip_quantile":
            if "alpha" not in kwargs:
                raise ValueError("Quantile level alpha is required for quantile operation")
            alpha = kwargs["alpha"]
            low_quantile = torch.nanquantile(tensor, alpha, dim=dimension, keepdim=True)
            high_quantile = torch.nanquantile(tensor, 1 - alpha, dim=dimension, keepdim=True)
            res = torch.clip(tensor, min=low_quantile, max=high_quantile)
        elif op == "filter":
            if tensor.ndim > 1 and tensor.shape[1] >= 2:
                logging.warning("The tensor passed to the filter operation has a matrix form, it will be flattened")
            mask = torch.isnan(tensor)
            res = tensor[~mask]
        elif op == "mean":
            res = torch.nanmean(tensor, dim=dimension, keepdim=True)
        elif op is None:
            res = tensor
        else:
            raise ValueError(f"Unknown aggregation operation: {op}")
        return res

    def joint_nan_filter(self, pred, true):
        mask1 = torch.isnan(pred)
        mask2 = torch.isnan(true)
        mask = mask1 | mask2
        return pred[~mask], true[~mask]


# class ClassificationLoss(Loss):
#     def __init__(self, name, aggr, preds_argmin, allow_nans=False) -> None:
#         super().__init__(name, aggr, allow_nans)
#         self.preds_argmin = preds_argmin

#     def get_prediction_classes(self, pred):
#         if self.preds_argmin:
#             # pred_class = torch.argmin(pred, dim=1)
#             pred_class = self.nan_operation(pred, "argmin", 1)
#         else:
#             # pred_class = torch.argmax(pred, dim=1)
#             pred_class = self.nan_operation(pred, "argmax", 1)
#         return pred_class


# class F1(ClassificationLoss):
#     def __init__(self, aggr="none", preds_argmin=False, allow_nans: bool = False) -> None:
#         super().__init__(
#             name="F1",
#             aggr=aggr,
#             preds_argmin=preds_argmin,
#             allow_nans=allow_nans,
#         )
#         if allow_nans:
#             raise ValueError(
#                 "The F1 score is not well defined with NaNs, please set allow_nans = False to use the metric"
#             )

#     def forward(self, pred, true):
#         true = true.reshape(pred.shape)
#         true_class = self.get_prediction_classes(true)
#         pred_class = self.get_prediction_classes(pred)
#         return self.aggr(torch.tensor([f1_score(true_class, pred_class, average="weighted")]))


# class WeightedNLL(ClassificationLoss):
#     def __init__(self, aggr="mean", preds_argmin=False, allow_nans: bool = False) -> None:
#         super().__init__(
#             name="WeightedNLL",
#             aggr=aggr,
#             preds_argmin=preds_argmin,
#             allow_nans=allow_nans,
#         )
#         if allow_nans:
#             raise ValueError(
#                 "The Weighted NLL is not well defined with NaNs, please set allow_nans = False to use the metric"
#             )
#         self.nll_loss = nn.NLLLoss(reduction="none")

#     def forward(self, pred, true):
#         # true = true.reshape(pred.shape)
#         true_class = self.nan_operation(true, op="argmin", dimension=1)
#         pred_class = self.nan_operation(pred, op="argmax", dimension=1)
#         nll_loss_out = self.nll_loss(pred, true_class)
#         weights = 1 + true[torch.arange(pred.shape[0]), pred_class] - torch.min(true, dim=1).values
#         return self.aggr(nll_loss_out * weights)


# class NLL(ClassificationLoss):
#     def __init__(self, aggr="mean", preds_argmin=False, allow_nans: bool = False) -> None:
#         super().__init__(
#             name="NegativeLogLikelihood",
#             aggr=aggr,
#             preds_argmin=preds_argmin,
#             allow_nans=allow_nans,
#         )
#         if allow_nans:
#             raise ValueError("The NLL is not well defined with NaNs, please set allow_nans = False to use the metric")
#         self.nll_loss = nn.NLLLoss(reduction="none")

#     def forward(self, pred, true):
#         true = true.reshape(pred.shape)
#         true_class = self.nan_operation(true, op="argmin", dimension=1)
#         return self.aggr(self.nll_loss(pred, true_class))


# class TimeSaved(ClassificationLoss):
#     def __init__(self, aggr="mean", preds_argmin=False, allow_nans: bool = False) -> None:
#         super().__init__(
#             name="TimeSaved",
#             preds_argmin=preds_argmin,
#             aggr=aggr,
#             allow_nans=allow_nans,
#         )
#         # NOTE: if we predict running time, we have to minimize
#         # if we predict probabilities, we maximize
#         if allow_nans:
#             raise ValueError(
#                 "The TimeSaved is not well defined with NaNs, since single_engine_times also sums over NaNs. Please set allow_nans = False to use the metric"
#             )
#         self.preds_argmin = preds_argmin

#     def forward(self, pred, true):
#         true = true.reshape(pred.shape)
#         single_engine_times = self.nan_operation(true, op="sum", dimension=0)
#         pred_class = self.get_prediction_classes(pred)
#         predicted_total_runtime = 0
#         for i in range(pred_class.shape[0]):
#             predicted_total_runtime += true[i][pred_class[i]]
#         return predicted_total_runtime - torch.min(single_engine_times)


# class TimeCost(ClassificationLoss):
#     def __init__(self, aggr="mean", preds_argmin=False, allow_nans: bool = False) -> None:
#         super().__init__(
#             name="TimeCost",
#             preds_argmin=preds_argmin,
#             aggr=aggr,
#             allow_nans=allow_nans,
#         )
#         # NOTE: if we predict running time, we have to minimize
#         # if we predict probabilities, we maximize
#         self.preds_argmin = preds_argmin

#     def forward(self, pred, true):
#         true = true.reshape(pred.shape)
#         true_class = self.get_prediction_classes(true)
#         pred_class = self.get_prediction_classes(pred)
#         predicted_total_runtime = 0
#         best_total_runtime = 0
#         for i in range(pred_class.shape[0]):
#             if torch.isnan(true[i][pred_class[i]]):
#                 logging.warning("The model selected an engine for which we have no measurements")
#             predicted_total_runtime += true[i][pred_class[i]]
#             best_total_runtime += true[i][true_class[i]]

#         return predicted_total_runtime - best_total_runtime


# class Accuracy(ClassificationLoss):
#     """NOTE: this expects that the labels are not yet transformed to one-hot"""

#     def __init__(self, aggr="mean", preds_argmin=False, allow_nans: bool = False):
#         super().__init__(
#             name="Accuracy",
#             aggr=aggr,
#             preds_argmin=preds_argmin,
#             allow_nans=allow_nans,
#         )

#     def forward(self, pred, true):
#         true = true.reshape(pred.shape)
#         assert pred.shape == true.shape
#         assert 0 < true.shape[0]

#         true_class = self.get_prediction_classes(true)
#         pred_class = self.get_prediction_classes(pred)
#         accuracy = (pred_class == true_class).float()
#         return self.aggr(accuracy)


class LogQLoss(Loss):
    """Log of Q-error for RAW (linear) scale predictions.

    Computes log(Q-error) = |log(pred) - log(true)|.

    IMPORTANT: This expects predictions and targets in RAW scale (not log-space).
    For normalized log-space predictions, use MetricNormalizer.inverse_transform()
    before computing.

    This is primarily used internally by QLoss. For training on normalized log-space
    data, use HuberLoss directly instead.

    Args:
        aggr: Aggregation method ("mean", "sum", None)
        numerical_stability: Small epsilon to prevent log(0)
        allow_nans: Whether to allow NaN values
    """

    def __init__(
        self,
        aggr: str | None = "mean",
        numerical_stability: float | None = 1e-7,
        allow_nans: bool | None = False,
    ) -> None:
        self.numerical_stability = numerical_stability
        super().__init__(name="log(q_error)", aggr=aggr, allow_nans=allow_nans)

    def forward(self, pred, true):
        """Compute log(Q-error).

        Args:
            pred: Predictions in RAW scale [batch, features]
            true: Ground truth in RAW scale [batch, features]

        Returns:
            log(Q-error) values (>= 0)
        """
        mask = (pred < true).flatten()
        loss = torch.zeros_like(pred)
        loss[mask] = torch.log(true[mask] + self.numerical_stability) - torch.log(pred[mask] + self.numerical_stability)
        loss[~mask] = torch.log(pred[~mask] + self.numerical_stability) - torch.log(
            true[~mask] + self.numerical_stability
        )
        loss_masked = self.nan_operation(loss, op="filter")
        return self.aggr(loss_masked)


class QLoss(Loss):
    """Q-error loss for RAW (linear) scale predictions.

    Q-error = max(pred/true, true/pred), always >= 1.0.

    IMPORTANT: This expects predictions and targets in RAW scale (not log-space).
    For normalized log-space predictions, use MetricNormalizer.inverse_transform()
    before computing Q-error.

    The Q-error is computed as: exp(|log(pred) - log(true)|)

    Typical usage:
        - Training loss: HuberLoss on normalized log-space
        - Evaluation metric: QLoss on inverse-transformed (raw) values

    Args:
        aggr: Aggregation method ("mean", "median", None for per-sample)
        allow_nans: Whether to allow NaN values
    """

    def __init__(
        self,
        aggr: str | None = "mean",
        allow_nans: bool = False,
    ) -> None:
        super().__init__(name="q_error", aggr=aggr, allow_nans=allow_nans)
        self._logqerror = LogQLoss(aggr=None, allow_nans=allow_nans)

    def forward(self, pred, true):
        """Compute Q-error.

        Args:
            pred: Predictions in RAW scale [batch, features]
            true: Ground truth in RAW scale [batch, features]

        Returns:
            Q-error values (>= 1.0)
        """
        return self.aggr(torch.exp(self._logqerror(pred, true)))


class MSE(Loss):
    ## FIXME: Give this a deeper look since I am not sure, moreover I think you
    ## didn t pass any aggregate to this metric, but this way we are still propagating NaNs, so
    ## we might need to further investigate this

    def __init__(self, aggr="none", allow_nans: bool = False) -> None:
        self.aggr_string = aggr
        super().__init__(name="MSE", aggr=aggr, allow_nans=allow_nans)

    def forward(self, pred, true):
        squares = (pred - true).pow(2)
        if self.aggr_string == "sum":
            return self.nan_operation(squares, op="sum")
        elif self.aggr_string == "mean":
            return self.nan_operation(squares, op="mean")
        return self.nan_operation(squares, op="filter")


class HuberLoss(Loss):
    """Huber Loss for normalized log-space predictions.

    Designed for use with MetricNormalizer: predictions and targets should be
    z-score normalized log values, i.e., z_norm = (log(y) - μ) / σ.

    This loss is equivalent to minimizing log(Q-error) for small errors (MSE region)
    and |log(Q-error)| for larger errors (MAE region), providing robustness to outliers.

    Loss = 0.5 * (pred - true)^2              if |pred - true| <= delta
           delta * (|pred - true| - 0.5*delta)  otherwise

    With delta=1.0 and normalized data:
    - MSE region: errors < 1σ (small relative errors)
    - MAE region: errors > 1σ (large relative errors, downweighted)

    Args:
        delta: Threshold between MSE and MAE regions (default: 1.0).
               In normalized space, delta=1.0 corresponds to ~1 std deviation.
        aggr: Aggregation method ("mean", "sum", None)
        allow_nans: Whether to allow NaN values
    """

    def __init__(self, delta: float = 1.0, aggr: str | None = "mean", allow_nans: bool = False) -> None:
        super().__init__(name="HuberLoss", aggr=aggr, allow_nans=allow_nans)
        self.delta = delta

    def forward(self, pred, true):
        """Compute Huber loss.

        Args:
            pred: Predictions in log space [batch, features]
            true: Ground truth in log space [batch, features]

        Returns:
            Aggregated Huber loss
        """
        pred_masked, true_masked = self.joint_nan_filter(pred, true)

        # Compute absolute error
        abs_error = torch.abs(pred_masked - true_masked)

        # Huber loss: quadratic for small errors, linear for large errors
        loss = torch.where(
            abs_error <= self.delta,
            0.5 * abs_error.pow(2),  # MSE region
            self.delta * (abs_error - 0.5 * self.delta),  # MAE region
        )

        return self.aggr(loss)


# class NumNegatives(Loss):
#     def __init__(self, aggr: str | None = "mean", allow_nans: bool = False) -> None:
#         super().__init__(name="NumNegatives", aggr=aggr, allow_nans=allow_nans)

#     def forward(self, pred, true):
#         return sum(pred < 0)


# class CustomQLoss(Loss):
#     """From ZeroShot paper"""

#     def __init__(
#         self,
#         min_val: float = 1e-10,
#         penalty_negative: float = 1e3,
#         aggr: str | None = "mean",
#         allow_nans: bool | None = False,
#     ) -> None:
#         super().__init__(name="CustomQLoss", aggr=aggr, allow_nans=allow_nans)
#         self.min_val = min_val
#         self.penalty_negative = penalty_negative
#         self.log_error = LogQLoss(aggr=None, allow_nans=allow_nans)

#     def forward(self, pred, true):
#         pred_masked, true_masked = self.joint_nan_filter(pred, true)
#         negatives = (pred_masked <= self.min_val).flatten()
#         loss = torch.zeros_like(pred_masked)
#         loss[~negatives] = self.log_error(pred_masked[~negatives], true_masked[~negatives])
#         loss[negatives] = self.penalty_negative * (1 - pred_masked[negatives])
#         if torch.any(torch.isinf(loss)) or torch.any(torch.isnan(loss)):
#             logging.warning("Inf or NaN in Loss!")
#         return self.aggr(loss)


# class CustomRelativeError(Loss):
#     def __init__(
#         self,
#         aggr: str | None = "mean",
#         damping: float = 1,
#         allow_nans: bool = False,
#     ) -> None:
#         super().__init__(
#             name="damped relative error",
#             aggr=aggr,
#             allow_nans=allow_nans,
#         )
#         self.aggr_string = aggr
#         self.damping = damping

#     def forward(self, pred, true, alpha: float = 0.05):
#         weights = self.nan_operation(true, op="clip_quantile", dimension=0, alpha=alpha)
#         res = weights * (pred - true) / (self.damping + true)
#         return self.nan_operation(res, op=self.aggr_string, dimension=0)


# class OverestimatingError(Loss):
#     def __init__(
#         self,
#         aggr: str | None = "mean",
#         allow_nans: bool = False,
#     ) -> None:
#         super().__init__(name="Overestimation", aggr=aggr, allow_nans=allow_nans)
#         self.aggr_string = aggr
#         self.loss = CustomRelativeError(aggr=self.aggr_string, allow_nans=allow_nans)

#     def forward(self, pred, true):
#         mask = pred > true
#         pred_masked = pred[mask]
#         true_masked = true[mask]

#         # Return 0 if no predictions are overestimating
#         if not torch.any(mask):
#             return torch.tensor(0)

#         return self.aggr(self.loss(pred_masked, true_masked))


# class UnderestimatingError(Loss):
#     def __init__(
#         self,
#         aggr: str | None = "mean",
#         allow_nans: bool = False,
#     ) -> None:
#         super().__init__(name="Underestimation", aggr=aggr, allow_nans=allow_nans)
#         self.aggr_string = aggr
#         self.loss = CustomRelativeError(aggr=self.aggr_string, allow_nans=allow_nans)

#     def forward(self, pred, true):
#         mask = pred < true
#         pred_masked = pred[mask]
#         true_masked = true[mask]

#         # Return 0 if no predictions are underestimating
#         if not torch.any(mask):
#             return torch.tensor(0)

#         return self.aggr(self.loss(pred_masked, true_masked))


class MultiMetricLoss(Loss):
    """Multi-metric loss for predicting multiple metrics per engine.

    Computes weighted loss across metric types (e.g., time, memory), then averages
    across engines. Designed for use with normalized log-space predictions.

    For each engine e and metric m:
        L_engine = Σ_m (λ_m × loss_m(pred_m, true_m))

    Total loss = mean(L_engine) across all engines

    With HuberLoss on normalized log-space data, this is equivalent to minimizing
    a robust version of log(Q-error) with per-metric importance weights.

    Args:
        num_metrics_per_engine: List of number of metrics per engine
            (e.g., [2, 2, 2, 2] for 4 engines with 2 metrics each)
        metric_weights: Importance weights for each metric type
            (e.g., [0.75, 0.25] for time=75%, memory=25%). Normalized internally.
        loss_types: Loss function for each metric type
            (e.g., [HuberLoss(), HuberLoss()] for normalized log-space)
        aggr: How to aggregate across engines ("mean", "sum", None)

    Example:
        # 4 engines, each predicting 2 metrics (time + memory)
        # Using HuberLoss on normalized log-space predictions
        loss_fn = MultiMetricLoss(
            num_metrics_per_engine=[2, 2, 2, 2],
            metric_weights=[0.75, 0.25],  # Prioritize time over memory
            loss_types=[HuberLoss(delta=1.0), HuberLoss(delta=1.0)]
        )
    """

    def __init__(
        self,
        num_metrics_per_engine: list[int],
        metric_weights: list[float] | None = None,
        loss_types: list[Loss] | None = None,
        aggr: str | None = "mean",
        allow_nans: bool = False,
    ) -> None:
        super().__init__(name="MultiMetricLoss", aggr=aggr, allow_nans=allow_nans)

        self.num_metrics_per_engine = num_metrics_per_engine
        self.num_engines = len(num_metrics_per_engine)

        # Assume all engines have same number of metrics
        self.num_metrics = num_metrics_per_engine[0]

        # Default weights: equal for all metrics
        if metric_weights is None:
            self.metric_weights = [1.0] * self.num_metrics
        else:
            self.metric_weights = metric_weights

        # Default loss types: CustomQLoss for all metrics
        if loss_types is None:
            self.loss_types = [CustomQLoss(aggr=None) for _ in range(self.num_metrics)]
        else:
            self.loss_types = loss_types

        # Validate inputs
        if len(self.metric_weights) != self.num_metrics:
            raise ValueError(
                f"metric_weights length ({len(self.metric_weights)}) must match " f"num_metrics ({self.num_metrics})"
            )

        if len(self.loss_types) != self.num_metrics:
            raise ValueError(
                f"loss_types length ({len(self.loss_types)}) must match " f"num_metrics ({self.num_metrics})"
            )

    def forward(self, pred, true):
        """Compute weighted multi-metric loss.

        Args:
            pred: Predictions tensor [batch_size, total_metrics]
            true: Ground truth tensor [batch_size, total_metrics]

        Returns:
            Aggregated loss across all engines and metrics

        Aggregation:
            - Weights are normalized to sum to 1.0 (so metric importance is relative)
            - Loss is averaged across engines (each engine contributes equally)
        """
        # Normalize weights so they sum to 1.0
        weight_sum = sum(self.metric_weights)
        normalized_weights = [w / weight_sum for w in self.metric_weights]

        total_loss = 0.0
        idx = 0

        # Iterate over each engine
        for engine_idx, num_metrics in enumerate(self.num_metrics_per_engine):
            engine_loss = 0.0

            # Iterate over each metric for this engine
            for metric_idx in range(num_metrics):
                # Extract predictions and targets for this specific metric
                pred_metric = pred[:, idx].unsqueeze(1)  # [batch_size, 1]
                true_metric = true[:, idx].unsqueeze(1)  # [batch_size, 1]

                # Compute loss for this metric
                loss_fn = self.loss_types[metric_idx]
                metric_loss = loss_fn(pred_metric, true_metric)

                # Apply normalized weight (metrics within engine sum to 1.0)
                engine_loss = engine_loss + normalized_weights[metric_idx] * metric_loss

                idx += 1

            # Add this engine's weighted loss to total
            total_loss = total_loss + engine_loss

        # Aggregate across engines
        if self.aggr == "mean":
            return total_loss / self.num_engines
        elif self.aggr == "sum":
            return total_loss
        else:
            return total_loss

    def forward_detailed(self, pred, true) -> dict:
        """Compute loss with per-metric breakdown for logging.

        Args:
            pred: Predictions tensor [batch_size, total_metrics]
            true: Ground truth tensor [batch_size, total_metrics]

        Returns:
            Dict with:
                - "total": aggregated loss (same as forward())
                - "per_metric": dict mapping metric_idx -> unweighted loss
                - "per_metric_weighted": dict mapping metric_idx -> weighted loss
        """
        weight_sum = sum(self.metric_weights)
        normalized_weights = [w / weight_sum for w in self.metric_weights]

        total_loss = 0.0
        idx = 0
        per_metric_losses = {}
        per_metric_weighted = {}

        for engine_idx, num_metrics in enumerate(self.num_metrics_per_engine):
            engine_loss = 0.0

            for metric_idx in range(num_metrics):
                pred_metric = pred[:, idx].unsqueeze(1)
                true_metric = true[:, idx].unsqueeze(1)

                loss_fn = self.loss_types[metric_idx]
                metric_loss = loss_fn(pred_metric, true_metric)

                # Track per-metric (accumulate across engines)
                if metric_idx not in per_metric_losses:
                    per_metric_losses[metric_idx] = []
                    per_metric_weighted[metric_idx] = []
                per_metric_losses[metric_idx].append(metric_loss.item())
                per_metric_weighted[metric_idx].append(normalized_weights[metric_idx] * metric_loss.item())

                engine_loss = engine_loss + normalized_weights[metric_idx] * metric_loss
                idx += 1

            total_loss = total_loss + engine_loss

        # Average per-metric across engines
        per_metric_avg = {k: sum(v) / len(v) for k, v in per_metric_losses.items()}
        per_metric_weighted_avg = {k: sum(v) / len(v) for k, v in per_metric_weighted.items()}

        if self.aggr == "mean":
            total_loss = total_loss / self.num_engines

        return {
            "total": total_loss,
            "per_metric": per_metric_avg,
            "per_metric_weighted": per_metric_weighted_avg,
        }


# ----------------------------------------------------------------------------------


class PairwiseRankingLoss(Loss):
    """Pairwise ranking loss to preserve engine ordering.

    For each query and metric (time/memory), ensures that if engine i is truly
    better than engine j (y_i < y_j), then the model predicts pred_i < pred_j.

    Works on normalized log-space predictions (same space as training).
    In log-space, smaller values = faster/cheaper, so we want to preserve
    the ordering: if log(y_i) < log(y_j), then pred_i < pred_j.

    Loss formulations:
        - hinge: max(0, margin - (pred_j - pred_i))  when y_i < y_j
        - logistic: log(1 + exp(pred_i - pred_j))    when y_i < y_j

    Margin weighting (optional):
        Weight pairs by relative margin to prioritize "important" orderings:
        w_ij = min(1, (y_j - y_i) / (y_i + ε))

    Args:
        num_engines: Number of engines (default: 4)
        num_metrics: Number of metrics per engine (default: 2 for time, memory)
        metric_indices: Which metrics to apply ranking to (default: [0] for time only)
        loss_type: "hinge" or "logistic" (default: "hinge")
        margin: Margin for hinge loss (default: 0.1, in normalized log-space)
        use_margin_weighting: Weight pairs by true margin (default: True)
        epsilon: Small constant for numerical stability (default: 1e-6)
        aggr: Aggregation method (default: "mean")

    Example:
        # Apply ranking loss to time predictions only (metric index 0)
        ranking_loss = PairwiseRankingLoss(
            num_engines=4,
            num_metrics=2,
            metric_indices=[0],  # Only time
            loss_type="hinge",
            margin=0.1,
            use_margin_weighting=True,
        )
    """

    def __init__(
        self,
        num_engines: int = 4,
        num_metrics: int = 2,
        metric_indices: list[int] | None = None,
        loss_type: str = "hinge",
        margin: float = 0.1,
        use_margin_weighting: bool = True,
        epsilon: float = 1e-6,
        aggr: str | None = "mean",
        allow_nans: bool = False,
    ) -> None:
        super().__init__(name="PairwiseRankingLoss", aggr=aggr, allow_nans=allow_nans)

        self.num_engines = num_engines
        self.num_metrics = num_metrics
        self.metric_indices = metric_indices if metric_indices is not None else [0]  # Default: time only
        self.loss_type = loss_type
        self.margin = margin
        self.use_margin_weighting = use_margin_weighting
        self.epsilon = epsilon

        if loss_type not in ["hinge", "logistic"]:
            raise ValueError(f"loss_type must be 'hinge' or 'logistic', got {loss_type}")

    def forward(self, pred, true):
        """Compute pairwise ranking loss.

        Args:
            pred: Predictions [batch_size, num_engines * num_metrics] in normalized log-space
            true: Ground truth [batch_size, num_engines * num_metrics] in normalized log-space

        Returns:
            Aggregated ranking loss
        """
        batch_size = pred.shape[0]

        # Reshape to [batch, num_engines, num_metrics]
        pred_grid = pred.view(batch_size, self.num_engines, self.num_metrics)
        true_grid = true.view(batch_size, self.num_engines, self.num_metrics)

        total_loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        num_pairs = 0

        # For each metric we want to rank
        for m_idx in self.metric_indices:
            pred_m = pred_grid[:, :, m_idx]  # [batch, num_engines]
            true_m = true_grid[:, :, m_idx]  # [batch, num_engines]

            # Generate all pairs (i, j) where i < j
            for i in range(self.num_engines):
                for j in range(i + 1, self.num_engines):
                    pred_i = pred_m[:, i]  # [batch]
                    pred_j = pred_m[:, j]  # [batch]
                    true_i = true_m[:, i]  # [batch]
                    true_j = true_m[:, j]  # [batch]

                    # Compute pairwise loss for this pair
                    pair_loss = self._compute_pair_loss(pred_i, pred_j, true_i, true_j)
                    total_loss = total_loss + pair_loss
                    num_pairs += 1

        # Average across all pairs
        if num_pairs > 0:
            total_loss = total_loss / num_pairs

        return total_loss

    def _compute_pair_loss(self, pred_i, pred_j, true_i, true_j):
        """Compute ranking loss for a single pair of engines.

        For samples where true_i < true_j (engine i is better), we want pred_i < pred_j.
        For samples where true_i > true_j (engine j is better), we want pred_j < pred_i.
        For samples where true_i ≈ true_j (tie), we don't penalize either ordering.

        Args:
            pred_i, pred_j: Predictions for engines i, j [batch]
            true_i, true_j: Ground truth for engines i, j [batch]

        Returns:
            Scalar loss for this pair
        """
        # Mask for valid (non-NaN) samples
        valid_mask = ~(torch.isnan(true_i) | torch.isnan(true_j) | torch.isnan(pred_i) | torch.isnan(pred_j))

        if not valid_mask.any():
            return torch.tensor(0.0, device=pred_i.device, dtype=pred_i.dtype)

        pred_i = pred_i[valid_mask]
        pred_j = pred_j[valid_mask]
        true_i = true_i[valid_mask]
        true_j = true_j[valid_mask]

        # Case 1: true_i < true_j (engine i is better) → want pred_i < pred_j
        mask_i_better = true_i < true_j
        # Case 2: true_j < true_i (engine j is better) → want pred_j < pred_i
        mask_j_better = true_j < true_i

        loss = torch.zeros_like(pred_i)

        # Loss when engine i should be predicted as better
        if mask_i_better.any():
            loss[mask_i_better] = self._ranking_loss(
                pred_i[mask_i_better],
                pred_j[mask_i_better],
                true_i[mask_i_better],
                true_j[mask_i_better],
            )

        # Loss when engine j should be predicted as better (swap i and j)
        if mask_j_better.any():
            loss[mask_j_better] = self._ranking_loss(
                pred_j[mask_j_better],
                pred_i[mask_j_better],
                true_j[mask_j_better],
                true_i[mask_j_better],
            )

        # Ties (true_i ≈ true_j): no loss
        # Already initialized to 0

        return loss.mean()

    def _ranking_loss(self, pred_better, pred_worse, true_better, true_worse):
        """Compute ranking loss when we know pred_better should be < pred_worse.

        Args:
            pred_better: Predictions for the truly better engine [n]
            pred_worse: Predictions for the truly worse engine [n]
            true_better: Ground truth for better engine [n]
            true_worse: Ground truth for worse engine [n]

        Returns:
            Per-sample ranking loss [n]
        """
        # Compute margin weighting if enabled
        if self.use_margin_weighting:
            # w_ij = min(1, (y_worse - y_better) / (y_better + ε))
            # In normalized log-space, this captures relative importance
            margin_ratio = (true_worse - true_better) / (torch.abs(true_better) + self.epsilon)
            weights = torch.clamp(margin_ratio, min=0.0, max=1.0)
        else:
            weights = torch.ones_like(pred_better)

        # Compute ranking loss
        # We want pred_better < pred_worse, i.e., (pred_worse - pred_better) > 0
        diff = pred_worse - pred_better  # Should be positive

        if self.loss_type == "hinge":
            # Hinge: max(0, margin - diff) = max(0, margin - (pred_worse - pred_better))
            # Loss is 0 when diff >= margin (correct ordering with margin)
            loss = torch.relu(self.margin - diff)
        else:  # logistic
            # Logistic: log(1 + exp(-diff)) = log(1 + exp(pred_better - pred_worse))
            # Smooth approximation to hinge, always positive
            loss = torch.log1p(torch.exp(-diff))

        return weights * loss


class RankingAugmentedLoss(Loss):
    """Combined regression + ranking loss for multi-metric prediction.

    Combines MultiMetricLoss (regression) with PairwiseRankingLoss (ranking)
    to jointly optimize for prediction accuracy and correct engine ordering.

    Total loss = regression_loss + λ * ranking_loss

    The ranking component helps the model learn to correctly order engines
    even when absolute predictions have errors, which is crucial for routing.

    Args:
        regression_loss: Base regression loss (MultiMetricLoss)
        ranking_loss: Pairwise ranking loss (PairwiseRankingLoss)
        ranking_weight: Weight λ for ranking loss (default: 0.1)
        aggr: Aggregation method (default: "mean")

    Example:
        regression = MultiMetricLoss(
            num_metrics_per_engine=[2, 2, 2, 2],
            metric_weights=[0.5, 0.5],
            loss_types=[HuberLoss(delta=2.0), HuberLoss(delta=2.0)],
        )
        ranking = PairwiseRankingLoss(
            num_engines=4,
            num_metrics=2,
            metric_indices=[0],  # Rank on time only
            loss_type="hinge",
            margin=0.1,
        )
        combined = RankingAugmentedLoss(
            regression_loss=regression,
            ranking_loss=ranking,
            ranking_weight=0.1,
        )
    """

    def __init__(
        self,
        regression_loss: MultiMetricLoss,
        ranking_loss: PairwiseRankingLoss,
        ranking_weight: float = 0.1,
        aggr: str | None = "mean",
        allow_nans: bool = False,
    ) -> None:
        super().__init__(name="RankingAugmentedLoss", aggr=aggr, allow_nans=allow_nans)

        self.regression_loss = regression_loss
        self.ranking_loss = ranking_loss
        self.ranking_weight = ranking_weight

        # Expose num_metrics_per_engine for train_step_multihead detection
        # This tells the training loop to concatenate all head outputs before calling loss
        self.num_metrics_per_engine = regression_loss.num_metrics_per_engine

    def forward(self, pred, true):
        """Compute combined regression + ranking loss.

        Args:
            pred: Predictions [batch_size, num_engines * num_metrics]
            true: Ground truth [batch_size, num_engines * num_metrics]

        Returns:
            Combined loss: regression + λ * ranking
        """
        reg_loss = self.regression_loss(pred, true)
        rank_loss = self.ranking_loss(pred, true)

        return reg_loss + self.ranking_weight * rank_loss

    def forward_detailed(self, pred, true) -> dict:
        """Compute loss with breakdown for logging.

        Returns:
            Dict with:
                - "total": combined loss
                - "regression": regression loss component
                - "ranking": ranking loss component
                - "per_metric": from regression loss
        """
        reg_detailed = self.regression_loss.forward_detailed(pred, true)
        rank_loss = self.ranking_loss(pred, true)

        total = reg_detailed["total"] + self.ranking_weight * rank_loss

        return {
            "total": total,
            "regression": reg_detailed["total"],
            "ranking": rank_loss,
            "per_metric": reg_detailed["per_metric"],
            "per_metric_weighted": reg_detailed["per_metric_weighted"],
        }


# ----------------------------------------------------------------------------------
