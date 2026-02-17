"""
Metric Normalizer

Handles log transformation, zero imputation, and per-metric/per-engine z-score normalization.

The normalization pipeline is:
1. Impute zeros/negatives with p5 percentile of positive values
2. Apply log transform: z = log(y)
3. Standardize: z_norm = (z - μ) / σ

The model predicts normalized log-space values. At inference time, inverse transform to get
raw predictions: y_pred = exp(z_norm * σ + μ)

Usage:
    # Fit on training data (raw metrics, NOT log-transformed)
    normalizer = MetricNormalizer()
    normalizer.fit(train_labels, engines, metric_names)

    # Transform labels
    train_labels_norm = normalizer.transform(train_labels)
    val_labels_norm = normalizer.transform(val_labels)

    # Inverse transform predictions back to raw scale
    preds_raw = normalizer.inverse_transform(preds_norm)
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch


@dataclass
class MetricStats:
    """Statistics for a single (engine, metric) pair."""

    mean: float  # Mean of log-transformed values
    std: float  # Std of log-transformed values
    impute_value: float  # Value used to impute zeros/negatives (in raw space)
    n_samples: int  # Number of samples used to compute stats
    n_imputed: int = 0  # Number of zeros/negatives imputed


@dataclass
class MetricNormalizer:
    """Normalizer for multi-metric, multi-engine labels.

    Applies:
    1. Zero/negative imputation with percentile of positive values
    2. Log transformation
    3. Z-score normalization (per-metric, per-engine)

    Attributes:
        stats: Dict mapping (engine, metric) -> MetricStats
        engines: List of engine names (ordered)
        metric_names: List of metric names per engine
        impute_percentile: Percentile for zero/negative imputation (default: 5)
        eps: Small value to prevent log(0) and division by zero
    """

    stats: Dict[Tuple[str, str], MetricStats] = field(default_factory=dict)
    engines: List[str] = field(default_factory=list)
    metric_names: List[str] = field(default_factory=list)
    impute_percentile: float = 5.0
    eps: float = 1e-8
    _fitted: bool = False

    def fit(
        self,
        labels: np.ndarray,
        engines: List[str],
        metric_names: List[str],
    ) -> "MetricNormalizer":
        """Fit normalizer on training data.

        Computes per-(engine, metric) statistics:
        - Imputation value (p5 of positive values in raw space)
        - Mean and std (of log-transformed values after imputation)

        Args:
            labels: Raw labels array, shape [n_samples, n_engines * n_metrics]
                    Layout: [e1_m1, e1_m2, ..., e2_m1, e2_m2, ...]
                    Values should be in RAW space (not log-transformed)
            engines: List of engine names
            metric_names: List of metric names (same for all engines)

        Returns:
            self (for method chaining)
        """
        self.engines = engines
        self.metric_names = metric_names
        n_metrics = len(metric_names)

        logging.info(f"Fitting MetricNormalizer on {len(labels)} samples")
        logging.info(f"  Engines: {engines}")
        logging.info(f"  Metrics: {metric_names}")

        for e_idx, engine in enumerate(engines):
            for m_idx, metric in enumerate(metric_names):
                # Get column index in flattened labels
                col_idx = e_idx * n_metrics + m_idx

                # Extract values for this (engine, metric) pair
                values = labels[:, col_idx].copy()

                # Filter out NaNs for statistics
                valid_mask = ~np.isnan(values)
                valid_values = values[valid_mask]

                if len(valid_values) == 0:
                    logging.warning(f"No valid values for ({engine}, {metric}), using defaults")
                    self.stats[(engine, metric)] = MetricStats(
                        mean=0.0, std=1.0, impute_value=1.0, n_samples=0, n_imputed=0
                    )
                    continue

                # Compute imputation value from positive values
                positive_mask = valid_values > 0
                positive_values = valid_values[positive_mask]

                if len(positive_values) == 0:
                    logging.warning(f"No positive values for ({engine}, {metric}), using 1.0 as impute value")
                    impute_value = 1.0
                else:
                    impute_value = np.percentile(positive_values, self.impute_percentile)

                # Count zeros/negatives to impute
                n_imputed = np.sum(valid_values <= 0)

                # Impute zeros/negatives
                valid_values[valid_values <= 0] = impute_value

                # Apply log transform
                log_values = np.log(valid_values + self.eps)

                # Compute mean and std
                mean = float(np.mean(log_values))
                std = float(np.std(log_values))

                # Prevent division by zero
                if std < self.eps:
                    logging.warning(f"Std for ({engine}, {metric}) is ~0, using 1.0")
                    std = 1.0

                self.stats[(engine, metric)] = MetricStats(
                    mean=mean,
                    std=std,
                    impute_value=float(impute_value),
                    n_samples=len(valid_values),
                    n_imputed=int(n_imputed),
                )

                logging.info(
                    f"  ({engine}, {metric}): μ={mean:.4f}, σ={std:.4f}, "
                    f"impute_p{self.impute_percentile}={impute_value:.4f}, "
                    f"n={len(valid_values)}, n_imputed={n_imputed}"
                )

        self._fitted = True
        return self

    def transform(self, labels: np.ndarray) -> np.ndarray:
        """Transform raw labels to normalized log-space.

        Pipeline: impute → log → z-score

        Args:
            labels: Raw labels, shape [n_samples, n_engines * n_metrics]

        Returns:
            Normalized labels, same shape
        """
        if not self._fitted:
            raise RuntimeError("MetricNormalizer must be fit before transform")

        result = labels.copy()
        n_metrics = len(self.metric_names)

        for e_idx, engine in enumerate(self.engines):
            for m_idx, metric in enumerate(self.metric_names):
                col_idx = e_idx * n_metrics + m_idx
                stats = self.stats[(engine, metric)]

                # Get column
                col = result[:, col_idx]

                # Impute zeros/negatives (keep NaNs as NaN)
                valid_mask = ~np.isnan(col)
                impute_mask = valid_mask & (col <= 0)
                col[impute_mask] = stats.impute_value

                # Log transform (NaN stays NaN)
                col[valid_mask] = np.log(col[valid_mask] + self.eps)

                # Z-score normalize
                col[valid_mask] = (col[valid_mask] - stats.mean) / stats.std

                result[:, col_idx] = col

        return result

    def inverse_transform(self, labels_norm: np.ndarray) -> np.ndarray:
        """Inverse transform normalized predictions back to raw scale.

        Pipeline: un-z-score → exp

        Args:
            labels_norm: Normalized labels, shape [n_samples, n_engines * n_metrics]

        Returns:
            Raw labels, same shape
        """
        if not self._fitted:
            raise RuntimeError("MetricNormalizer must be fit before inverse_transform")

        result = labels_norm.copy()
        n_metrics = len(self.metric_names)

        for e_idx, engine in enumerate(self.engines):
            for m_idx, metric in enumerate(self.metric_names):
                col_idx = e_idx * n_metrics + m_idx
                stats = self.stats[(engine, metric)]

                # Get column
                col = result[:, col_idx]

                # Un-z-score: z_norm * σ + μ
                valid_mask = ~np.isnan(col)
                col[valid_mask] = col[valid_mask] * stats.std + stats.mean

                # Exp to get raw values
                col[valid_mask] = np.exp(col[valid_mask])

                result[:, col_idx] = col

        return result

    def transform_tensor(self, labels: torch.Tensor) -> torch.Tensor:
        """Transform raw labels tensor to normalized log-space.

        Args:
            labels: Raw labels tensor, shape [batch, n_engines * n_metrics]

        Returns:
            Normalized labels tensor, same shape
        """
        # Convert to numpy, transform, convert back
        device = labels.device
        dtype = labels.dtype
        result_np = self.transform(labels.cpu().numpy())
        return torch.from_numpy(result_np).to(device=device, dtype=dtype)

    def inverse_transform_tensor(self, labels_norm: torch.Tensor) -> torch.Tensor:
        """Inverse transform normalized predictions tensor back to raw scale.

        Args:
            labels_norm: Normalized labels tensor, shape [batch, n_engines * n_metrics]

        Returns:
            Raw labels tensor, same shape
        """
        device = labels_norm.device
        dtype = labels_norm.dtype
        result_np = self.inverse_transform(labels_norm.detach().cpu().numpy())
        return torch.from_numpy(result_np).to(device=device, dtype=dtype)

    def get_stats_dict(self) -> Dict[str, Dict[str, float]]:
        """Get stats as a serializable dictionary.

        Returns:
            Dict with format: {"engine/metric": {"mean": ..., "std": ..., ...}}
        """
        result = {}
        for (engine, metric), stats in self.stats.items():
            key = f"{engine}/{metric}"
            result[key] = {
                "mean": stats.mean,
                "std": stats.std,
                "impute_value": stats.impute_value,
                "n_samples": stats.n_samples,
                "n_imputed": stats.n_imputed,
            }
        return result

    def save(self, path: str | Path) -> None:
        """Save normalizer to JSON file.

        Args:
            path: Path to save file
        """
        data = {
            "engines": self.engines,
            "metric_names": self.metric_names,
            "impute_percentile": self.impute_percentile,
            "eps": self.eps,
            "stats": self.get_stats_dict(),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logging.info(f"Saved MetricNormalizer to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "MetricNormalizer":
        """Load normalizer from JSON file.

        Args:
            path: Path to load file

        Returns:
            Loaded MetricNormalizer
        """
        with open(path, "r") as f:
            data = json.load(f)

        normalizer = cls(
            engines=data["engines"],
            metric_names=data["metric_names"],
            impute_percentile=data["impute_percentile"],
            eps=data["eps"],
        )

        # Reconstruct stats
        for key, stats_dict in data["stats"].items():
            engine, metric = key.split("/")
            normalizer.stats[(engine, metric)] = MetricStats(
                mean=stats_dict["mean"],
                std=stats_dict["std"],
                impute_value=stats_dict["impute_value"],
                n_samples=stats_dict["n_samples"],
                n_imputed=stats_dict.get("n_imputed", 0),
            )

        normalizer._fitted = True
        logging.info(f"Loaded MetricNormalizer from {path}")
        return normalizer

    def summary(self) -> str:
        """Return a summary string of the normalizer stats."""
        if not self._fitted:
            return "MetricNormalizer (not fitted)"

        lines = ["MetricNormalizer Statistics:"]
        lines.append(f"  Engines: {self.engines}")
        lines.append(f"  Metrics: {self.metric_names}")
        lines.append(f"  Impute percentile: p{self.impute_percentile}")
        lines.append("")

        for (engine, metric), stats in sorted(self.stats.items()):
            lines.append(f"  {engine}/{metric}:")
            lines.append(f"    μ={stats.mean:.4f}, σ={stats.std:.4f}")
            lines.append(f"    impute={stats.impute_value:.4f}, n={stats.n_samples}, n_imputed={stats.n_imputed}")

        return "\n".join(lines)
