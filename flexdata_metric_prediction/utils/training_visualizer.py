"""
Training Visualizer

Simple, clean visualization utility for multi-metric training progress.
Designed for research-focused plotting with clear axes, titles, and legends.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np


class TrainingVisualizer:
    """Minimal visualizer for multi-metric training progress.

    Tracks and plots:
    - Training/validation loss curves
    - Per-metric Q-error over epochs
    - Per-engine Q-error over epochs

    Usage:
        viz = TrainingVisualizer(engines=["presto-w1", "spark-w1"], metrics=["time", "memory"])

        for epoch in range(n_epochs):
            # ... training ...
            viz.log_epoch(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                tracker_results=tracker.compute("val")
            )

        viz.plot_all(save_dir="./plots")
    """

    def __init__(
        self,
        engines: list[str],
        metrics: list[str],
        figsize: tuple[int, int] = (10, 6),
    ):
        """Initialize visualizer.

        Args:
            engines: List of engine names (e.g., ["presto-w1", "spark-w1"])
            metrics: List of metric names (e.g., ["time", "memory"])
            figsize: Default figure size for plots
        """
        self.engines = engines
        self.metrics = metrics
        self.figsize = figsize

        # Storage for logged values
        self.epochs: list[int] = []
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []

        # Per-metric Q-error (median): {metric_name: [values]}
        self.metric_qerror: dict[str, list[float]] = {m: [] for m in metrics}

        # Per-engine Q-error (median): {engine_name: [values]}
        self.engine_qerror: dict[str, list[float]] = {e: [] for e in engines}

        # Overall Q-error
        self.overall_qerror: list[float] = []

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        tracker_results: Optional[dict] = None,
    ) -> None:
        """Log metrics for one epoch.

        Args:
            epoch: Epoch number
            train_loss: Training loss
            val_loss: Validation loss
            tracker_results: Output from MultiMetricTracker.compute()
        """
        self.epochs.append(epoch)
        self.train_losses.append(float(train_loss))
        self.val_losses.append(float(val_loss))

        if tracker_results:
            # Extract overall Q-error
            overall = tracker_results.get("overall", {}).get("q_error", {})
            self.overall_qerror.append(overall.get("median", np.nan))

            # Extract per-metric Q-error
            for metric in self.metrics:
                metric_data = tracker_results.get("metric", {}).get(metric, {}).get("q_error", {})
                self.metric_qerror[metric].append(metric_data.get("median", np.nan))

            # Extract per-engine Q-error
            for engine in self.engines:
                engine_data = tracker_results.get("engine", {}).get(engine, {}).get("q_error", {})
                self.engine_qerror[engine].append(engine_data.get("median", np.nan))

    def plot_losses(self, ax=None, title: str = "Training Progress"):
        """Plot training and validation loss curves.

        Args:
            ax: Matplotlib axes (creates new figure if None)
            title: Plot title

        Returns:
            Matplotlib axes
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=self.figsize)

        ax.plot(self.epochs, self.train_losses, label="Train Loss", linewidth=2, color="#2E86AB")
        ax.plot(self.epochs, self.val_losses, label="Val Loss", linewidth=2, color="#E94F37")

        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Loss (Huber)", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        return ax

    def plot_metric_qerror(self, ax=None, title: str = "Q-Error by Metric Type"):
        """Plot per-metric Q-error over epochs.

        Args:
            ax: Matplotlib axes (creates new figure if None)
            title: Plot title

        Returns:
            Matplotlib axes
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=self.figsize)

        colors = ["#2E86AB", "#E94F37", "#A23B72", "#F18F01"]
        for i, metric in enumerate(self.metrics):
            color = colors[i % len(colors)]
            ax.plot(
                self.epochs,
                self.metric_qerror[metric],
                label=f"{metric}",
                linewidth=2,
                color=color,
                marker="o",
                markersize=3,
                markevery=max(1, len(self.epochs) // 20),
            )

        # Add overall as dashed line
        ax.plot(
            self.epochs,
            self.overall_qerror,
            label="Overall",
            linewidth=2,
            color="#333333",
            linestyle="--",
        )

        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Q-Error (median)", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=1.0, color="green", linestyle=":", alpha=0.5, label="_nolegend_")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        return ax

    def plot_engine_qerror(self, ax=None, title: str = "Q-Error by Engine"):
        """Plot per-engine Q-error over epochs.

        Args:
            ax: Matplotlib axes (creates new figure if None)
            title: Plot title

        Returns:
            Matplotlib axes
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=self.figsize)

        # Use a colormap for multiple engines
        cmap = plt.cm.tab10
        for i, engine in enumerate(self.engines):
            ax.plot(
                self.epochs,
                self.engine_qerror[engine],
                label=engine,
                linewidth=2,
                color=cmap(i),
                marker="s",
                markersize=3,
                markevery=max(1, len(self.epochs) // 20),
            )

        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Q-Error (median)", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(fontsize=10, framealpha=0.9, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=1.0, color="green", linestyle=":", alpha=0.5, label="_nolegend_")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        return ax

    def plot_all(self, save_dir: Optional[str] = None, show: bool = False) -> None:
        """Generate all plots in a single figure.

        Args:
            save_dir: Directory to save plots (None to skip saving)
            show: Whether to display plots interactively
        """
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        self.plot_losses(axes[0], title="Training Loss")
        self.plot_metric_qerror(axes[1], title="Q-Error by Metric")
        self.plot_engine_qerror(axes[2], title="Q-Error by Engine")

        plt.tight_layout()

        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)

            # Save combined figure
            combined_path = save_path / "training_summary.png"
            fig.savefig(combined_path, dpi=150, bbox_inches="tight")
            logging.info(f"Saved training summary to {combined_path}")

            # Also save individual plots
            for name, plot_fn in [
                ("loss_curves", self.plot_losses),
                ("qerror_by_metric", self.plot_metric_qerror),
                ("qerror_by_engine", self.plot_engine_qerror),
            ]:
                fig_single, ax_single = plt.subplots(figsize=self.figsize)
                plot_fn(ax_single)
                fig_single.savefig(save_path / f"{name}.png", dpi=150, bbox_inches="tight")
                plt.close(fig_single)

        if show:
            plt.show()
        else:
            plt.close(fig)

    def get_summary(self) -> dict:
        """Get summary statistics from training.

        Returns:
            Dict with best values and final values
        """
        if not self.epochs:
            return {}

        return {
            "best_val_loss": min(self.val_losses),
            "best_val_epoch": self.epochs[np.argmin(self.val_losses)],
            "final_val_loss": self.val_losses[-1],
            "best_overall_qerror": min(self.overall_qerror) if self.overall_qerror else None,
            "final_overall_qerror": self.overall_qerror[-1] if self.overall_qerror else None,
            "final_metric_qerror": {m: self.metric_qerror[m][-1] for m in self.metrics},
            "final_engine_qerror": {e: self.engine_qerror[e][-1] for e in self.engines},
        }
