"""Simple model checkpoint manager for saving and loading best models."""

import logging
import shutil
from pathlib import Path

import torch


class ModelCheckpoint:
    """Minimal checkpoint manager for saving best model during training.

    Args:
        checkpoint_dir: Directory to save checkpoints
        monitor_metric: Metric name to monitor (default: "val_loss")
        mode: "min" or "max" - whether lower or higher is better
        save_best_only: Only save when metric improves
        run_id: Unique run identifier for checkpoint filename (e.g., wandb run ID)
    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        monitor_metric: str = "val_loss",
        mode: str = "min",
        save_best_only: bool = True,
        run_id: str | None = None,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.monitor_metric = monitor_metric
        self.mode = mode
        self.save_best_only = save_best_only

        # Use run_id for unique checkpoint filename, or fallback to generic name
        if run_id:
            self.checkpoint_filename = f"best_model_{run_id}.pt"
        else:
            self.checkpoint_filename = "best_model.pt"

        self.best_metric = float("inf") if mode == "min" else float("-inf")
        self.best_epoch = -1

        logging.info(f"Checkpoint manager initialized: {self.checkpoint_dir}")
        logging.info(f"  Monitoring: {monitor_metric} (mode={mode})")
        logging.info(f"  Checkpoint file: {self.checkpoint_filename}")

    def is_better(self, current: float) -> bool:
        """Check if current metric is better than best."""
        if self.mode == "min":
            return current < self.best_metric
        else:
            return current > self.best_metric

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        epoch: int,
        metric_value: float,
        optimizer: torch.optim.Optimizer | None = None,
        metadata: dict | None = None,
    ) -> Path | None:
        """Save checkpoint if metric improved.

        Args:
            model: Model to save
            epoch: Current epoch number
            metric_value: Current metric value
            optimizer: Optional optimizer state
            metadata: Optional additional metadata

        Returns:
            Path to saved checkpoint if saved, None otherwise
        """
        if self.save_best_only and not self.is_better(metric_value):
            return None

        # Update best tracking
        self.best_metric = metric_value
        self.best_epoch = epoch

        # Prepare checkpoint
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "metric_value": metric_value,
            "metric_name": self.monitor_metric,
        }

        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        if metadata is not None:
            checkpoint["metadata"] = metadata

        # Save checkpoint
        checkpoint_path = self.checkpoint_dir / self.checkpoint_filename
        torch.save(checkpoint, checkpoint_path)

        logging.info(f"Checkpoint saved at epoch {epoch}: {self.monitor_metric}={metric_value:.6f} → {checkpoint_path}")

        return checkpoint_path

    def load_best_model(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None = None) -> dict:
        """Load best checkpoint into model.

        Args:
            model: Model to load state into
            optimizer: Optional optimizer to load state into

        Returns:
            Checkpoint metadata dict
        """
        checkpoint_path = self.checkpoint_dir / self.checkpoint_filename

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        model.load_state_dict(checkpoint["model_state_dict"])
        logging.info(f"Loaded best model from epoch {checkpoint['epoch']}")
        logging.info(f"  {checkpoint['metric_name']}={checkpoint['metric_value']:.6f}")

        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            logging.info("  Optimizer state restored")

        return checkpoint

    def get_best_checkpoint_path(self) -> Path:
        """Get path to best checkpoint."""
        return self.checkpoint_dir / self.checkpoint_filename

    def has_checkpoint(self) -> bool:
        """Check if checkpoint exists."""
        return self.get_best_checkpoint_path().exists()

    def cleanup(self, keep_best: bool = True):
        """Remove checkpoint directory.

        Args:
            keep_best: If True, only remove other files but keep best_model.pt
        """
        if not keep_best:
            shutil.rmtree(self.checkpoint_dir)
            logging.info(f"Removed checkpoint directory: {self.checkpoint_dir}")
        else:
            # Remove all files except best_model.pt
            for path in self.checkpoint_dir.glob("*"):
                if path.name != "best_model.pt":
                    if path.is_dir():
                        shutil.rmtree(path)
                    else:
                        path.unlink()
