from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import wandb


def init_wandb(config: dict[str, Any]) -> wandb.sdk.wandb_run.Run | None:
    """Initialize wandb run from config.

    Args:
        config: Full configuration dictionary with 'experiment', 'wandb', and 'training' sections.

    Returns:
        wandb Run object, or None if wandb is disabled.
    """
    wb_cfg = config.get("wandb", {})
    exp_cfg = config.get("experiment", {})
    train_cfg = config.get("training", {})

    # Check if wandb is disabled
    mode = wb_cfg.get("mode", "online")
    if mode == "disabled":
        logging.info("WandB disabled via config")
        return None

    project = wb_cfg.get("project", "flexdata-metric-prediction")
    entity = wb_cfg.get("entity")  # Can be None (uses default)

    # Group: paradigm + dataset_version for organizing related runs
    group = (
        exp_cfg.get("group")
        or f"{exp_cfg.get('paradigm', 'experiment')}_{exp_cfg.get('dataset_version', exp_cfg.get('dataset', 'unknown'))}"
    )

    # Run name: experiment name + seed
    base_name = exp_cfg.get("name", "run")
    seed = train_cfg.get("seed", 0)
    run_name = f"{base_name}-seed{seed}"

    # Tags: combine explicit tags with dataset/paradigm
    tags = list(exp_cfg.get("tags", []))
    if exp_cfg.get("dataset") and exp_cfg["dataset"] not in tags:
        tags.append(exp_cfg["dataset"])
    if exp_cfg.get("paradigm") and exp_cfg["paradigm"] not in tags:
        tags.append(exp_cfg["paradigm"])

    # Initialize wandb
    run = wandb.init(
        project=project,
        entity=entity,
        mode=mode,
        config=config,  # Log full yaml config for reproducibility
        group=group,
        name=run_name,
        notes=exp_cfg.get("description", ""),
        tags=tags,
        job_type=exp_cfg.get("job_type", "train"),
    )

    logging.info(f"WandB initialized: {run.url}")

    return run


def log_metrics(
    epoch: int,
    train_loss: float,
    val_loss: float,
    lr: float,
    tracker_results: dict[str, Any] | None = None,
) -> None:
    """Log metrics to wandb.

    Args:
        epoch: Current epoch number
        train_loss: Training loss
        val_loss: Validation loss
        lr: Current learning rate
        tracker_results: Optional dict from MultiMetricTracker.finish_epoch()["val"]
    """
    if wandb.run is None:
        return

    log_dict = {
        "train/loss": train_loss,
        "val/loss": val_loss,
        "optimizer/lr": lr,
    }

    # Add tracker metrics if provided (validation metrics breakdown)
    if tracker_results:
        _add_tracker_metrics(log_dict, tracker_results, "val")

    wandb.log(log_dict, step=epoch)


def _add_tracker_metrics(log_dict: dict, results: dict, split: str) -> None:
    """Add MultiMetricTracker results to log dict.

    Logs all 45 metrics in canonical naming schema:
    - Overall: metrics/{split}/overall/{metric}/{aggr} (3 metrics)
    - Per engine (combined): metrics/{split}/engine/{engine}/{metric}/{aggr} (4×3=12 metrics)
    - Per metric type: metrics/{split}/metric_type/{type}/{metric}/{aggr} (2×3=6 metrics)
    - Per engine-metric: metrics/{split}/engine/{engine}/{type}/{metric}/{aggr} (4×2×3=24 metrics)

    Total: 3 + 12 + 6 + 24 = 45 metrics

    Args:
        log_dict: Dict to add metrics to
        results: Results dict from tracker.finish_epoch()[split]
        split: Split name ("train", "val", "test")
    """
    prefix = f"metrics/{split}"

    # Overall metrics (e.g., overall q_error median)
    for metric_name, aggr_dict in results.get("overall", {}).items():
        for aggr, value in aggr_dict.items():
            log_dict[f"{prefix}/overall/{metric_name}/{aggr}"] = value

    # Per-engine metrics (combined across time+memory)
    for engine, metric_dict in results.get("engine", {}).items():
        for metric_name, aggr_dict in metric_dict.items():
            for aggr, value in aggr_dict.items():
                log_dict[f"{prefix}/engine/{engine}/{metric_name}/{aggr}"] = value

    # Per-metric-type metrics (time vs memory, aggregated across engines)
    for mtype, metric_dict in results.get("metric", {}).items():
        for metric_name, aggr_dict in metric_dict.items():
            for aggr, value in aggr_dict.items():
                log_dict[f"{prefix}/metric_type/{mtype}/{metric_name}/{aggr}"] = value

    # Per engine-metric breakdown (e.g., presto-w1/time, presto-w1/memory)
    for (engine, mtype), metric_dict in results.get("engine_metric", {}).items():
        for metric_name, aggr_dict in metric_dict.items():
            for aggr, value in aggr_dict.items():
                log_dict[f"{prefix}/engine/{engine}/{mtype}/{metric_name}/{aggr}"] = value


def log_test_results(test_loss: float, tracker_results: dict[str, Any]) -> None:
    """Log final test results to wandb summary.

    Logs all 45 test metrics to wandb summary (not as time series since test
    is only evaluated once at the end).

    Args:
        test_loss: Test loss value
        tracker_results: Results dict from tracker.finish_epoch()["test"]
    """
    if wandb.run is None:
        return

    wandb.summary["test/loss"] = test_loss

    # Add all test metrics to summary using same structure as training metrics
    # This ensures consistent naming between val and test metrics
    summary_dict: dict[str, Any] = {}
    _add_tracker_metrics(summary_dict, tracker_results, "test")

    for key, value in summary_dict.items():
        wandb.summary[key] = value


def log_model_artifact(
    checkpoint_path: Path | str,
    epoch: int,
    val_loss: float,
    config: dict[str, Any],
) -> None:
    """Log model checkpoint as wandb artifact.

    Args:
        checkpoint_path: Path to checkpoint file
        epoch: Epoch number
        val_loss: Validation loss at this checkpoint
        config: Full config dict
    """
    if wandb.run is None:
        return

    wb_cfg = config.get("wandb", {})
    if not wb_cfg.get("log_artifacts", True):
        return

    exp_cfg = config.get("experiment", {})
    seed = config.get("training", {}).get("seed", 0)
    artifact_name = f"{exp_cfg.get('name', 'model')}-seed{seed}"

    try:
        artifact = wandb.Artifact(
            name=artifact_name,
            type="model",
            metadata={
                "epoch": epoch,
                "val_loss": float(val_loss),
                "paradigm": exp_cfg.get("paradigm"),
                "dataset": exp_cfg.get("dataset"),
            },
        )

        # Verify file exists
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            logging.error(f"Checkpoint file does not exist: {checkpoint_path}")
            return

        artifact.add_file(str(checkpoint_path))
        wandb.log_artifact(artifact, aliases=["best", f"epoch-{epoch}"])
        logging.info(f"Logged model artifact: epoch {epoch}, val_loss={val_loss:.4f}")
    except Exception as e:
        logging.error(f"Failed to log model artifact: {e}")


def log_metrics_table(table_rows: list[dict], epoch: int, stage: str = "val") -> None:
    """Log metrics summary table to wandb as a table and CSV artifact.

    Creates a wandb.Table with one row per engine plus overall, showing all
    metric aggregations (time, memory, combined × median, mean, p95).

    Also creates a CSV artifact for long-term archival and easier downloads.

    This is intended to be called once at the end of training per stage
    (val and test) to create a nice summary table.

    Args:
        table_rows: List of row dicts from MultiMetricTracker.to_table()
        epoch: Epoch number (stored in table metadata for reference)
        stage: Split name ("val" or "test")
    """
    if wandb.run is None or not table_rows:
        return

    # Create wandb Table from the rows
    columns = list(table_rows[0].keys())
    table = wandb.Table(columns=columns)

    for row in table_rows:
        table.add_data(*[row.get(col) for col in columns])

    # Log the table to the run history so it appears in the UI (Tables panel).
    # Avoid setting an explicit `step` to prevent step-monotonicity warnings
    # if this is logged after training finished.
    try:
        wandb.log({f"{stage}/metrics_summary": table})
    except Exception:
        # Fallback: put table into summary if log fails for some reason
        wandb.summary[f"{stage}/metrics_summary_table"] = table

    # Save epoch in summary for reference
    wandb.summary[f"{stage}/metrics_summary_epoch"] = epoch

    # Create CSV artifact for long-term archival and easier downloads
    _log_metrics_csv_artifact(table_rows, columns, epoch, stage)


def _log_metrics_csv_artifact(
    table_rows: list[dict],
    columns: list,
    epoch: int,
    stage: str,
) -> None:
    """Create and log a CSV artifact from the metrics table.

    Args:
        table_rows: List of row dicts
        columns: Column names
        epoch: Epoch number
        stage: Split name ("val" or "test")
    """
    import csv
    import tempfile

    if wandb.run is None:
        return

    try:
        # Create a temporary CSV file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(table_rows)
            csv_path = f.name

        # Create artifact with metadata
        run_name = wandb.run.name or "run"
        artifact_name = f"{run_name}-{stage}-metrics"

        artifact = wandb.Artifact(
            name=artifact_name,
            type="metrics_summary",
            metadata={
                "epoch": epoch,
                "stage": stage,
                "num_rows": len(table_rows),
                "columns": columns,
            },
        )
        artifact.add_file(csv_path, name=f"{stage}_metrics_summary.csv")
        wandb.log_artifact(artifact, aliases=[stage, f"epoch-{epoch}"])
        logging.info(f"Logged {stage} metrics CSV artifact: epoch {epoch}")

        # Clean up temp file
        Path(csv_path).unlink(missing_ok=True)

    except Exception as e:
        logging.warning(f"Failed to log metrics CSV artifact: {e}")


def finish_wandb() -> None:
    """Finish wandb run cleanly."""
    if wandb.run is not None:
        wandb.finish()
