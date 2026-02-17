"""
Multi-Metric Training Script using DatasetBuilder

This script demonstrates the simplified training workflow:
1. Load datasets using DatasetBuilder (replaces manual query loading)
2. Create GNNDataset with multi-metric support
3. Train MultiHeadModel with per-engine multi-metric heads
4. Evaluate on validation and test sets

Usage:
    python scripts/train_multi_metric.py --config config/default_configs/combined_dataset_schema_aware.yaml
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from torch.optim.adamw import AdamW

from flexdata_metric_prediction.dataset.dataset_builder import DatasetBuilder
from flexdata_metric_prediction.dataset.gnn_dataloader import GNNDataloader
from flexdata_metric_prediction.dataset.gnn_dataset import GNNDataset
from flexdata_metric_prediction.dataset.metric_normalizer import MetricNormalizer
from flexdata_metric_prediction.encoder.hint_encoder import HintEncoder
from flexdata_metric_prediction.model.bottom_up_gnn import MLP, BottomUpGNN
from flexdata_metric_prediction.model.checkpoint import ModelCheckpoint
from flexdata_metric_prediction.model.eval import evaluate_multi_metric, evaluate_multihead_detailed
from flexdata_metric_prediction.model.loss import HuberLoss, MultiMetricLoss, QLoss
from flexdata_metric_prediction.model.metrics import MultiMetricTracker
from flexdata_metric_prediction.model.multi_head_model import MultiHeadModel
from flexdata_metric_prediction.model.train import train_step_multi_metric
from flexdata_metric_prediction.tree.tree_nodes import NODE_TYPES
from flexdata_metric_prediction.utils.logging_utils import setup_training_logger
from flexdata_metric_prediction.utils.model_summary import model_summary
from flexdata_metric_prediction.utils.read_config import read_yaml_config
from flexdata_metric_prediction.utils.seed import set_global_seed
from flexdata_metric_prediction.utils.timer import Timer
from flexdata_metric_prediction.utils.training_visualizer import TrainingVisualizer
from flexdata_metric_prediction.utils.wandb_utils import (
    finish_wandb,
    init_wandb,
    log_metrics,
    log_metrics_table,
    log_model_artifact,
    log_test_results,
)
from flexdata_metric_prediction.routing_eval.cost_model import (
    CostModel,
    PredictionRunner,
    RoutingEvaluator,
    Task,
    save_results_csv,
)


def setup_logging(log_dir: str | Path | None = None):
    """Configure logging for training.

    Args:
        log_dir: Directory to save logs. If None, only console logging.
    """
    if log_dir is not None:
        return setup_training_logger(log_dir=log_dir)
    else:
        # Fallback to basic console logging
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        return None


def build_datasets(config):
    """Build QueryDatasets using DatasetBuilder.

    Args:
        config: Configuration dictionary

    Returns:
        List of QueryDataset objects, one per schema
    """
    logging.info("Building datasets using DatasetBuilder...")

    query_datasets = []
    data_config = config["data"]

    for schema_name in data_config:
        schema_config = data_config[schema_name]

        logging.info(f"Processing schema: {schema_name}")

        # Initialize DatasetBuilder
        builder = DatasetBuilder(
            plans_dir=schema_config["plan_path"],
            queries_file=schema_config["queries_path"] + "/selected_queries.json",  # Adjust if needed
            metrics_dir=Path(schema_config["labels_path"][list(schema_config["labels_path"].keys())[0]]).parent,
            schema=schema_name,
        )

        # Build engine configs from labels_path
        # metrics config is now dict: {semantic_name: csv_column_name}
        engine_configs = {}
        for engine_name, csv_path in schema_config["labels_path"].items():
            if "metrics" in schema_config and engine_name in schema_config["metrics"]:
                # Extract CSV column names from the semantic mapping
                # e.g., {"time": "elapsedTime", "memory": "peakNodeTotalMemory"} → ["elapsedTime", "peakNodeTotalMemory"]
                metrics_mapping = schema_config["metrics"][engine_name]
                engine_configs[engine_name] = list(metrics_mapping.values())
            else:
                engine_configs[engine_name] = None  # Auto-detect

        # Build dataset
        dataset = builder.build_dataset(
            engine_configs=engine_configs,
            query_range=schema_config.get("query_range"),  # Use query_range if specified
            dataset_name=schema_name,
        )

        query_datasets.append(dataset)
        logging.info(f"  Loaded {len(dataset.queries)} queries for {schema_name}")

    return query_datasets


def build_target_metrics(config, engines: list[str]) -> dict[str, list[str]]:
    """Build target_metrics dict from config.

    Config format (dict-based):
        metrics:
          presto-w1:
            time: elapsedTime      # semantic_name: csv_column_name
            memory: peakNodeTotalMemory

    Returns:
        Dict mapping engine -> list of CSV column names (in semantic order: time, memory, ...)
    """
    data_config = config["data"]
    target_metrics = {}

    # Get first schema config (assumes all schemas use same engines/metrics)
    first_schema = next(iter(data_config.values()))

    if "metrics" in first_schema:
        for engine in engines:
            if engine in first_schema["metrics"]:
                # Config is now {semantic: column_name}, extract column names in order
                metrics_mapping = first_schema["metrics"][engine]
                target_metrics[engine] = list(metrics_mapping.values())
            else:
                logging.warning(f"No metrics specified for {engine}, using default ['time']")
                target_metrics[engine] = ["time"]
    else:
        logging.info("No metrics config found, using single metric ['time'] per engine")
        target_metrics = {engine: ["time"] for engine in engines}

    return target_metrics


def get_engines_from_config(config) -> list[str]:
    """Extract sorted engine list directly from config."""
    first_schema = next(iter(config["data"].values()))
    return sorted(first_schema["labels_path"].keys())


def create_gnn_dataset(query_datasets, encoder, config):
    """Create GNNDataset with multi-metric support.

    Args:
        query_datasets: List of QueryDataset objects
        encoder: HintEncoder instance
        config: Configuration dictionary

    Returns:
        GNNDataset instance
    """
    logging.info("Creating GNN dataset...")

    # Extract engines directly from config (no need to create dataset first)
    engines = get_engines_from_config(config)
    target_metrics = build_target_metrics(config, engines)

    logging.info("Target metrics configuration:")
    for engine, metrics in target_metrics.items():
        logging.info(f"  {engine}: {metrics}")

    # Create GNNDataset with target_metrics
    gnn_dataset = GNNDataset(
        datasets=query_datasets, encoder=encoder, target_metrics=target_metrics, skip_on_error=True
    )

    logging.info(f"GNN dataset created:")
    logging.info(f"  Engines: {gnn_dataset.engines}")
    logging.info(f"  Total graphs: {sum(len(gnn_dataset.data[d]) for d in gnn_dataset.data)}")

    return gnn_dataset


def extract_labels_from_data(data_list: list) -> np.ndarray:
    """Extract labels from GNN data list as numpy array.

    Args:
        data_list: List of HeteroData objects with .y attribute

    Returns:
        numpy array of shape [n_samples, n_labels]
    """
    labels = []
    for data in data_list:
        # data.y is shape [1, n_labels] - squeeze to [n_labels]
        labels.append(data.y.squeeze(0).numpy())
    return np.stack(labels, axis=0)


def apply_normalization_to_data(data_list: list, normalizer: MetricNormalizer) -> list:
    """Apply normalizer to labels in GNN data list (in-place modification).

    Args:
        data_list: List of HeteroData objects with .y attribute
        normalizer: Fitted MetricNormalizer

    Returns:
        Same list with normalized labels
    """
    for data in data_list:
        # Get raw labels, transform, update
        raw_labels = data.y.numpy()  # [1, n_labels]
        norm_labels = normalizer.transform(raw_labels)
        data.y = torch.from_numpy(norm_labels).to(data.y.dtype)
    return data_list


def build_model(gnn_dataset, train_data, config):
    """Build MultiHeadModel with per-engine heads.

    Args:
        gnn_dataset: GNNDataset instance
        train_data: Training dataset (for inferring input dimensions)
        config: Configuration dictionary

    Returns:
        MultiHeadModel instance
    """
    logging.info("Building model...")

    model_config = config["model"]

    # Infer input dimensions from training data
    # Scan ALL training data to find maximum dimension for each node type
    input_dims = {}
    train_dl = GNNDataloader(train_data, batch_size=32, shuffle=False)
    for batch in train_dl:
        batch_dims = dict(batch.num_node_features.items())
        for node_type, dim in batch_dims.items():
            if node_type not in input_dims:
                input_dims[node_type] = dim
            else:
                # Take maximum dimension seen across all batches
                input_dims[node_type] = max(input_dims[node_type], dim)

    # Add missing node types (set to dimension 1)
    for nodeType in NODE_TYPES:
        if nodeType.__name__ not in input_dims:
            logging.warning(f"{nodeType.__name__} not in training set. Setting feature dimension to 1.")
            input_dims[nodeType.__name__] = 1

    logging.info(f"Inferred input dimensions: {input_dims}")

    # Build encoder
    if model_config["model_type"] == "BottomUpGNN":
        encoder = BottomUpGNN(input_dims, model_config)
    else:
        raise NotImplementedError(f"Model type {model_config['model_type']} not supported yet")

    # Determine number of metrics per engine
    num_metrics_per_engine = [len(gnn_dataset.target_metrics[engine]) for engine in gnn_dataset.engines]

    # Build heads (one per engine)
    # TODO: Update head output_dim based on num_metrics_per_engine
    # For now, use config value (will need to fix this in next step)
    heads = []
    for i, engine in enumerate(gnn_dataset.engines):
        num_metrics = num_metrics_per_engine[i]
        head_config = model_config["head"].copy()
        head_config["output_dim"] = num_metrics  # Override config value

        head = MLP(model_config["final_mlp"]["output_dim"], head_config)
        heads.append(head)
        logging.info(f"  Created head for {engine}: output_dim={num_metrics}")

    # Create MultiHeadModel and register heads
    model = MultiHeadModel(encoder_model=encoder)
    for head, engine in zip(heads, gnn_dataset.engines):
        model.register_head(engine, head)

    logging.info(model_summary(model))

    return model


def train(config):
    """Main training function.

    Args:
        config: Configuration dictionary loaded from YAML
    """
    # Setup logging with file output if configured
    train_config = config["training"]
    log_dir = train_config.get("log_dir")
    logger = setup_logging(log_dir=log_dir)

    if logger is not None:
        logging.info(f"Logs will be saved to: {logger.get_log_path()}")

    # Set random seed for reproducibility
    seed = train_config.get("seed", 42)
    set_global_seed(seed)
    logging.info(f"Random seed set to: {seed}")

    # Initialize wandb for experiment tracking
    wandb_run = init_wandb(config)

    logging.info("=" * 70)
    logging.info("Multi-Metric Training Script")
    logging.info("=" * 70)

    # Extract config sections
    data_config = config["data"]
    enc_config = config["encoder"]
    model_config = config["model"]
    train_config = config["training"]

    # 1. Build datasets using DatasetBuilder
    query_datasets = build_datasets(config)

    # 2. Create encoder
    logging.info("Creating encoder...")
    if enc_config["encoder"] == "hintEncoder":
        encoder = HintEncoder(
            op_mapping="./flexdata_metric_prediction/encoder/opMappingNew.json",
            rel_mapping="./flexdata_metric_prediction/encoder/relMapping.json",
            type_mapping="./flexdata_metric_prediction/encoder/typeMapping.json",
        )
    else:
        raise RuntimeError(f"Unknown encoder: {enc_config['encoder']}")

    # 3. Create GNNDataset with multi-metric support
    gnn_dataset = create_gnn_dataset(query_datasets, encoder, config)

    # 4. Create train/val/test splits
    logging.info("Creating data splits...")
    val_split = {}
    test_split = {}
    for dataset_name in data_config:
        val_split[dataset_name] = data_config[dataset_name]["num_val"]
        test_split[dataset_name] = data_config[dataset_name]["num_test"]

    train_data, val_data, test_data = gnn_dataset.get_splits(val_split, test_split)

    logging.info(f"Data splits:")
    logging.info(f"  Train: {len(train_data)} samples")
    logging.info(f"  Val: {len(val_data)} samples")
    logging.info(f"  Test: {len(test_data)} samples")

    # 4b. Fit MetricNormalizer on training data and apply to all splits
    logging.info("Fitting MetricNormalizer on training data...")

    # Get semantic metric names from config
    first_schema = next(iter(config["data"].values()))
    first_engine = next(iter(first_schema["metrics"].keys()))
    semantic_names = list(first_schema["metrics"][first_engine].keys())  # e.g., ["time", "memory"]

    # Extract raw labels from training data
    train_labels_raw = extract_labels_from_data(train_data)
    logging.info(f"  Raw label shape: {train_labels_raw.shape}")
    logging.info(f"  Raw label range: [{np.nanmin(train_labels_raw):.2f}, {np.nanmax(train_labels_raw):.2f}]")

    # Fit normalizer
    normalizer = MetricNormalizer(impute_percentile=5.0)
    normalizer.fit(train_labels_raw, gnn_dataset.engines, semantic_names)
    logging.info(normalizer.summary())

    # Apply normalization to all splits
    logging.info("Applying normalization to train/val/test splits...")
    train_data = apply_normalization_to_data(train_data, normalizer)
    val_data = apply_normalization_to_data(val_data, normalizer)
    test_data = apply_normalization_to_data(test_data, normalizer)

    # Verify normalization
    train_labels_norm = extract_labels_from_data(train_data)
    logging.info(f"  Normalized label range: [{np.nanmin(train_labels_norm):.2f}, {np.nanmax(train_labels_norm):.2f}]")

    # 5. Build model
    model = build_model(gnn_dataset, train_data, config)

    # 6. Setup training
    logging.info("Setting up training...")

    # Import torch for optimizer, scheduler, device, dtype
    import torch

    # Determine number of metrics per engine
    num_metrics_per_engine = [len(gnn_dataset.target_metrics[engine]) for engine in gnn_dataset.engines]

    # Loss function - always multi-metric mode
    logging.info("Using MultiMetricLoss for multi-metric training")

    # Extract metric weights from config (keyed by semantic names: time, memory)
    weight_config = train_config.get("metric_weights", {})

    # semantic_names already defined above during normalization fitting

    # Build weights list in same order as semantic names
    metric_weights = [weight_config.get(name, 1.0) for name in semantic_names]

    logging.info(f"  Metric weights: {dict(zip(semantic_names, metric_weights))}")

    # Get Huber delta from config
    huber_delta = train_config.get("huber_delta", 1.0)

    # Create loss functions for each metric type (one per semantic metric)
    num_metrics = len(semantic_names)
    loss_types = [HuberLoss(delta=huber_delta, aggr="mean") for _ in range(num_metrics)]

    loss = MultiMetricLoss(
        num_metrics_per_engine=num_metrics_per_engine,
        metric_weights=metric_weights,
        loss_types=loss_types,
        aggr="mean",
    )

    # Optimizer with weight decay from config
    weight_decay = train_config.get("weight_decay", 0.0)
    if train_config["optimizer"] == "AdamW":
        optimizer = AdamW(model.parameters(), lr=train_config["lr"], weight_decay=weight_decay)
    else:
        raise RuntimeError(f"Unknown optimizer: {train_config['optimizer']}")
    logging.info(f"  Weight decay: {weight_decay}")

    # Learning rate scheduler from config
    sched_cfg = train_config.get("scheduler", {})
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=sched_cfg.get("mode", "min"),
        factor=sched_cfg.get("factor", 0.5),
        patience=sched_cfg.get("patience", 25),
        cooldown=sched_cfg.get("cooldown", 25),
    )
    logging.info(
        f"  Scheduler: ReduceLROnPlateau (factor={sched_cfg.get('factor', 0.5)}, patience={sched_cfg.get('patience', 25)})"
    )

    # Multi-metric tracker for detailed metrics at engine/metric/overall levels
    # Pass normalizer so Q-error is computed on raw (not normalized) values
    tracker = MultiMetricTracker(
        engines=gnn_dataset.engines,
        metric_names=semantic_names,
        metric_fns={"q_error": QLoss(aggr=None)},
        normalizer=normalizer,
    )

    # Dataloaders
    train_dl = GNNDataloader(train_data, train_config["batch_size"], shuffle=True)
    val_dl = GNNDataloader(val_data, batch_size=256, shuffle=False)
    test_dl = GNNDataloader(test_data, batch_size=256, shuffle=False)

    # 7. Training loop
    logging.info("=" * 70)
    logging.info("Starting training")
    logging.info("=" * 70)
    logging.info(f"Device: {train_config['device']}")
    logging.info(f"Epochs: {train_config['epochs']}")
    logging.info(f"Batch size: {train_config['batch_size']}")
    logging.info(f"Learning rate: {train_config['lr']}")

    # Gradient clipping from config
    grad_clip_norm = train_config.get("grad_clip_norm")
    if grad_clip_norm is not None:
        logging.info(f"  Gradient clipping: {grad_clip_norm}")

    # Early stopping patience from config
    early_stopping_patience = train_config.get("early_stopping_patience", 100)
    logging.info(f"  Early stopping patience: {early_stopping_patience}")

    # Convert device and dtype strings to torch objects
    device = torch.device(train_config["device"])
    dtype = getattr(torch, train_config["dtype"])

    epoch_timer = Timer()
    best_val_loss = float("inf")
    best_val_epoch = -1  # Track which epoch had best validation loss

    # Initialize checkpoint manager (use wandb run ID for unique filename)
    checkpoint = ModelCheckpoint(
        checkpoint_dir=train_config.get("checkpoint_dir", "./checkpoints"),
        monitor_metric=train_config.get("monitor_metric", "val_loss"),
        mode="min",
        save_best_only=train_config.get("save_best_only", True),
        run_id=wandb_run.id if wandb_run else None,
    )

    # Initialize visualizer for training progress plots
    visualizer = TrainingVisualizer(
        engines=gnn_dataset.engines,
        metrics=semantic_names,
    )

    for epoch in range(train_config["epochs"]):
        epoch_timer.start("Training epoch")

        tracker.start_epoch()

        # Train step (with optional gradient clipping)
        train_loss = train_step_multi_metric(
            model,
            optimizer,
            loss,
            train_dl,
            tracker,  # Pass tracker for metrics collection
            device,
            dtype,
            grad_clip=grad_clip_norm,
        )

        # Validation step
        val_loss = evaluate_multi_metric(
            model,
            val_dl,
            loss,
            tracker,  # Pass tracker for metrics collection
            device,
            dtype,
        )

        # Update scheduler and compute metrics
        scheduler.step(val_loss)
        epoch_results = tracker.finish_epoch(epoch)

        # Log to visualizer (use results from finish_epoch)
        visualizer.log_epoch(
            epoch=epoch,
            train_loss=train_loss.item() if torch.is_tensor(train_loss) else float(train_loss),
            val_loss=val_loss.item() if torch.is_tensor(val_loss) else float(val_loss),
            tracker_results=epoch_results.get("val", {}),
        )

        # Log metrics to wandb
        log_metrics(
            epoch=epoch,
            train_loss=train_loss.item() if torch.is_tensor(train_loss) else float(train_loss),
            val_loss=val_loss.item() if torch.is_tensor(val_loss) else float(val_loss),
            lr=optimizer.param_groups[0]["lr"],
            tracker_results=epoch_results.get("val", {}),
        )

        # Track best model and epoch
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_epoch = epoch
            logging.info(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f} (best)")
        else:
            logging.info(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        # Save checkpoint if improved
        ckpt_path = checkpoint.save_checkpoint(
            model=model,
            epoch=epoch,
            metric_value=val_loss,
            optimizer=optimizer,
            metadata={
                "train_loss": train_loss.item() if torch.is_tensor(train_loss) else float(train_loss),
                "engines": gnn_dataset.engines,
                "metric_names": semantic_names,
            },
        )

        # Log model artifact to wandb if checkpoint was saved
        if ckpt_path is not None:
            log_model_artifact(ckpt_path, epoch, val_loss, config)

        # Log detailed metrics every 10 epochs (including per-metric loss breakdown)
        if epoch % 10 == 0:
            # Get per-metric loss breakdown
            detailed = evaluate_multihead_detailed(model, val_dl, loss, device, dtype)
            per_metric_str = ", ".join(
                f"{semantic_names[k]}={v:.4f}" for k, v in sorted(detailed["per_metric"].items())
            )
            logging.info(f"  Per-metric loss (unweighted): {per_metric_str}")
            logging.info(tracker.report(epoch, "val"))

        epoch_timer.lap()

        # Early stopping check
        if best_val_epoch >= 0 and (epoch - best_val_epoch) >= early_stopping_patience:
            logging.info(
                f"Early stopping at epoch {epoch} (no improvement for {early_stopping_patience} epochs, "
                f"best was epoch {best_val_epoch} with val_loss={best_val_loss:.4f})"
            )
            break

    # Log final validation metrics table to wandb
    log_metrics_table(tracker.to_table("val"), best_val_epoch if best_val_epoch >= 0 else epoch, "val")

    # 8. Test evaluation with best model
    if len(test_dl) > 0:
        logging.info("=" * 70)
        logging.info("Evaluating on test set with best model")
        logging.info("=" * 70)

        # Load best model checkpoint
        if checkpoint.has_checkpoint():
            checkpoint.load_best_model(model)
            logging.info(f"Loaded best model from epoch {best_val_epoch}")
        else:
            logging.warning("No checkpoint found, using final model state")

        tracker.start_epoch()

        test_loss = evaluate_multi_metric(
            model,
            test_dl,
            loss,
            tracker,
            device,
            dtype,
            split="test",
        )

        test_results = tracker.finish_epoch(best_val_epoch + 1)

        logging.info(f"Test loss: {test_loss:.4f}")
        logging.info(tracker.report(best_val_epoch + 1, "test"))

        # Log test results to wandb summary
        log_test_results(test_loss, test_results.get("test", {}))

        # Log final test metrics table
        log_metrics_table(tracker.to_table("test"), best_val_epoch + 1, "test")

    # 9. Routing Evaluation (if enabled in config)
    eval_config = config.get("evaluation", {})
    if eval_config.get("enabled", False) and len(test_dl) > 0:
        logging.info("=" * 70)
        logging.info("Running Routing Evaluation (Cost Model)")
        logging.info("=" * 70)

        # Ensure best model is loaded
        if checkpoint.has_checkpoint():
            checkpoint.load_best_model(model)

        # Initialize cost model and prediction runner
        cost_model = CostModel()
        runner = PredictionRunner(
            model=model,
            normalizer=normalizer,
            engines=gnn_dataset.engines,
            cost_model=cost_model,
            device=str(device),
        )

        # Run predictions on test set
        logging.info("Running predictions on test set...")
        query_results = runner.run(test_dl)
        logging.info(f"Generated {len(query_results)} query results")

        # Initialize evaluator
        evaluator = RoutingEvaluator(gnn_dataset.engines, cost_model)

        # Get tasks from config (default to all)
        task_names = eval_config.get("tasks", ["MIN_TIME", "MIN_COST", "MIN_COST_TIME_SLO", "MIN_TIME_COST_SLO"])
        tasks = [Task[t] for t in task_names]
        slo_percentiles = eval_config.get("slo_percentiles", [50, 75, 90])

        logging.info(f"Evaluating tasks: {[t.name for t in tasks]}")
        logging.info(f"SLO percentiles: {slo_percentiles}")

        # Run evaluation
        routing_results = evaluator.run_evaluation(query_results, tasks, slo_percentiles)

        # Save results to CSV with run_id
        run_id = wandb_run.id if wandb_run else "local"
        output_dir = Path(eval_config.get("output_dir", "./routing_eval_results"))
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"routing_eval_{run_id}.csv"

        save_results_csv(routing_results, str(output_path), config=config, run_id=run_id)
        logging.info(f"Routing evaluation results saved to: {output_path}")

        # Log summary
        for task_name, task_results in routing_results.items():
            logging.info(f"\nTask {task_name}:")
            for level, methods in task_results.items():
                learned = methods.get("learned")
                oracle = methods.get("oracle")
                if learned and oracle:
                    logging.info(
                        f"  {level}: Learned MeanObj={learned.mean_objective:.2f}, "
                        f"Δ={learned.improvement_vs_best_fixed:+.1f}%, "
                        f"Regret={learned.mean_regret:.3f} (p95={learned.p95_regret:.3f}), "
                        f"Oracle={oracle.mean_objective:.2f}"
                    )

        # Log to wandb if available
        if wandb_run:
            import wandb
            # Log CSV as artifact
            artifact = wandb.Artifact(
                name=f"routing_eval_{run_id}",
                type="evaluation",
                description="Routing evaluation results",
            )
            artifact.add_file(str(output_path))
            wandb_run.log_artifact(artifact)

            # Log detailed metrics to wandb for learned router
            for task_name, task_results in routing_results.items():
                for level, methods in task_results.items():
                    learned = methods.get("learned")
                    if learned:
                        prefix = f"routing/{task_name}/{level}"
                        # Core metrics
                        wandb_run.summary[f"{prefix}/mean_objective"] = learned.mean_objective
                        wandb_run.summary[f"{prefix}/improvement_vs_best_fixed"] = learned.improvement_vs_best_fixed
                        wandb_run.summary[f"{prefix}/mean_regret"] = learned.mean_regret
                        wandb_run.summary[f"{prefix}/p95_regret"] = learned.p95_regret
                        # Constraint metrics (if applicable)
                        if learned.feasibility_rate is not None:
                            wandb_run.summary[f"{prefix}/feasibility_rate"] = learned.feasibility_rate
                        if learned.violation_rate is not None:
                            wandb_run.summary[f"{prefix}/violation_rate"] = learned.violation_rate

            # Log a summary table for easy comparison
            routing_table_data = []
            for task_name, task_results in routing_results.items():
                for level, methods in task_results.items():
                    learned = methods.get("learned")
                    oracle = methods.get("oracle")
                    if learned:
                        routing_table_data.append([
                            task_name, 
                            level, 
                            learned.mean_objective,
                            learned.improvement_vs_best_fixed,
                            learned.mean_regret,
                            learned.p95_regret,
                            oracle.mean_objective if oracle else None,
                        ])
            
            routing_table = wandb.Table(
                columns=["task", "level", "mean_obj", "improvement_%", "mean_regret", "p95_regret", "oracle_obj"],
                data=routing_table_data
            )
            wandb_run.log({"routing/summary_table": routing_table})

    logging.info("=" * 70)
    logging.info("Training complete!")
    logging.info("=" * 70)

    # Generate training progress plots
    try:
        visualizer.plot_all(save_dir="./plots", show=False)
        logging.info(f"Training summary: {visualizer.get_summary()}")
    except ImportError:
        logging.warning("matplotlib not available, skipping plot generation")

    # Finish wandb run
    finish_wandb()


def main():
    parser = argparse.ArgumentParser(description="Train multi-metric query performance model")
    parser.add_argument(
        "--config",
        type=str,
        default="config/default_configs/combined_dataset_schema_aware.yaml",
        help="Path to config YAML file",
    )
    args = parser.parse_args()

    # Load config
    config = read_yaml_config(args.config)

    # Run training
    train(config)


if __name__ == "__main__":
    main()
