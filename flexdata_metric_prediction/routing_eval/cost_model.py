"""
Routing Evaluation System

Minimal classes for evaluating multi-engine query routing based on predicted metrics.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import wandb


# =============================================================================
# Core Data Structures
# =============================================================================


class Task(Enum):
    """Routing tasks."""

    MIN_TIME = "A"  # Unconstrained time minimization
    MIN_COST = "B"  # Unconstrained cost minimization
    MIN_COST_TIME_SLO = "C"  # Cost minimization under time SLO
    MIN_TIME_COST_SLO = "D"  # Time minimization under cost SLO


@dataclass
class QueryResult:
    """Predictions and ground truth for a single query across all engines."""

    query_id: str
    t_true: np.ndarray  # [n_engines] true execution times
    t_pred: np.ndarray  # [n_engines] predicted execution times
    m_true: np.ndarray  # [n_engines] true memory
    m_pred: np.ndarray  # [n_engines] predicted memory
    c_true: np.ndarray  # [n_engines] true cost (derived)
    c_pred: np.ndarray  # [n_engines] predicted cost (derived)


@dataclass
class RoutingMetrics:
    """Evaluation metrics for a routing policy."""

    mean_objective: float
    improvement_vs_best_fixed: float  # Percentage
    mean_regret: float
    p95_regret: float
    n_routed: int = None  # Number of queries where policy made a choice
    n_total: int = None  # Total number of queries
    # Constraint-specific (Tasks C/D only)
    feasibility_rate: float = None
    violation_rate: float = None

    # Should we add median regret?


# =============================================================================
# Cost Model
# =============================================================================

POD_TIERS = {2048: 1, 4096: 2, 8192: 4, 16384: 8}  # MB -> cost multiplier


def get_pod_cost(memory_gb: float) -> int:
    """Get cost multiplier for smallest pod that fits memory requirement."""
    memory_mb = memory_gb * 1024
    for pod_mb in sorted(POD_TIERS.keys()):
        if memory_mb <= pod_mb:
            return POD_TIERS[pod_mb]
    return POD_TIERS[max(POD_TIERS.keys())]


@dataclass
class CostModel:
    """Compute query cost from time, memory, and worker count."""

    # Engine -> num_workers mapping
    workers: dict = field(default_factory=lambda: {"spark-w1": 1, "spark-w4": 4, "presto-w1": 1, "presto-w4": 4})

    # Spark memory conversion params
    spark_mem_fraction: float = 0.75  # previously 75%
    spark_reserved_mb: float = 300.0  # previously 300
    spark_overhead: float = 0.10

    # Presto memory conversion params
    presto_spike: float = 1.1
    presto_headroom: float = 1.1
    presto_f_total: float = 0.75
    presto_f_heap: float = 0.80

    def mvp_pod_gb(self, memory_metric: float, engine: str) -> float:
        """Convert engine-specific memory metric to MVP pod size in GB."""
        if "spark" in engine.lower():
            heap_mb = memory_metric / self.spark_mem_fraction + self.spark_reserved_mb
            return heap_mb * (1 + self.spark_overhead) / 1024
        elif "presto" in engine.lower():
            return (self.presto_spike * self.presto_headroom * memory_metric) / (
                self.presto_f_total * self.presto_f_heap * 1024
            )
        else:
            raise ValueError(f"Unknown engine: {engine}. Expected 'spark' or 'presto' in engine name.")

    def compute_cost(self, time_s: float, memory: float, engine: str) -> float:
        """Compute cost = time * pod_cost * num_workers."""
        if engine not in self.workers:
            raise ValueError(f"Unknown engine: {engine}. Available: {list(self.workers.keys())}")
        pod_gb = self.mvp_pod_gb(memory, engine)
        pod_cost = get_pod_cost(pod_gb)
        return time_s * pod_cost * self.workers[engine]


# =============================================================================
# Routing Policies
# =============================================================================


def oracle_policy(
    results: list[QueryResult],
    objective: str,
    constraint: str = None,
    threshold: float = None,
) -> np.ndarray:
    """Per-query optimal engine selection. Returns [n_queries] engine indices.

    For constrained tasks, picks engine minimizing objective among those meeting constraint.
    Returns -1 for queries where no engine meets the constraint.
    """
    obj_map = {"time": "t_true", "cost": "c_true"}
    if objective not in obj_map:
        raise ValueError(f"Unknown objective: {objective}. Expected 'time' or 'cost'.")

    obj_attr = obj_map[objective]

    # Unconstrained case
    if constraint is None or threshold is None:
        return np.array([np.nanargmin(getattr(r, obj_attr)) for r in results])

    # Constrained case: minimize objective among feasible engines
    con_attr = obj_map[constraint]
    choices = []
    for r in results:
        obj_vals = getattr(r, obj_attr)
        con_vals = getattr(r, con_attr)
        feasible = con_vals <= threshold

        if feasible.any():
            masked = np.where(feasible, obj_vals, np.inf)
            choices.append(np.argmin(masked))
        else:
            choices.append(-1)  # No feasible engine

    return np.array(choices)


def fixed_policy(results: list[QueryResult], engine_idx: int) -> np.ndarray:
    """Always select the same engine."""
    return np.full(len(results), engine_idx)


def random_policy(results: list[QueryResult], n_engines: int, seed: int = 42) -> np.ndarray:
    """Uniform random engine selection."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_engines, size=len(results))


def pred_policy(results: list[QueryResult], objective: str) -> np.ndarray:
    """Select engine minimizing predicted objective."""
    obj_map = {"time": "t_pred", "cost": "c_pred"}
    if objective not in obj_map:
        raise ValueError(f"Unknown objective: {objective}. Expected 'time' or 'cost'.")
    return np.array([np.nanargmin(getattr(r, obj_map[objective])) for r in results])


def pred_slo_policy(
    results: list[QueryResult],
    objective: str,  # "time" or "cost" - what to minimize
    constraint: str,  # "time" or "cost" - what's constrained
    threshold: float,  # Absolute SLO threshold
) -> np.ndarray:
    """Select engine minimizing objective among those meeting constraint (predicted)."""
    if objective not in {"time", "cost"}:
        raise ValueError(f"Unknown objective: {objective}. Expected 'time' or 'cost'.")
    if constraint not in {"time", "cost"}:
        raise ValueError(f"Unknown constraint: {constraint}. Expected 'time' or 'cost'.")

    obj_attr = "t_pred" if objective == "time" else "c_pred"
    con_attr = "t_pred" if constraint == "time" else "c_pred"

    choices = []
    for r in results:
        obj_vals = getattr(r, obj_attr)
        con_vals = getattr(r, con_attr)
        feasible = con_vals <= threshold

        if feasible.any():
            # Among feasible, pick min objective
            masked = np.where(feasible, obj_vals, np.inf)
            choices.append(np.argmin(masked))
        else:
            # No feasible engine - return -1 (infeasible)
            choices.append(-1)

    return np.array(choices)


# =============================================================================
# Evaluator
# =============================================================================


class RoutingEvaluator:
    """Evaluate routing policies on predicted/true metrics."""

    def __init__(self, engines: list[str], cost_model: CostModel = None):
        self.engines = engines
        self.cost_model = cost_model or CostModel()

    def compute_slo_thresholds(
        self, results: list[QueryResult], objective: str, percentiles: list[int] = [50, 75, 90]  # "time" or "cost"
    ) -> dict[int, float]:
        """Compute percentile-based SLO thresholds from oracle values."""
        if objective not in {"time", "cost"}:
            raise ValueError(f"Unknown objective: {objective}. Expected 'time' or 'cost'.")
        obj_attr = "t_true" if objective == "time" else "c_true"
        oracle_vals = [np.nanmin(getattr(r, obj_attr)) for r in results]
        return {p: np.percentile(oracle_vals, p) for p in percentiles}

    def evaluate_policy(
        self,
        results: list[QueryResult],
        choices: np.ndarray,  # [n_queries] engine indices (-1 = infeasible)
        task: Task,
        slo_threshold: float = None,
        oracle_choices: np.ndarray = None,  # For regret computation against constrained oracle
    ) -> RoutingMetrics:
        """Evaluate a routing policy's choices."""

        # Determine objective
        if task in [Task.MIN_TIME, Task.MIN_TIME_COST_SLO]:
            obj_true, obj_pred = "t_true", "t_pred"
        else:
            obj_true, obj_pred = "c_true", "c_pred"

        # Get true objective values for chosen engines
        chosen_obj = np.array([getattr(results[i], obj_true)[c] if c >= 0 else np.nan for i, c in enumerate(choices)])

        # Oracle values - use constrained oracle if provided, else unconstrained
        if oracle_choices is not None:
            oracle_obj = np.array(
                [getattr(results[i], obj_true)[c] if c >= 0 else np.nan for i, c in enumerate(oracle_choices)]
            )
        else:
            oracle_obj = np.array([np.nanmin(getattr(r, obj_true)) for r in results])

        # Best fixed engine
        fixed_means = [np.nanmean([getattr(r, obj_true)[e] for r in results]) for e in range(len(self.engines))]
        best_fixed_mean = min(fixed_means)

        # Core metrics (on non-infeasible queries)
        valid = choices >= 0
        mean_obj = np.nanmean(chosen_obj[valid]) if valid.any() else np.nan

        improvement = (best_fixed_mean - mean_obj) / best_fixed_mean * 100 if best_fixed_mean > 0 else 0

        regret = chosen_obj[valid] / oracle_obj[valid] - 1
        mean_regret = np.nanmean(regret)
        p95_regret = np.nanpercentile(regret, 95)

        metrics = RoutingMetrics(
            mean_objective=mean_obj,
            improvement_vs_best_fixed=improvement,
            mean_regret=mean_regret,
            p95_regret=p95_regret,
            n_routed=int(valid.sum()),
            n_total=len(results),
        )

        # Constraint metrics for Tasks C/D
        if task in [Task.MIN_COST_TIME_SLO, Task.MIN_TIME_COST_SLO] and slo_threshold is not None:
            con_attr = "t_true" if task == Task.MIN_COST_TIME_SLO else "c_true"

            # Feasibility: does any engine meet constraint?
            n_feasible = sum(1 for r in results if np.nanmin(getattr(r, con_attr)) <= slo_threshold)
            metrics.feasibility_rate = n_feasible / len(results)

            # Violation: among queries where router chose an engine, how many violate?
            committed = choices >= 0
            if committed.any():
                violations = sum(
                    1 for i, c in enumerate(choices) if c >= 0 and getattr(results[i], con_attr)[c] > slo_threshold
                )
                metrics.violation_rate = violations / committed.sum()

        return metrics

    def run_evaluation(
        self,
        results: list[QueryResult],
        tasks: list[Task] = None,
        slo_percentiles: list[int] = [50, 75, 90],
    ) -> dict:
        """Run full evaluation across tasks and baselines."""

        tasks = tasks or list(Task)
        n_engines = len(self.engines)
        output = {}

        for task in tasks:
            task_results = {"baselines": {}, "learned": {}, "oracle": {}}

            # Determine objective and constraint
            if task == Task.MIN_TIME:
                obj, con = "time", None
            elif task == Task.MIN_COST:
                obj, con = "cost", None
            elif task == Task.MIN_COST_TIME_SLO:
                obj, con = "cost", "time"
            else:  # MIN_TIME_COST_SLO
                obj, con = "time", "cost"

            # SLO thresholds (if constrained task)
            thresholds = {}
            if con:
                thresholds = self.compute_slo_thresholds(results, con, slo_percentiles)

            # For unconstrained tasks or each SLO level
            slo_levels = list(thresholds.keys()) if thresholds else [None]

            for slo_p in slo_levels:
                slo_key = f"P{slo_p}" if slo_p else "unconstrained"
                threshold = thresholds.get(slo_p)

                level_results = {}

                # Oracle (compute first - needed for regret computation)
                oracle_obj = "time" if task in [Task.MIN_TIME, Task.MIN_TIME_COST_SLO] else "cost"
                oracle_choices = oracle_policy(results, oracle_obj, constraint=con, threshold=threshold)
                level_results["oracle"] = self.evaluate_policy(
                    results, oracle_choices, task, threshold, oracle_choices=oracle_choices
                )

                # Baselines: fixed engines (regret computed vs oracle)
                for e_idx, e_name in enumerate(self.engines):
                    choices = fixed_policy(results, e_idx)
                    level_results[f"fixed_{e_name}"] = self.evaluate_policy(
                        results, choices, task, threshold, oracle_choices=oracle_choices
                    )

                # Baseline: random
                choices = random_policy(results, n_engines)
                level_results["random"] = self.evaluate_policy(
                    results, choices, task, threshold, oracle_choices=oracle_choices
                )

                # Learned router
                if con:
                    choices = pred_slo_policy(results, obj, con, threshold)
                else:
                    choices = pred_policy(results, obj)
                level_results["learned"] = self.evaluate_policy(
                    results, choices, task, threshold, oracle_choices=oracle_choices
                )

                task_results[slo_key] = level_results

            output[task.name] = task_results

        return output


# =============================================================================
# Model Loader (wandb integration)
# =============================================================================


class ModelLoader:
    """Load trained model checkpoint from wandb or local path."""

    def __init__(self, project: str = None, entity: str = None):
        self.project = project
        self.entity = entity

    def load_from_wandb(self, run_id: str, device: str = "cpu") -> dict:
        """Load checkpoint from wandb artifact."""
        api = wandb.Api()
        run = api.run(f"{self.entity}/{self.project}/{run_id}")

        for art in run.logged_artifacts():
            if art.type == "model":
                artifact_dir = art.download()
                return torch.load(Path(artifact_dir) / "best_model.pt", map_location=device)

        raise ValueError(f"No model artifact found in run {run_id}")

    def load_from_path(self, path: str | Path, device: str = "cpu") -> dict:
        """Load checkpoint from local path."""
        return torch.load(path, map_location=device)


# =============================================================================
# Prediction Runner
# =============================================================================


class PredictionRunner:
    """Run predictions on test data and compute costs."""

    def __init__(
        self,
        model: torch.nn.Module,
        normalizer,  # MetricNormalizer
        engines: list[str],
        cost_model: CostModel = None,
        device: str = "cpu",
    ):
        self.model = model.to(device).eval()
        self.normalizer = normalizer
        self.engines = engines
        self.cost_model = cost_model or CostModel()
        self.device = device
        self.n_metrics = len(normalizer.metric_names)

    def run(self, dataloader) -> list[QueryResult]:
        """Run predictions and return QueryResult for each query."""
        results = []

        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.device)

                # Model outputs: {engine: [batch, n_metrics]} (normalized)
                preds_dict = self.model.predict_engines(batch)

                # Stack predictions: [batch, n_engines * n_metrics]
                preds_norm = torch.cat([preds_dict[e] for e in self.engines], dim=1)
                preds_raw = self.normalizer.inverse_transform(preds_norm.cpu().numpy())

                # True labels
                y_true_norm = batch.y
                y_true_raw = self.normalizer.inverse_transform(y_true_norm.cpu().numpy())

                batch_size = y_true_norm.shape[0]
                n_engines = len(self.engines)

                for i in range(batch_size):
                    query_id = batch.query_id[i] if hasattr(batch, "query_id") else f"q_{len(results)}"

                    t_true, t_pred = np.zeros(n_engines), np.zeros(n_engines)
                    m_true, m_pred = np.zeros(n_engines), np.zeros(n_engines)

                    for e_idx, eng in enumerate(self.engines):
                        # Layout: [e0_time, e0_mem, e1_time, e1_mem, ...]
                        t_true[e_idx] = y_true_raw[i, e_idx * self.n_metrics]
                        m_true[e_idx] = y_true_raw[i, e_idx * self.n_metrics + 1]
                        t_pred[e_idx] = preds_raw[i, e_idx * self.n_metrics]
                        m_pred[e_idx] = preds_raw[i, e_idx * self.n_metrics + 1]

                    c_true = np.array(
                        [self.cost_model.compute_cost(t_true[e], m_true[e], eng) for e, eng in enumerate(self.engines)]
                    )
                    c_pred = np.array(
                        [self.cost_model.compute_cost(t_pred[e], m_pred[e], eng) for e, eng in enumerate(self.engines)]
                    )

                    results.append(
                        QueryResult(
                            query_id=query_id,
                            t_true=t_true,
                            t_pred=t_pred,
                            m_true=m_true,
                            m_pred=m_pred,
                            c_true=c_true,
                            c_pred=c_pred,
                        )
                    )

        return results


# =============================================================================
# Main Evaluation Pipeline (Config-Driven)
# =============================================================================


@dataclass
class EvalConfig:
    """Configuration for routing evaluation."""

    config_path: str  # Path to training config YAML
    checkpoint_path: str = None  # Local checkpoint path (preferred)
    wandb_project: str = None
    wandb_entity: str = None
    wandb_run_id: str = None
    normalizer_path: str = None  # Path to saved normalizer JSON
    tasks: list[str] = field(default_factory=lambda: ["A", "B", "C", "D"])
    slo_percentiles: list[int] = field(default_factory=lambda: [50, 75, 90])
    device: str = "cpu"


def run_evaluation(eval_config: EvalConfig) -> dict:
    """Main evaluation pipeline."""
    from flexdata_metric_prediction.utils.read_config import read_yaml_config
    from flexdata_metric_prediction.dataset.gnn_dataloader import GNNDataloader
    from flexdata_metric_prediction.dataset.gnn_dataset import GNNDataset
    from flexdata_metric_prediction.dataset.metric_normalizer import MetricNormalizer
    from flexdata_metric_prediction.dataset.dataset_builder import DatasetBuilder
    from flexdata_metric_prediction.encoder.hint_encoder import HintEncoder
    from flexdata_metric_prediction.model.bottom_up_gnn import MLP, BottomUpGNN
    from flexdata_metric_prediction.model.multi_head_model import MultiHeadModel
    from flexdata_metric_prediction.tree.tree_nodes import NODE_TYPES

    config = read_yaml_config(eval_config.config_path)
    data_config = config["data"]
    model_config = config["model"]

    # 1. Load checkpoint
    loader = ModelLoader(eval_config.wandb_project, eval_config.wandb_entity)
    if eval_config.checkpoint_path:
        checkpoint = loader.load_from_path(eval_config.checkpoint_path, eval_config.device)
    else:
        checkpoint = loader.load_from_wandb(eval_config.wandb_run_id, device=eval_config.device)

    # 2. Build datasets
    query_datasets = []
    for schema_name, schema_config in data_config.items():
        builder = DatasetBuilder(
            plans_dir=schema_config["plan_path"],
            queries_file=schema_config["queries_path"] + "/selected_queries.json",
            metrics_dir=Path(schema_config["labels_path"][list(schema_config["labels_path"].keys())[0]]).parent,
            schema=schema_name,
        )
        engine_configs = {}
        for engine_name in schema_config["labels_path"]:
            engine_configs[engine_name] = (
                list(schema_config["metrics"][engine_name].values()) if "metrics" in schema_config else None
            )
        query_datasets.append(builder.build_dataset(engine_configs, schema_config.get("query_range"), schema_name))

    # 3. Get engines and target_metrics
    first_schema = next(iter(data_config.values()))
    engines = sorted(first_schema["labels_path"].keys())
    target_metrics = {eng: list(first_schema["metrics"][eng].values()) for eng in engines}
    semantic_names = list(first_schema["metrics"][engines[0]].keys())

    # 4. Create encoder and GNN dataset
    encoder = HintEncoder(
        op_mapping="./flexdata_metric_prediction/encoder/opMappingNew.json",
        rel_mapping="./flexdata_metric_prediction/encoder/relMapping.json",
        type_mapping="./flexdata_metric_prediction/encoder/typeMapping.json",
    )
    gnn_dataset = GNNDataset(
        datasets=query_datasets, encoder=encoder, target_metrics=target_metrics, skip_on_error=True
    )

    # 5. Get splits
    val_split = {name: cfg["num_val"] for name, cfg in data_config.items()}
    test_split = {name: cfg["num_test"] for name, cfg in data_config.items()}
    train_data, val_data, test_data = gnn_dataset.get_splits(val_split, test_split)

    # 6. Load or fit normalizer
    if eval_config.normalizer_path:
        normalizer = MetricNormalizer.load(eval_config.normalizer_path)
    else:
        train_labels = np.stack([d.y.squeeze(0).numpy() for d in train_data], axis=0)
        normalizer = MetricNormalizer()
        normalizer.fit(train_labels, engines, semantic_names)

    # Apply normalization
    for data in train_data + val_data + test_data:
        data.y = torch.from_numpy(normalizer.transform(data.y.numpy())).to(data.y.dtype)

    # 7. Build model architecture
    input_dims = {}
    for batch in GNNDataloader(train_data, batch_size=32, shuffle=False):
        for node_type, dim in dict(batch.num_node_features.items()).items():
            input_dims[node_type] = max(input_dims.get(node_type, 0), dim)
    for nt in NODE_TYPES:
        if nt.__name__ not in input_dims:
            input_dims[nt.__name__] = 1

    encoder_model = BottomUpGNN(input_dims, model_config)
    model = MultiHeadModel(encoder_model=encoder_model)

    for eng in engines:
        head_config = model_config["head"].copy()
        head_config["output_dim"] = len(target_metrics[eng])
        model.register_head(eng, MLP(model_config["final_mlp"]["output_dim"], head_config))

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.double()  # Match training dtype

    # 8. Run predictions
    test_dataloader = GNNDataloader(test_data, batch_size=64, shuffle=False)
    cost_model = CostModel()
    runner = PredictionRunner(model, normalizer, engines, cost_model, eval_config.device)
    results = runner.run(test_dataloader)

    # 9. Evaluate routing
    evaluator = RoutingEvaluator(engines, cost_model)
    tasks = [Task[t] for t in eval_config.tasks]
    return evaluator.run_evaluation(results, tasks, eval_config.slo_percentiles)


# =============================================================================
# CLI Entry Point
# =============================================================================


def save_results_csv(results: dict, output_path: str, config: dict = None, run_id: str = None):
    """Save routing evaluation results to CSV with optional config metadata.

    Args:
        results: Routing evaluation results dict
        output_path: Path to save CSV
        config: Optional training config dict to include as metadata
        run_id: Optional wandb run ID
    """
    import pandas as pd

    # Extract config metadata if provided
    config_meta = {}
    if config:
        # Experiment metadata
        if "experiment" in config:
            exp = config["experiment"]
            config_meta["experiment_name"] = exp.get("name", "")
            config_meta["experiment_description"] = exp.get("description", "")
            config_meta["dataset"] = exp.get("dataset", "")
            config_meta["paradigm"] = exp.get("paradigm", "")

        # Training hyperparameters
        if "training" in config:
            train = config["training"]
            config_meta["seed"] = train.get("seed", "")
            config_meta["epochs"] = train.get("epochs", "")
            config_meta["batch_size"] = train.get("batch_size", "")
            config_meta["lr"] = train.get("lr", "")
            config_meta["weight_decay"] = train.get("weight_decay", "")
            config_meta["huber_delta"] = train.get("huber_delta", "")
            # Metric weights
            weights = train.get("metric_weights", {})
            config_meta["weight_time"] = weights.get("time", "")
            config_meta["weight_memory"] = weights.get("memory", "")

        # Model architecture summary
        if "model" in config:
            model = config["model"]
            config_meta["model_type"] = model.get("model_type", "")
            if "final_mlp" in model:
                config_meta["final_mlp_hidden"] = model["final_mlp"].get("hidden_dim", "")
            if "head" in model:
                config_meta["head_hidden"] = model["head"].get("hidden_dim", "")
                config_meta["head_layers"] = model["head"].get("num_layers", "")

    if run_id:
        config_meta["run_id"] = run_id

    rows = [
        {"task": t, "level": lvl, "method": m, **config_meta, **vars(met)}
        for t, tr in results.items()
        for lvl, methods in tr.items()
        for m, met in methods.items()
    ]
    pd.DataFrame(rows).to_csv(output_path, index=False)


if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="Evaluate routing policies")
    parser.add_argument("--config", type=str, required=True, help="Eval config YAML")
    parser.add_argument("--output", type=str, help="Output CSV path")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    eval_config = EvalConfig(**cfg)
    results = run_evaluation(eval_config)

    # Save to CSV if requested
    if args.output:
        save_results_csv(results, args.output)

    # Print summary
    for task, task_results in results.items():
        print(f"\n{'='*60}")
        print(f"Task {task}")
        print(f"{'='*60}")
        for level, methods in task_results.items():
            print(f"\n  {level}:")
            for method, metrics in methods.items():
                routed_str = f"{metrics.n_routed}/{metrics.n_total}" if metrics.n_routed else ""
                print(
                    f"    {method:20s} | MeanObj={metrics.mean_objective:8.2f} | "
                    f"Δ={metrics.improvement_vs_best_fixed:+6.1f}% | "
                    f"Regret={metrics.mean_regret:.3f} | n={routed_str}"
                )
