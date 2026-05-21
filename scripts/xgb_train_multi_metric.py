import os
os.chdir('/Users/shazz/code/flexdata-metric-prediction')

import argparse
import json
import logging
import itertools
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
import xgboost as xgb
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


from flexdata_metric_prediction.dataset.gnn_dataset import GNNDataset
from flexdata_metric_prediction.dataset.metric_normalizer import MetricNormalizer
from flexdata_metric_prediction.encoder.hint_encoder import HintEncoder
from flexdata_metric_prediction.tree.tree_nodes import NODE_TYPES as NODE_TYPE_CLASSES
from flexdata_metric_prediction.utils.read_config import read_yaml_config
from flexdata_metric_prediction.utils.seed import set_global_seed

from scripts.train_multi_metric import (
    build_datasets,
    create_gnn_dataset,
    extract_labels_from_data,
    apply_normalization_to_data,
)

# String node-type names in a fixed order (NODE_TYPES is a list of classes).
NODE_TYPE_NAMES: List[str] = [cls.__name__ for cls in NODE_TYPE_CLASSES]

# =============================================================================
# Featurization
# =============================================================================
 
def discover_encoding_dims(graphs) -> Dict[str, int]:
    """Per-node-type encoding dim, scanned from the graphs themselves.
 
    Needed because some graphs may lack certain node types (e.g. no
    LiteralNode); zero-padding a fixed-length vector requires knowing
    the expected dim per type.
    """
    dims = {nt: 0 for nt in NODE_TYPE_NAMES}
    for g in graphs:
        for nt in NODE_TYPE_NAMES:
            if nt in g.node_types and dims[nt] == 0:
                dims[nt] = int(g[nt].x.shape[1])
        if all(d > 0 for d in dims.values()):
            break
    return dims
 
 
def featurize(data, dims: Dict[str, int]) -> np.ndarray:
    """HeteroData -> fixed-size numpy feature vector.
 
    Layout per node type (in NODE_TYPE_NAMES order):
        [count, sum(d), depth_mean, depth_max]
    Global (appended once):
        [total_nodes, global_max_depth]
    Missing types contribute zero-padded slots so vector length is invariant.
    """
    parts: List[np.ndarray] = []
    total_nodes = 0
    global_max_depth = 0
 
    for nt in NODE_TYPE_NAMES:
        d = dims[nt]
        present = nt in data.node_types and data[nt].x.shape[0] > 0
 
        if present:
            x = data[nt].x.detach().cpu().numpy().astype(np.float32)
            depth = data[nt].depth.detach().cpu().numpy().astype(np.float32)
            n = x.shape[0]
            total_nodes += n
            global_max_depth = max(global_max_depth, int(depth.max()))
            parts.extend([
                np.array([n], dtype=np.float32),
                x.sum(axis=0),
                np.array([float(depth.mean()), float(depth.max())], dtype=np.float32),
            ])
        else:
            parts.extend([
                np.array([0], dtype=np.float32),
                np.zeros(d, dtype=np.float32),
                np.array([0.0, 0.0], dtype=np.float32),
            ])
 
    parts.append(np.array([total_nodes, global_max_depth], dtype=np.float32))
    return np.concatenate(parts)
 
 
def featurize_all(graphs, dims: Dict[str, int]) -> np.ndarray:
    return np.stack([featurize(g, dims) for g in graphs], axis=0)
 
 
def extract_label_column(graphs, col_idx: int) -> np.ndarray:
    """One scalar label per graph at flat index col_idx (post-normalization)."""
    return np.array([float(g.y[0, col_idx].item()) for g in graphs], dtype=np.float32)

# =============================================================================
# Evaluation helpers
# =============================================================================
 
def q_error(y_true_raw: np.ndarray, y_pred_raw: np.ndarray) -> dict:
    """Q-error = max(y/y_hat, y_hat/y), in raw (un-normalized) space."""
    eps = 1e-6
    yt = np.maximum(y_true_raw, eps)
    yp = np.maximum(y_pred_raw, eps)
    q = np.maximum(yt / yp, yp / yt)
    return {
        "q_median": float(np.median(q)),
        "q_mean":   float(np.mean(q)),
        "q_90":     float(np.percentile(q, 90)),
        "q_99":     float(np.percentile(q, 99)),
        "q_max":    float(np.max(q)),
    }
 
 
def evaluate_target(
    y_true_norm: np.ndarray, y_pred_norm: np.ndarray,
    y_true_raw:  np.ndarray, y_pred_raw:  np.ndarray,
) -> dict:
    """Both normalized-space regression (comparable to GNN loss) and
    raw-space Q-error (standard cost-model metric)."""
    return {
        "mae_norm":  float(mean_absolute_error(y_true_norm, y_pred_norm)),
        "rmse_norm": float(np.sqrt(mean_squared_error(y_true_norm, y_pred_norm))),
        "r2_norm":   float(r2_score(y_true_norm, y_pred_norm)),
        **q_error(y_true_raw, y_pred_raw),
    }
 
 
def inverse_transform_column(
    normalizer: MetricNormalizer,
    y_norm: np.ndarray,
    flat_idx: int,
    total_cols: int,
) -> np.ndarray:
    """Invert a single flat column of normalized predictions.
 
    MetricNormalizer.inverse_transform expects the full [n, total_labels]
    layout. We zero-pad the other columns, invert, and slice out the column
    we care about.
    """
    full = np.zeros((len(y_norm), total_cols), dtype=np.float32)
    full[:, flat_idx] = y_norm
    inv = normalizer.inverse_transform(full)
    return inv[:, flat_idx]


# -------------------------------------------------------------------------
# 1. Config + seed (reproduces GNN splits when seed matches)
# -------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="config/xgb_lcm_baseline_config.yaml")
args = parser.parse_args()

config = read_yaml_config(args.config)
data_config  = config["data"]
model_config = config["model"]

seed = config.get("seed", 42)
set_global_seed(seed)
logging.info(f"Seed set to {seed}")


# -------------------------------------------------------------------------
# 2. Datasets + encoder + GNNDataset (identical to GNN trainer)
# -------------------------------------------------------------------------
query_datasets = build_datasets(config)

encoder = HintEncoder(
    op_mapping="./flexdata_metric_prediction/encoder/opMappingNew.json",
    rel_mapping="./flexdata_metric_prediction/encoder/relMapping.json",
    type_mapping="./flexdata_metric_prediction/encoder/typeMapping.json",
)

gnn_dataset = create_gnn_dataset(query_datasets, encoder, config)

engines: List[str] = gnn_dataset.engines                # sorted, fixed
label_mapping = gnn_dataset.get_label_mapping()         # {(engine, csv_col): idx}
first_schema = next(iter(data_config.values()))
first_engine = next(iter(first_schema["metrics"].keys()))
semantic_names: List[str] = list(first_schema["metrics"][first_engine].keys())
sem_to_csv: Dict[str, Dict[str, str]] = {e: dict(first_schema["metrics"][e]) for e in engines}
num_metrics_per_engine = len(semantic_names)
total_label_cols = len(engines) * num_metrics_per_engine

# -------------------------------------------------------------------------
# 3. Splits (identical to GNN trainer; seed above ensures reproducibility)
# -------------------------------------------------------------------------
val_split  = {n: data_config[n]["num_val"]  for n in data_config}
test_split = {n: data_config[n]["num_test"] for n in data_config}
train_data, val_data, test_data = gnn_dataset.get_splits(val_split, test_split)


# -------------------------------------------------------------------------
# 4. MetricNormalizer: fit on train, apply in-place to all splits.
# -------------------------------------------------------------------------
# Grab raw labels BEFORE applying normalization -- we need them later for
# raw-space Q-error. apply_normalization_to_data only mutates data.y; it
# does not reorder, so these arrays stay aligned with the split graphs.
train_labels_raw = extract_labels_from_data(train_data)
val_labels_raw   = extract_labels_from_data(val_data)
test_labels_raw  = extract_labels_from_data(test_data)

normalizer = MetricNormalizer(impute_percentile=5.0)
normalizer.fit(train_labels_raw, engines, semantic_names)
logging.info(normalizer.summary())

train_data = apply_normalization_to_data(train_data, normalizer)
val_data   = apply_normalization_to_data(val_data,   normalizer)
test_data  = apply_normalization_to_data(test_data,  normalizer)

# -------------------------------------------------------------------------
# 5. Featurize heterographs -> fixed-size vectors
# -------------------------------------------------------------------------
dims = discover_encoding_dims(train_data + val_data + test_data)

X_train = featurize_all(train_data, dims)
X_val   = featurize_all(val_data,   dims)
X_test  = featurize_all(test_data,  dims)

flat_order = [
    label_mapping[(engine, sem_to_csv[engine][sem])]
    for engine in engines for sem in semantic_names
]
target_tags = [f"{engine}__{sem}" for engine in engines for sem in semantic_names]

def stack_labels(data_list, raw_labels):
    Y_norm = np.stack([extract_label_column(data_list, i) for i in flat_order], axis=1)
    Y_raw  = raw_labels[:, flat_order]
    return Y_norm, Y_raw

Y_tr_n_full, Y_tr_r_full = stack_labels(train_data, train_labels_raw)
Y_vl_n_full, Y_vl_r_full = stack_labels(val_data,   val_labels_raw)
Y_te_n_full, Y_te_r_full = stack_labels(test_data,  test_labels_raw)

def _clean_multi(X, Yn, Yr):
    mask = ~(np.isnan(Yn).any(axis=1) | np.isnan(Yr).any(axis=1))
    return X[mask], Yn[mask], Yr[mask]

X_tr, Y_tr_n, Y_tr_r = _clean_multi(X_train, Y_tr_n_full, Y_tr_r_full)
X_vl, Y_vl_n, Y_vl_r = _clean_multi(X_val,   Y_vl_n_full, Y_vl_r_full)
X_te, Y_te_n, Y_te_r = _clean_multi(X_test,  Y_te_n_full, Y_te_r_full)

# Per-target raw range for prediction clipping (same for all combos)
raw_min = np.nanmin(Y_tr_r, axis=0)
raw_max = np.nanmax(Y_tr_r, axis=0)

# =============================================================================
# 6. Train and evaluate
# =============================================================================
out = Path(config["out"] + config["name"])
out.mkdir(parents=True, exist_ok=True)

GRID_SEARCH_PARAMS: dict = {}

BASE_PARAMS: dict = model_config["xgb_params"]

param_keys = list(GRID_SEARCH_PARAMS.keys())
all_rows: list[dict] = []

combos = list(itertools.product(*GRID_SEARCH_PARAMS.values()))
print(f"Running {len(combos)} combos")

for combo_id, values in enumerate(tqdm(combos)):
    overrides = dict(zip(param_keys, values))

    model = xgb.XGBRegressor(
        **BASE_PARAMS,
        **overrides,
        multi_strategy="multi_output_tree",
        early_stopping_rounds=model_config["early_stopping_rounds"],
    )

    model.fit(X_tr, Y_tr_n, eval_set=[(X_vl, Y_vl_n)], verbose=False)

    Y_pred_n = model.predict(X_te)
    Y_pred_r = np.zeros_like(Y_pred_n)
    for j, flat_idx in enumerate(flat_order):
        inv = inverse_transform_column(
            normalizer, Y_pred_n[:, j], flat_idx, total_label_cols,
        )
        Y_pred_r[:, j] = np.clip(inv, raw_min[j], raw_max[j])

    target_rows = []
    eps = 1e-6
    for j, tag in enumerate(target_tags):
        yt = np.maximum(Y_te_r[:, j], eps)
        yp = np.maximum(Y_pred_r[:, j], eps)
        q = np.maximum(yt / yp, yp / yt)
        target_rows.append({
            "target":    tag,
            "engine":    tag.split("__")[0],
            "metric":    tag.split("__")[1],
            "q_median":  float(np.median(q)),
            "q_mean":    float(np.mean(q)),
            "q_90":      float(np.percentile(q, 90)),
            "q_95":      float(np.percentile(q, 95)),
            "q_99":      float(np.percentile(q, 99)),
            "mae_norm":  float(mean_absolute_error(Y_te_n[:, j], Y_pred_n[:, j])),
            "rmse_norm": float(np.sqrt(mean_squared_error(Y_te_n[:, j], Y_pred_n[:, j]))),
            "r2_norm":   float(r2_score(Y_te_n[:, j], Y_pred_n[:, j])),
        })
    best_iter = int(getattr(model, "best_iteration", -1) or -1)

    for row in target_rows:
        all_rows.append({
            "combo_id":  combo_id,
            **overrides,
            "best_iter": best_iter,
            **row,
        })

    # Persist incrementally
    pd.DataFrame(all_rows).to_csv(out / "sweep_results.csv", index=False)

# =============================================================================
# 7. Save results
# =============================================================================

df = pd.DataFrame(all_rows)
df.to_csv(out / "sweep_results.csv", index=False)

# Mean across 8 targets per combo — for ranking
summary_keys = param_keys
summary = (
    df.groupby(["combo_id"] + summary_keys, dropna=False)[["q_median", "q_95"]]
      .mean()
      .reset_index()
      .sort_values("q_median")
)
summary.to_csv(out / "sweep_summary_mean_over_targets.csv", index=False)