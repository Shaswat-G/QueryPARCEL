# Appendix A8 — XGBoost Baseline: Featurization and Training

The XGBoost baseline is a deliberately simpler alternative to the GNN: gradient-boosted
trees over a flattened, topology-free plan summary. It consumes the **same** decorated
Substrait plans, per-node encodings, supervision targets, and preprocessing pipeline as
the GNN, and reuses the **same train/validation/test splits** (identical global seed),
so the comparison isolates the effect of structural modeling and model class while
holding all data-side factors fixed.

## Featurization

Each heterogeneous plan graph is collapsed to a fixed-length vector. For every node type
$t \in \{\text{Relation}, \text{Operation}, \text{Literal}, \text{Field}, \text{Table}\}$
we emit:

- node count $n_t$;
- the **sum-pooled** node encoding $\sum_i x_i^{(t)}$ ($d_t$ dims), carrying the
  cardinality- and size-hint features produced by the encoder;
- the per-type depth **mean** and **max**.

Two global scalars are appended once: total node count and global **max** plan depth.
Missing node types contribute zero-padded slots so the vector length is invariant.

| Component | Per node type ($\times 5$) | Global ($\times 1$) |
|---|---|---|
| Count | $n_t$ (1) | total nodes (1) |
| Encoding | sum-pool $\sum_i x_i^{(t)}$ ($d_t$) | — |
| Depth | mean, max (2) | max depth (1) |

Total dimensionality: $\;5\,(1 + d_t + 2) + 2 = 15 + \sum_t d_t + 2 = \mathbf{71}$
(with $\sum_t d_t = 54$ under the encoder used here).

**Excluded by construction.** No edges or connectivity, no message passing, no
individual-node depth, and no mean/max pooling of encodings (sum-pooling only). Depth
enters solely as per-type and global aggregates. The representation thus preserves
operator composition and cardinality statistics while discarding plan topology — the
single axis along which it differs from the GNN's input.

## Supervision targets

A single multi-output regressor jointly predicts all **eight** outputs (four engine
configurations $\times$ {latency, peak memory}). Targets are the engine-native signals
used by the cost adapters (§4.5), ensuring label consistency between the two models:

| Engine | Latency target | Memory target |
|---|---|---|
| `presto-w1`, `presto-w4` | `elapsedTime` | `peakNodeTotalMemory` |
| `spark-w1`, `spark-w4` | `wall_clock_duration` | `on_heap_execution_memory` |

## Target normalization

Targets are normalized in the log domain (`MetricNormalizer`, fit on the **training
split only**, with 5th-percentile imputation of degenerate values) and inverted at
prediction time. Inverted predictions are then clipped per target to the training-set
observed range $[\min, \max]$; this prevents exponential blow-ups from moderately
negative normalized predictions on high-variance targets, which would otherwise inflate
mean Q-error.

## Model and hyperparameters

A single `XGBRegressor` is trained on the 71-dimensional vectors with early stopping on
the validation split. Rows with any NaN across the eight targets are dropped jointly to
keep targets aligned.

| Hyperparameter | Value |
|---|---|
| `n_estimators` | 1000 |
| `max_depth` | 6 |
| `learning_rate` | 0.1 |
| `reg_lambda` | 1.0 |
| `reg_alpha` | 0 |
| `objective` | `reg:squarederror` |
| `eval_metric` | `rmse` |
| `tree_method` | `hist` |
| `early_stopping_rounds` | 50 |
| `random_state` / global seed | 123 |

Remaining parameters retain library defaults (`subsample` $=1.0$,
`colsample_bytree` $=1.0$, `min_child_weight` $=1$, `gamma` $=0$). The 1000-tree
ensemble is sized generously so that any baseline shortfall cannot be attributed to
under-capacity.

## Evaluation

We report metrics in two spaces. In **normalized** space (comparable to the GNN
training loss): MAE, RMSE, and $R^2$. In **raw** space (the standard cost-model metric):
Q-error $= \max(y/\hat{y},\, \hat{y}/y)$, summarized by median, mean, p90, p99, and max.

## Reproducibility

The global seed (123) reproduces the GNN trainer's splits and normalization, so the two
models are trained and evaluated on identical query partitions. Code path:
`flexdata-metric-prediction` (XGBoost trainer), config `with_depth_features_SE`.