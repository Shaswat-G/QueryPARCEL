# QueryPARCEL — Online Appendix

Supplementary material for the VLDB paper. Each appendix section is a self-contained document linked below.

---

## Table of Contents

### [Appendix A1 — Motivating Example: Queries & Measurements](A1_Motivation_Queries.md)

Full SQL text and per-engine runtime/cost measurements for the two TPC-H queries used in Section 1 (Figure 1), illustrating the Pareto trade-off and "silver bullet" scenarios that motivate provisioning-aware routing.

- [Query 1 — Pareto frontier (no single engine dominates)](A1_Motivation_Queries.md#query-1)
- [Query 2 — Silver-bullet engine (simultaneously fastest and cheapest)](A1_Motivation_Queries.md#query-2)

---

### [Appendix A2 — Model Architecture & Training Configuration](A2_ModelArch_Training.md)

Complete specification of the BottomUpGNN cost model (~500k parameters), including graph construction from Substrait plans, feature encoding, message-passing mechanics, and all training hyperparameters.

- [A.1 Graph Representation](A2_ModelArch_Training.md#a1--graph-representation) — Node types (Rel, Table, Field, OP, Literal) and 8 directed edge types
- [A.2 Feature Encoding](A2_ModelArch_Training.md#a2--feature-encoding-hintencoder) — Per-node-type feature vectors from optimizer hints
- [A.3 Model Architecture](A2_ModelArch_Training.md#a3--model-architecture) — Three-stage pipeline: embedding, bottom-up message passing, per-engine heads
- [A.4 Label Normalization](A2_ModelArch_Training.md#a4--label-normalization) — Log-transform + Z-score pipeline for time and memory targets
- [A.5 Training Configuration](A2_ModelArch_Training.md#a5--training-configuration) — AdamW optimizer, Huber loss, LR schedule, early stopping
- [A.6 Evaluation Metrics](A2_ModelArch_Training.md#a6--evaluation-metrics) — Q-error definition and downstream routing tasks
- [A.7 Parameter Count](A2_ModelArch_Training.md#a7--parameter-count-summary) — Per-component breakdown totalling ~500k parameters

---

### [Appendix A3 — Engine Configurations & Instrumentation](A3_EngineConfiguration.md)

Detailed Spark and Presto configurations (w1/w4 workers) with JVM heap sizing, parallelism settings, and telemetry — designed for spill-free, reproducible execution across all 6,348 queries.

- [Overview Table](A3_EngineConfiguration.md#overview-table) — Side-by-side comparison of all four engine configurations
- [Spark Configuration](A3_EngineConfiguration.md#spark-configuration-details) — Resource allocation, memory management, AQE disablement, telemetry ([spark-w1.json](spark_config/spark-w1.json), [spark-w4.json](spark_config/spark-w4.json))
- [Presto Configuration](A3_EngineConfiguration.md#presto-configuration-details) — Three-layer memory model, resource groups, session properties, spill verification

---

### [Appendix A4 — Datasets & Workloads](A4_Datasets.md)

Schema statistics, query counts, complexity distributions, and data splits for TPC-H (1,960 queries), TPC-DS (4,388 queries), and IMDB JOB (728 queries).

- [Training Corpora (TPC-H / TPC-DS)](A4_Datasets.md#training-and-evaluation-corpora-tpc-h-and-tpc-ds) — SF10 benchmarks, 6,348 queries total, 80/10/10 stratified splits
- [Query Complexity Distribution](A4_Datasets.md#query-complexity-distribution) — Join depth, group-by width, and operator counts
- [Zero-Shot Corpus (IMDB JOB)](A4_Datasets.md#zero-shot-transfer-corpus-imdb-job) — 728 multi-join queries for few-shot adaptation
- [Storage Format](A4_Datasets.md#storage-format) — Parquet on S3 with Iceberg/Hive catalog

---

### [Appendix A5 — Cluster & Execution Environment](A5_ClusterDetails.md)

Hardware specs, storage infrastructure, and engine deployment for the IBM Research Zurich OpenShift cluster.

- [Hardware](A5_ClusterDetails.md#hardware) — 16-node Dell PowerEdge cluster, 512 cores, 12.3 TiB RAM
- [Storage](A5_ClusterDetails.md#storage) — Ceph S3 object store + GPFS persistent volumes
- [Engine Deployment](A5_ClusterDetails.md#engine-deployment) — Containerized Spark/Presto on dedicated OpenShift pods

---

### [Appendix A6 — Zero-Shot Transfer & Workload Shift (IMDB JOB)](A6_ZeroShot.md)

Quantitative analysis of the distribution shift between TPC training workloads and IMDB JOB, motivating few-shot fine-tuning for out-of-distribution onboarding.

- [Latency Regime Shift](A6_ZeroShot.md#latency-regime-shift-bulk-and-tail) — ECDF comparison showing 10x-23x median latency reduction on JOB
- [Distributional Separation](A6_ZeroShot.md#distributional-separation-on-log-latency) — KS statistics (mean 0.80) and Cohen's *d* (1.2-2.4) on log-latency
- [Implication for Onboarding](A6_ZeroShot.md#implication-for-onboarding) — Why zero-shot transfer is unreliable and few-shot adaptation is necessary

---

### [Appendix A7 — Adapter Sketches for Additional Engines](A7_EngineAdapter_Ext.md)

Conceptual adapter sketches for Trino and DuckDB, demonstrating that the three-stage Extract → Invert → Quantize pipeline extends beyond the Spark/Presto engines evaluated in the paper. Neither adapter has been instantiated or tested on real workloads.

- [Trino](A7_EngineAdapter_Ext.md#trino) — Stage 1 instrumentation extension for per-node peak memory; Stage 2 inversion algebraically identical to PrestoDB
- [DuckDB](A7_EngineAdapter_Ext.md#duckdb) — Embedded single-process engine without JVM or multi-node coordination; serial-execution extraction via `SYSTEM_PEAK_BUFFER_MEMORY`

---

### [Appendix A8 — XGBoost Baseline: Featurization and Training](A8_XGBoost_Baseline.md)

Details on the canonical LCM on flattened plan features, serving as both a literature-established baseline and a model complexity ablation.

- [Featurization](A8_XGBoost_Baseline.md#featurization) — Flattening heterogeneous plan graphs to a fixed-length 71-dimensional vector preserving composition but discarding topology
- [Supervision targets](A8_XGBoost_Baseline.md#supervision-targets) — Joint prediction of latency and peak memory across all configurations
- [Model and hyperparameters](A8_XGBoost_Baseline.md#model-and-hyperparameters) — Configuration, parameters, and early stopping for the `XGBRegressor` ensemble
- [Evaluation](A8_XGBoost_Baseline.md#evaluation) — Metrics and seed-level reproducibility matching the GNN pipeline

---

### [Appendix A9 — Positioning of Prior Systems on Multi-Objective Query Optimization](A9_MultiObjectiveQueryOptimization.md)

Detailed breakdown and component-wise evaluation positioning prior MOQO/MPQ, learned-router, polystore, and cross-platform systems against our multi-objective provisioning-aware routing.

- [Column definitions](A9_MultiObjectiveQueryOptimization.md#column-definitions) — Setup of the six criteria axes (Scope, Granularity, Decision space, Decision-time info, Objective, Constraints)
- [Component-wise alignment](A9_MultiObjectiveQueryOptimization.md#table-a91--component-wise-alignment) — Summary comparison against related systems (T&K MOQO, RHEEM, BRAD, etc.)
- [Per-cell mismatch reasons](A9_MultiObjectiveQueryOptimization.md#table-a92--per-cell-mismatch-reasons) — Granular rationale for why specific systems solve fundamentally different optimization formulations
- [Synthesis](A9_MultiObjectiveQueryOptimization.md#synthesis) — Categorization of mismatches into plan-space optimization, latency-only routing, and workload-level blueprinting

---

### [Appendix A10 — Cost-Opportunity Decomposition](A10_CostOpportunityDecomposition.md)

Isolates the amount of cost-routing opportunity attributable to per-query memory-tier selection independent of predictor error, demonstrating that the memory term dominates the margin between latency-optimal and cost-optimal routing.

- [Setup](A10_CostOpportunityDecomposition.md#setup) — Definition of cost vs. latency oracles and the capture metric
- [The memory-blind baseline](A10_CostOpportunityDecomposition.md#the-memory-blind-baseline) — Using a constant-threshold policy to factor out parallelism from memory scaling
- [Results & Interpretation](A10_CostOpportunityDecomposition.md#results) — Showing that structural memory predictions unlock 69.1% of the opportunity, whereas parallelism-only or tabular models fail to clear the baseline floor

---

