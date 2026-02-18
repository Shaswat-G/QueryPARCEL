# QueryPARCEL: Online Appendix

This folder contains supplementary material referenced as “Appendix A#” in the paper.

## Table of Contents
- **Appendix A1 — Queries in the Motivation**  
  `A1_motivation_queries/`  
  Contents: SQL, (optional) explain plans, and context for where each appears in the paper.

- **Appendix A2 — Model Architecture**  
  `A2_model_architecture/`  
  Contents: layer-by-layer dims table + diagram + script to export from code.

- **Appendix A3 — Engine Configurations**  
  `A3_engine_configs/`  
  Contents: Spark/Presto configs, JVM heap settings, and summary table.

- **Appendix A4 — Engine Instrumentation Details**  
  *(inline, this document)*  
  Contents: Spark/Presto metric extraction protocols, signal selection rationale, query identity propagation, and peak memory reconstruction.

- **Appendix A5 — Datasets and Storage Layout**  
  `A5_datasets/`  
  Contents: dataset manifests, schema, stats, and download links/checksums.

- **Appendix A6 — Cluster and Resource Details**  
  `A6_cluster_details/`  
  Contents: node/pod resources, scheduling, and spill-free verification notes.

- **Appendix A7 — Zero-shot Transfer and Workload Shift (IMDB JOB)**  
  `A7_zero_shot_and_shift/`  
  Contents: zero-shot results, distributions, and few-shot sweep tables/figures.

---

## Appendix A3 — Engine Configurations

This section describes the detailed engine configurations used for the experiments in the paper. We evaluated four configurations by varying the engine (Apache SparkSQL 3.4, PrestoDB 0.28x) and parallelism level (w1, w4 workers): **Presto-w1**, **Presto-w4**, **Spark-w1**, and **Spark-w4**. Each configuration provides **16 task slots per worker** to ensure comparable concurrency baselines.

All configurations were designed to guarantee **spill-free execution** across the entire corpus of 6,348 queries, allowing us to measure true memory demand without the confounding effects of spill-to-disk mechanisms.

### Overview Table

| Configuration | Engine | Workers | Cores/Worker | JVM Heap/Worker | Aggregate Cores | Aggregate JVM Heap | Task Slots/Worker | Effective Exec Memory/Worker† |
|---------------|--------|---------|--------------|-----------------|-----------------|---------------------|-------------------|-------------------------------|
| **Presto-w1** | PrestoDB | 1 | 16 | 10 GiB | 16 | 10 GiB | 16 | 7.5 GiB (user) |
| **Presto-w4** | PrestoDB | 4 | 16 | 60 GiB | 64 | 240 GiB | 16 | 40 GiB (user) |
| **Spark-w1** | SparkSQL | 1 | 16 | 60 GiB | 16 | 60 GiB | 16 | 36 GiB (exec pool) |
| **Spark-w4** | SparkSQL | 4 | 16 | 10 GiB | 64 | 40 GiB | 16 | 6 GiB (exec pool) |

†Presto: `query.max-memory-per-node` (tracked user memory cap per query per node). Spark: `spark.executor.memory × spark.memory.fraction × (1 − spark.memory.storageFraction)` = execution pool available for operators.

**Note on memory provisioning:** The per-worker JVM heap limits were deliberately overprovisioned to guarantee spill-free execution (paper §5.1 reports: Presto-w1: 10 GiB, Presto-w4: 60 GiB, Spark-w1: 60 GiB, Spark-w4: 10 GiB). Post-hoc analysis confirmed zero spill across all 6,348 queries via engine-native telemetry (`memoryBytesSpilled` in Spark event logs and `spilledDataSize` in Presto query summaries).


---

### Spark Configuration Details

The complete Spark configurations for w1 and w4 are provided in [`spark_config/spark-w1.json`](spark_config/spark-w1.json) and [`spark_config/spark-w4.json`](spark_config/spark-w4.json). Below we document the key configuration parameters that were critical for ensuring reproducible, spill-free measurements suitable for training the learned cost model.

#### Core Resource Allocation

| Parameter | Spark-w1 | Spark-w4 | Description |
|-----------|----------|----------|-------------|
| `ae.spark.executor.count` | 1 | 4 | Fixed number of executors (IBM AE-specific property; dynamic allocation disabled) |
| `spark.executor.cores` | 16 | 16 | CPU cores per executor → 16 task slots per executor |
| `spark.executor.memory` | 60G | 10G | JVM heap per executor (IBM AE adds pod overhead automatically) |
| `spark.driver.cores` | 2 | 2 | Driver CPU cores |
| `spark.driver.memory` | 8G | 8G | Driver JVM heap |
| `spark.task.cpus` | 1 | 1 | CPUs per task (16 cores ÷ 1 = 16 concurrent task slots per executor) |

**Rationale:** We disabled dynamic executor allocation (`spark.dynamicAllocation.enabled = false`) to ensure fixed resource footprints, eliminating runtime variability and enabling precise cost attribution per query-engine pair.

#### Parallelism Configuration

| Parameter | Spark-w1 | Spark-w4 | Description |
|-----------|----------|----------|-------------|
| `spark.sql.shuffle.partitions` | 32 | 128 | Number of partitions for shuffle operations |
| `spark.default.parallelism` | 32 | 128 | Default parallelism for RDDs |

**Rationale:** Partition counts scale with worker count to maintain balanced task distribution. With 16 task slots per worker, 32 partitions (w1) and 128 partitions (w4) ensure each task slot processes 2 partitions on average, preventing skew and underutilization.

#### Memory Management

| Parameter | Value | Description |
|-----------|-------|-------------|
| `spark.memory.fraction` | 0.6 | Fraction of heap reserved for execution + storage memory pool |
| `spark.memory.storageFraction` | 0.0 | Fraction of execution memory reserved for caching (disabled) |

**Rationale:**  
- **Memory Fraction (0.6):** Allocates 60% of executor heap to Spark's unified memory manager, with the remaining 40% reserved for user data structures, internal metadata, and JVM overhead. The default value (0.6) strikes a balance between execution memory availability and GC overhead.
- **Storage Fraction (0.0):** We completely disable storage memory (caching) to dedicate the entire unified memory pool to execution (shuffles, aggregations, joins). This configuration ensures that the measured peak memory reflects true execution demand rather than opportunistic caching behavior, which is critical for the spill-free memory threshold abstraction introduced in the paper.

#### Query Optimization Disablement

| Parameter | Value | Description |
|-----------|-------|-------------|
| `spark.sql.adaptive.enabled` | false | Disables Adaptive Query Execution (AQE) |
| `spark.sql.autoBroadcastJoinThreshold` | -1 | Disables automatic broadcast joins |

**Rationale:**  
- **Disabling AQE:** Adaptive Query Execution dynamically re-optimizes physical plans during execution based on runtime statistics (e.g., coalescing shuffle partitions, converting sort-merge joins to broadcast joins). While beneficial in production, AQE introduces non-determinism that complicates cost model training: the same logical plan may execute differently across runs. Disabling AQE ensures that the optimizer's decisions are stable and reproducible, tying physical plan structure deterministically to the logical plan's cardinality estimates.
- **Disabling Broadcast Joins:** Automatic broadcast join conversion depends on runtime table size estimates, which can vary with data skew and caching state. By setting the threshold to `-1`, we force all joins to use sort-merge or shuffle-hash strategies, making memory consumption predictable and plan-dependent rather than data-dependent. This is essential for learning a cost model that generalizes across queries rather than overfitting to specific data distributions.

#### Telemetry and Monitoring

| Parameter | Value | Description |
|-----------|-------|-------------|
| `spark.eventLog.enabled` | true | Enables structured event logging |
| `spark.eventLog.dir` | `/mnts/spark-data-collection/logs/events` | Event log directory |
| `spark.executor.metrics.pollingInterval` | 20 | Executor metrics polling interval (ms) |
| `spark.executor.heartbeatInterval` | 250ms | Executor→driver heartbeat interval |
| `spark.eventLog.logStageExecutorMetrics` | true | Logs per-stage executor metrics |
| `spark.executor.processTreeMetrics.enabled` | true | Enables OS-level memory tracking |

**Rationale:**  
- **Metrics Polling (20ms):** High-frequency polling ensures that peak memory usage during short-lived stages is accurately captured. The default interval (10s) is too coarse for fine-grained memory analysis.
- **Heartbeat Interval (250ms):** Increased heartbeat frequency (default: 10s) reduces the lag between executor state changes and driver awareness, critical for timely telemetry in jobs with many short stages.
- **Process Tree Metrics:** Enables tracking of RSS (Resident Set Size) and other OS-level memory metrics, though the paper ultimately relies on Spark's internal `peakExecutionMemory` metric derived from the unified memory manager for spill-free threshold computation.

These telemetry settings enabled the ETL pipeline (described in Section 4.3 of the paper) to extract high-fidelity training labels for the multi-task learned cost model.

---

### Presto Configuration Details

PrestoDB configurations for w1 and w4 use Presto-specific resource management properties. Unlike Spark's executor-based model, each Presto worker runs a **single JVM process** handling all task threads for that node.

#### Core Resource Allocation

| Parameter | Presto-w1 | Presto-w4 | Description |
|-----------|-----------|-----------|-------------|
| Workers | 1 | 4 | Fixed worker count (coordinator excluded from data processing via `node-scheduler.include-coordinator=false`) |
| `-Xmx` (JVM heap) | 10 GiB | 60 GiB | Hard cap on Java heap per worker node |
| `task.concurrency` | 16 | 16 | Concurrent task drivers per worker → 16 task slots |
| `query.max-concurrent-queries` | 1 | 1 | Sequential execution per engine; eliminates cross-query memory contention |

**Note:** Presto does not use the same executor-based model as Spark. Each Presto worker runs a single JVM process; `task.concurrency` controls intra-node parallelism.

#### Memory Management

Presto uses a three-layer memory model inside the JVM heap. Configurations governing each layer are shown below.

| Parameter | Presto-w1 | Presto-w4 | Description |
|-----------|-----------|-----------|-------------|
| `-Xmx` | 10 GiB | 60 GiB | Total JVM heap |
| `memory.heap-headroom-per-node` | ~2 GiB | ~12 GiB | Reserved for untracked allocations (3rd-party libs, GC overhead). Presto never allocates tracked query memory here. |
| `query.max-memory-per-node` | 7.5 GiB | 40 GiB | Max **user** memory (hash tables, join buffers, aggregation state) a single query may reserve on this node |
| `query.max-total-memory-per-node` | 7.5 GiB | 45 GiB | Max **user + system** memory (includes exchange buffers, internal state) per query per node |
| `query.max-memory` | 30 GiB† | 40 GiB | Cluster-wide user memory cap for a single query across all workers |
| `experimental.reserved-pool-enabled` | false | false | Disables the reserved memory pool; all non-headroom heap forms one general pool, maximising per-query headroom |
| `query.low-memory-killer.policy` | `total-reservation-on-blocked-nodes` | same | When a node's pool is exhausted and tasks block, the OOM killer terminates the query with the largest reservation on blocked nodes |
| `query.execution-policy` | `phased` | `phased` | Only one pipeline phase runs at a time, reducing peak concurrent memory footprint |

†For Presto-w1 (single worker) the per-node user cap (7.5 GiB) always binds before the 30 GiB cluster-wide limit, which is therefore unreachable in practice.

**Memory budget sanity check:**
- **Presto-w1:** `query.max-total-memory-per-node (7.5 GiB) + heap-headroom (~2 GiB) = 9.5 GiB < 10 GiB (-Xmx)` → ~0.5 GiB JVM slack.
- **Presto-w4:** `query.max-total-memory-per-node (45 GiB) + heap-headroom (~12 GiB) = 57 GiB < 60 GiB (-Xmx)` → ~3 GiB JVM slack.

This margin prevents `OutOfMemoryError` even when queries approach their tracked limits.

#### Telemetry

Presto exposes runtime statistics via:
- **Query completion REST API:** Final query stats including `spilledDataSize`, `elapsedTime`, and `peakMemoryBytes` per query.
- **Presto memory manager internals:** Stage-level and operator-level memory tracked via Presto's internal memory contexts, recoverable from the REST API and query stats.

The ETL pipeline (`etl_presto.py`) parses these REST API responses to extract per-query training labels.

---

### Spill-Free Verification

As stated in Section 5.1 of the paper, post-hoc analysis confirmed **zero spill across all 6,348 queries** via engine-native telemetry:
- **Spark:** `memoryBytesSpilled = 0` for all stages in event logs
- **Presto:** `spilledDataSize = 0` in query summaries

This verification is critical because the spill-free memory threshold abstraction (Section 3.1) requires observing peak memory under in-memory execution. If queries had spilled, the observed memory would underestimate true demand, biasing the learned cost model toward under-provisioning.

---

### Reference Files

- **Spark w1 configuration:** [`spark_config/spark-w1.json`](spark_config/spark-w1.json)
- **Spark w4 configuration:** [`spark_config/spark-w4.json`](spark_config/spark-w4.json)

These JSON files contain the complete configuration dictionaries used to submit Spark applications, including environment variables, volume mounts, and application paths.

---

## Appendix A4 — Engine Instrumentation Details

This section documents the per-query metric extraction protocols for execution time $t(q,e)$ and spill-free peak memory $m^\star(q,e)$. Engine configuration (heap sizes, parallelism, optimization flags) is covered in A3; this section concerns only signal selection, query identity propagation, and post-hoc reconstruction.

---

### Spark Instrumentation

#### Memory Signal: `OnHeapExecutionMemory`

Among executor metrics emitted via `SparkListenerExecutorMetricsUpdate`, we extract `OnHeapExecutionMemory` — the bytes tracked by `MemoryManager.acquireExecutionMemory()`. This signal satisfies three requirements:

| Property | Justification |
|----------|---------------|
| **Spill-aligned** | Insufficiency in execution memory directly triggers disk spill or OOM; the signal is thus semantically tied to the provisioning decision |
| **GC-independent** | Tracked at allocation time via application-level accounting, not via post-GC heap snapshots |
| **Storage-isolated** | Excludes cached partitions and broadcast variables (`spark.memory.storageFraction = 0.0`), isolating operator demand from caching artifacts |

Excluded alternatives: total heap and RSS conflate GC history, JVM overhead, and multi-tenant effects; `storage memory` reflects caching policy rather than query demand.

#### Telemetry Configuration

| Parameter | Value | Effect |
|-----------|-------|--------|
| `spark.eventLog.enabled` | `true` | Structured JSON event log per application |
| `spark.eventLog.logStageExecutorMetrics` | `true` | Associates executor memory snapshots with stage boundaries |
| `spark.executor.processTreeMetrics.enabled` | `true` | Process-level memory visibility (RSS, JVM heap) |
| `spark.executor.metrics.pollingInterval` | `20ms` | Intra-executor sampling at sub-heartbeat resolution |
| `spark.executor.heartbeatInterval` | `250ms` | Bounds latency between executor-side sampling and driver-side visibility |

The 20 ms polling interval captures intra-stage peaks that would be missed at Spark's default 10 s interval. Peak memory spikes shorter than the heartbeat interval are recorded locally but reported with up to 250 ms delay.

#### Query Identity Propagation

Spark job IDs are assigned dynamically; a single SQL query may spawn multiple jobs (e.g., unions, subqueries). Query-to-metric association is achieved by injecting metadata before each execution:

```python
sc.setJobGroup(f"q{q}_run{r}", f"Query {q}, Run {r}")
sc.setLocalProperty("run.type", "measurement")  # or "warmup"
sc.setLocalProperty("query.id",  str(q))
sc.setLocalProperty("run.id",    str(r))
```

These thread-local properties propagate into `SparkListenerJobStart` events in the event log, enabling deterministic query-to-metric association and exclusion of warmup runs during ETL.

#### Peak Memory Reconstruction

Event logs are line-delimited JSON. Per-query $m^\star$ is reconstructed in four stages:

1. **Temporal filtering.** Restrict executor metric samples to $[t_0(q),\, t_1(q)]$ bounded by `SparkListenerSQLExecutionStart` / `SparkListenerSQLExecutionEnd` with matching `executionId`. Execution time $t(q,e) = t_1 - t_0$.
2. **Executor filtering.** Retain only executors that participated in at least one task belonging to query $q$ (via job→stage→task linkage in the event log).
3. **Peak extraction.**
$$m^\star(q,e) = \max_{x \in \mathcal{X}(q)} \bigl[\texttt{OnHeapExecutionMemory}(x)\bigr]$$
where $\mathcal{X}(q)$ is the set of executor metric update events within $[t_0, t_1]$ for participating executors.
4. **Spill verification.** Confirmed `memoryBytesSpilled = 0` and `shuffleWriteMetrics.bytesSpilled = 0` across all tasks of $q$. Runs with nonzero spill are excluded.

#### Execution Protocol

Sequential execution with no concurrent workload. JVM warmup precedes measurement: structurally representative queries execute until execution time stabilizes (CV < 5%), then excluded via `run.type = warmup` filtering. Each query executes 3 times; downstream analysis uses the median. Timeout enforced at 125 s via `ThreadPoolExecutor`-based monitoring with `cancelJobGroup()` + `cancelAllJobs()` on expiry.

---

### Presto Instrumentation

#### Memory Signal: `peakNodeTotalMemory`

Presto's `MemoryContext` hierarchy attributes all operator allocations to queries at reservation time, eliminating the reconstruction problem present in Spark. From the `queryStats` REST response, we use `peakNodeTotalMemory` — the maximum, over all participating workers, of a worker's total query memory reservation (user + system).

| Property | Justification |
|----------|---------------|
| **Query-attributable** | `MemoryContext.setBytes()` tracks reservations at allocation time, not via JVM introspection |
| **GC-independent** | Application-level accounting; unaffected by GC cycles |
| **Provisioning-aligned** | A query fails when any single worker exceeds its admissible memory; pod sizing must satisfy the per-worker bottleneck, not the cluster aggregate |

Excluded alternatives: `peakUserMemoryReservation` underestimates by excluding system memory (exchange buffers); `peakTotalMemoryReservation` is a cluster-wide aggregate and does not identify the bottleneck node.

$$m^\star_{\mathrm{Presto}}(q,e) := \texttt{peakNodeTotalMemory}(q,e)$$

#### Query Identity Propagation

Queries are submitted via the Presto Python client with an injected SQL comment:

```sql
-- RUN_ID: q{query_id}_run{run_id}
{query_text}
```

This comment persists in `system.runtime.queries`, enabling post-hoc association between executed queries and their `queryId`. The coordinator's globally unique `queryId` is then used as the primary key for REST metric retrieval.

#### Metrics Retrieval

Upon completion (success or failure), full statistics are fetched from:

```
GET /v1/query/{queryId}
```

Key fields extracted from the `queryStats` object:

| Field | Role |
|-------|------|
| `executionTime` | Wall-clock duration excluding queuing time |
| `peakNodeTotalMemory` | Adapter input $m^\star$ |
| `spilledDataSize` | Spill verification (must be 0) |
| `state` | Outcome filter (`FINISHED` only) |

#### Execution Protocol

Sequential execution with 20-query batches and 30 s cooldown between batches to avoid overwhelming the Hive metastore during catalog operations. Timeout enforced via `SET SESSION query_max_execution_time = '125s'`. Permanent errors (`EXCEEDED_TIME_LIMIT`, `EXCEEDED_LOCAL_MEMORY_LIMIT`, `SYNTAX_ERROR`) are logged and skipped for all subsequent runs of the same query. Each query executes 3 times; downstream analysis uses the median.

#### Spill Verification

`spilledDataSize = 0` confirmed for all retained queries via the `queryStats` REST response.