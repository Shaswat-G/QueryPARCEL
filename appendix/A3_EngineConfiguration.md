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

Dynamic executor allocation is disabled (`spark.dynamicAllocation.enabled = false`) to ensure fixed resource footprints and precise cost attribution.

#### Parallelism Configuration

| Parameter | Spark-w1 | Spark-w4 | Description |
|-----------|----------|----------|-------------|
| `spark.sql.shuffle.partitions` | 32 | 128 | Number of partitions for shuffle operations |
| `spark.default.parallelism` | 32 | 128 | Default parallelism for RDDs |

Partition counts scale with worker count (2 partitions per task slot on average) to maintain balanced distribution.

#### Memory Management

| Parameter | Value | Description |
|-----------|-------|-------------|
| `spark.memory.fraction` | 0.6 | Fraction of heap reserved for execution + storage memory pool |
| `spark.memory.storageFraction` | 0.0 | Fraction of execution memory reserved for caching (disabled) |

Storage memory is disabled (`storageFraction = 0.0`) so the entire unified memory pool serves execution operators, ensuring measured peak memory reflects true demand rather than caching artifacts.

#### Query Optimization Disablement

| Parameter | Value | Description |
|-----------|-------|-------------|
| `spark.sql.adaptive.enabled` | false | Disables Adaptive Query Execution (AQE) |
| `spark.sql.autoBroadcastJoinThreshold` | -1 | Disables automatic broadcast joins |

AQE is disabled to ensure deterministic, reproducible physical plans tied to logical-plan cardinality estimates. Broadcast joins are disabled (`threshold = -1`) to force sort-merge/shuffle-hash strategies, making memory consumption plan-dependent rather than data-dependent.

#### Telemetry and Monitoring

| Parameter | Value | Description |
|-----------|-------|-------------|
| `spark.eventLog.enabled` | true | Enables structured event logging |
| `spark.eventLog.dir` | `/mnts/spark-data-collection/logs/events` | Event log directory |
| `spark.executor.metrics.pollingInterval` | 20 | Executor metrics polling interval (ms) |
| `spark.executor.heartbeatInterval` | 250ms | Executor→driver heartbeat interval |
| `spark.eventLog.logStageExecutorMetrics` | true | Logs per-stage executor metrics |
| `spark.executor.processTreeMetrics.enabled` | true | Enables OS-level memory tracking |

High-frequency polling (20 ms vs. default 10 s) captures peak memory during short-lived stages. The 250 ms heartbeat interval reduces driver-side reporting lag. Process tree metrics provide OS-level visibility (RSS), though the paper relies on Spark's internal `peakExecutionMemory`.

---

### Presto Configuration Details

Each Presto worker runs a **single JVM process**; `task.concurrency` controls intra-node parallelism (unlike Spark's executor-based model).

#### Core Resource Allocation

| Parameter | Presto-w1 | Presto-w4 | Description |
|-----------|-----------|-----------|-------------|
| Workers | 1 | 4 | Fixed worker count (coordinator excluded from data processing via `node-scheduler.include-coordinator=false`) |
| `-Xmx` (JVM heap) | 10 GiB | 60 GiB | Hard cap on Java heap per worker node |
| `task.concurrency` | 16 | 16 | Concurrent task drivers per worker → 16 task slots |
| `query.max-concurrent-queries` | 1 | 1 | Sequential execution per engine; eliminates cross-query memory contention |

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

†For Presto-w1 (single worker) the per-node user cap (7.5 GiB) always binds before the 30 GiB cluster-wide limit.

#### Telemetry

Presto exposes runtime statistics via:
- **Query completion REST API:** Final query stats including `spilledDataSize`, `elapsedTime`, and `peakMemoryBytes` per query.
- **Presto memory manager internals:** Stage-level and operator-level memory tracked via Presto's internal memory contexts, recoverable from the REST API and query stats.

The ETL pipeline (`etl_presto.py`) parses these REST API responses to extract per-query training labels.

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

Excluded alternatives: total heap and RSS conflate GC history, JVM overhead, and multi-tenant effects.

#### Peak Memory Reconstruction

Event logs are line-delimited JSON. Per-query $m^\star$ is reconstructed in four stages:

1. **Temporal filtering.** Restrict executor metric samples to $[t_0(q),\, t_1(q)]$ bounded by `SparkListenerSQLExecutionStart` / `SparkListenerSQLExecutionEnd` with matching `executionId`. Execution time $t(q,e) = t_1 - t_0$.
2. **Executor filtering.** Retain only executors that participated in at least one task belonging to query $q$ (via job→stage→task linkage in the event log).
3. **Peak extraction.**
$$m^\star(q,e) = \max_{x \in \mathcal{X}(q)} \bigl[\texttt{OnHeapExecutionMemory}(x)\bigr]$$
where $\mathcal{X}(q)$ is the set of executor metric update events within $[t_0, t_1]$ for participating executors.
4. **Spill verification.** Confirmed `memoryBytesSpilled = 0` and `shuffleWriteMetrics.bytesSpilled = 0` across all tasks of $q$. Runs with nonzero spill are excluded.

#### Execution Protocol

Sequential, single-tenant execution. JVM warmup precedes measurement (excluded via metadata filtering). Each query executes 3 times; downstream analysis uses the median. Timeout: 125 s.

---

### Presto Instrumentation

#### Memory Signal: `peakNodeTotalMemory`

Presto's `MemoryContext` hierarchy attributes all operator allocations to queries at reservation time, eliminating the reconstruction problem present in Spark. From the `queryStats` REST response, we use `peakNodeTotalMemory` — the maximum, over all participating workers, of a worker's total query memory reservation (user + system).

| Property | Justification |
|----------|---------------|
| **Query-attributable** | `MemoryContext.setBytes()` tracks reservations at allocation time, not via JVM introspection |
| **GC-independent** | Application-level accounting; unaffected by GC cycles |
| **Provisioning-aligned** | A query fails when any single worker exceeds its admissible memory; pod sizing must satisfy the per-worker bottleneck, not the cluster aggregate |

Excluded alternatives: `peakUserMemoryReservation` excludes system memory; `peakTotalMemoryReservation` is a cluster aggregate that does not identify the bottleneck node.

$$m^\star_{\mathrm{Presto}}(q,e) := \texttt{peakNodeTotalMemory}(q,e)$$

Per-query statistics are retrieved via `GET /v1/query/{queryId}`. Key fields:

| Field | Role |
|-------|------|
| `executionTime` | Wall-clock duration excluding queuing time |
| `peakNodeTotalMemory` | Adapter input $m^\star$ |
| `spilledDataSize` | Spill verification (must be 0) |
| `state` | Outcome filter (`FINISHED` only) |

#### Execution Protocol

Sequential execution, 20-query batches with 30 s cooldown. Timeout: `query_max_execution_time = '125s'`. Each query executes 3 times; downstream analysis uses the median.