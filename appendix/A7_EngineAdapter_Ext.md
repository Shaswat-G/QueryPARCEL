## Appendix A7 — Adapter Sketches for Additional Engines

The adapter framework applies to any engine where (i) spill decisions are governed by an internal accounting mechanism rather than OS-level OOM, and (ii) peak memory demand is observable or reconstructible per query. Engines lacking query-scoped attribution would require instrumentation extensions or a learned surrogate for Stage 1. We sketch adapters for two additional engines following the three-stage pipeline of §3 (Engine Adapters). Neither has been instantiated or tested on real workloads.

---

### Trino

Trino retains PrestoDB's `MemoryContext` hierarchy and, since release 369, employs a governance model that closely mirrors PrestoDB. It has a unified query memory pool derived from the JVM heap minus a configurable headroom (`memory.heap-headroom-per-node`). The Stage 2 inversion is therefore algebraically identical to the Presto adapter. The key difference lies in Stage 1: Trino's REST API does not expose a time-correct per-node peak memory field equivalent to PrestoDB's `peakNodeTotalMemory`.

#### Stage 1 — Extract

Trino exposes `peakUserMemoryReservation` as a cluster-wide aggregate and per-task `totalMemoryReservation` values, but multiple tasks for the same query execute concurrently on the same node. Summing task peaks overestimates the per-node requirement (peaks are non-contemporaneous); taking the maximum task peak underestimates it (ignoring co-residency). To recover a correct signal, we outline an internal modification to `io.trino.execution.QueryStateMachine` that maintains a rolling sum of concurrent reservations grouped by `NodeId` and injects a `peakNodeTotalMemoryReservation` metric into the final `QueryStats`:

$$
m^{\star}_{\text{Trino}}(q,e) := \texttt{peakNodeTotalMemoryReservation}(q,e)
$$

#### Stage 2 — Invert

Identical algebraic form to the Presto adapter equation, with Trino's headroom parameter:

$$
\hat{p}_{\text{SF}}(q,e,\text{Trino}) = \frac{m^{\star}_{\text{Trino}}(q,e) + H_{\text{headroom}}}{f_{\text{jvm}}}
$$

where $H_{\text{headroom}}$ is set by `memory.heap-headroom-per-node` and $f_{\text{jvm}}$ is the JVM heap fraction available after GC overhead.

#### Stage 3 — Quantize

Stage 3 quantization is unchanged from the general framework. All constants derive from documented Trino configuration parameters.

---

### DuckDB

DuckDB is an embedded, single-process C++ analytical engine — neither distributed nor an MPP — and falls outside the lakehouse setting evaluated in this work. We include it to show the pipeline applies when neither JVM memory management nor multi-node coordination is present.

#### Stage 1 — Extract

DuckDB's `BufferManager` exposes a `SYSTEM_PEAK_BUFFER_MEMORY` profiling metric (enabled via `PRAGMA custom_profiling_settings`) that reports the high-water mark of pinned memory during `EXPLAIN ANALYZE`. However, this metric tracks global `BufferManager` state; under concurrent execution it loses query attribution. We therefore restrict to serial execution, mirroring the isolation protocol used for Spark (§5.1):

$$
m^{\star}_{\text{DuckDB}}(q) := \texttt{SYSTEM\_PEAK\_BUFFER\_MEMORY}(q) \quad \text{[serial execution]}
$$

#### Stage 2 — Invert

Without JVM heap indirection, the inversion reduces to the buffer reservation plus an OS overhead margin $\delta_{\text{OS}}$ (page tables, thread stacks, file descriptors):

$$
\hat{p}_{\text{SF}}(q, \text{DuckDB}) = m^{\star}_{\text{DuckDB}}(q) + \delta_{\text{OS}}
$$

In a containerized deployment utilizing DuckDB's default 80% `memory_limit`, the remaining 20% of container capacity implicitly serves as this $\delta_{\text{OS}}$ margin.

#### Stage 3 — Quantize

Stage 3 quantization applies in containerized deployments; in DuckDB's native embedded mode it reduces to setting `memory_limit` $\ge \hat{p}_{\text{SF}}$.
