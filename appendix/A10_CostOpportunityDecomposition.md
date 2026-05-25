# Appendix A10: Cost-Opportunity Decomposition

This appendix isolates how much of the cost-routing opportunity is attributable to per-query memory-tier selection, independent of predictor error. It supports the claim in §3.1 that the memory term dominates the gap between latency- and cost-optimal routing.

## Setup

The **cost-routing opportunity** is the gap between two oracles on the test set:

- **Latency oracle** selects $\arg\min_e t(q,e)$ (the fastest engine).
- **Cost oracle** selects $\arg\min_e c(q,e)$ (the cheapest engine under the cost functional of Eq. 3).

The latency oracle overspends the cost oracle by **57.6%** (§2). We attribute this gap with a **capture** metric:

$$\text{Capture}(\pi) = \frac{C_{\text{lat}} - C_\pi}{C_{\text{lat}} - C_{\text{cost}}}$$

where $C_\pi$ is the total realized cost under policy $\pi$. By construction the latency oracle captures 0% and the cost oracle 100%.

> **Note on comparability.** This denominator is the *oracle-to-oracle* gap, not the best-fixed-engine baseline used for the `Gain`/`Capture` metrics in §RQ2. The two capture quantities answer different questions and are not directly comparable.

## The memory-blind baseline

To separate the contribution of memory from that of parallelism, we evaluate a **constant-threshold** policy that is granted true latencies but holds the memory tier fixed. Holding $\kappa(p_{\text{SF}})$ constant in Eq. 3 collapses the cost objective to

$$\arg\min_e \; t(q,e)\cdot n_e,$$

so this policy is fully *parallelism-aware* — it sees both true latency and worker count — but *memory-blind*. Any opportunity it cannot reach is therefore attributable to per-query memory-tier differences alone, measured at the oracle bound and free of prediction error.

## Results

| Policy | Capture (aggregate) | Capture (per-query median) |
|---|---:|---:|
| Latency oracle (baseline) | 0.0% | 0.0% |
| Constant tier (true $t$, memory-blind) | 13.7% | — |
| XGBoost (cost-routed) | 16.2% | — |
| GNN (cost-routed) | 69.1% | — |
| Cost oracle (floor) | 100.0% | 100.0% |

*(Per-query median columns to be filled once computed; see note below.)*

**Engine selections under the constant-threshold policy:**

| Engine | Share |
|---|---:|
| `Spark-w1` | 53.1% |
| `Presto-w1` | 46.9% |
| `Spark-w4` | 0.0% |
| `Presto-w4` | 0.0% |

## Interpretation

**Memory drives the gap, not parallelism.** The memory-blind policy recovers only **13.7%** of the cost opportunity despite perfect latency and full worker-count awareness. The remaining **86.3%** is reachable only by selecting the correct per-query memory tier — isolating memory as the dominant cost lever at the oracle bound.

**The constant threshold degenerates to a latency-only choice among single-worker engines.** Because latency scales sub-linearly with workers, $t\cdot n_e$ never favors a 4-worker configuration; the policy selects exclusively single-worker engines (0% on both `w4` configurations) and reduces to picking the faster of the two. It thus collapses to the very latency-only regime shown insufficient in §2 — which is why it captures so little.

**A tabular model barely clears the memory-blind floor.** XGBoost recovers **16.2%**, only 2.5 points above the 13.7% memory-blind baseline, whereas the GNN recovers **69.1%** (4.3× the XGBoost capture). The cost benefit therefore requires *structured, query-specific* memory prediction: neither a constant threshold nor a tabular plan-aggregate model supplies it.