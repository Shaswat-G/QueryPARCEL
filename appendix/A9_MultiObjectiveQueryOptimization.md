# Appendix A9 — Positioning of Prior Systems on Multi-Objective Query Optimization

A prior system is a *directly comparable* baseline only if it solves the same optimization problem. We make this precise with a six-component problem definition:

$$\langle\\text{scope},\ \text{granularity},\ \text{decision space},\ \text{decision-time information},\ \text{objective},\ \text{constraints}\,\rangle,$$

instantiated for our setting as:

- **Scope:** cross-engine (Spark and Presto), not single-engine optimization;
- **Granularity:** per-query, not workload-level planning;
- **Decision space:** discrete routing tuples $(\text{engine},\ \text{worker count})$, not physical plans or tuning knobs;
- **Decision-time information:** engines pre-provisioned and static; no reprovisioning;
- **Objective:** monetary cost coupling predicted latency and a predicted peak-memory provisioning tier via the elastic-billing factor $\kappa(p_{\text{SF}})$;
- **Constraints:** per-query hard SLO with explicit infeasibility (abstention).

Operationally, a runnable baseline must therefore jointly support: **(L)** per-query latency prediction, **(M)** per-query peak-memory prediction as a provisioning-tier target, **(C)** constrained or multi-objective optimization with formalization of infeasibility, and **(X)** heterogeneous cross-engine routing. No prior MOQO/MPQ, learned-router, polystore, or cross-platform system supports all four.

## Column definitions

Each column is a genuine axis of variation across the literature, evaluated under a fixed criterion:

| Component | Requires |
|---|---|
| **Scope** | routing across $\geq 2$ heterogeneous engines |
| **Granularity** | a per-query decision (not a workload/blueprint-level plan) |
| **Decision space** | discrete $(\text{engine}, \text{worker count})$ routing tuples |
| **Decision-time info** | static, pre-provisioned engines; no reprovisioning rights |
| **Objective** | cost coupling latency and a peak-memory provisioning tier via $\kappa(p_{\text{SF}})$ † |
| **Constraints** | a per-query hard SLO with explicit infeasibility/abstention |

† Systems with *related but distinct* objectives — e.g., a latency–cost Pareto frontier without a memory-tier target — are marked ✗ on Objective, since the distinguishing quantity is the memory-tier coupling, not the presence of a cost term.

## Table A9.1 — Component-wise alignment

A baseline must match on all six components. No prior system matches all six; every system mismatches at least one component carrying this paper's contributions.

| System | Scope | Granularity | Decision space | Decision-time info | Objective | Constraints |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| T&K MOQO/MPQ '14–'17 [1,2] | ✗ | ✓ | ✗ | ✓ | ✗ | ✓ |
| Georgoulakis Misegiannis et al. '22 [3] | ✗ | ✓ | ✗ | ✓ | ✗ | ✓ |
| UDAO / UDAO-Spark [4] | ✗ | ✓ | ✗ | ✓ | ✗ | ✓ |
| Auto-WLM [5] | ✗ | ✓ | ✗ | ✓ | ✗ | ✗ |
| RHEEM / Wayang [6] | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ |
| Strausz et al. '25 [7] | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ |
| BRAD [8] | ✓ | ✗ | ✗ | ✗ | ✗ | ✓ |
| **This work** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

## Table A9.2 — Per-cell mismatch reasons

| System | Mismatched components |
|---|---|
| **T&K MOQO/MPQ** [1,2] | *Scope:* PostgreSQL only. *Decision space:* physical plans, exploiting subplan dynamic programming. *Objective:* analytical plan-cost formulas; no per-query memory prediction, no $\kappa(\cdot)$. |
| **Georgoulakis Misegiannis et al.** [3] | *Scope:* Spark SQL only. *Decision space:* Catalyst plans + configuration. *Objective:* assumes all data fits in memory; no $p_{\text{SF}}$ analog. |
| **UDAO / UDAO-Spark** [4] | *Scope:* Spark only. *Decision space:* runtime knobs. *Objective:* latency and monetary cost as decoupled Pareto axes; no memory tier as an optimization target. |
| **Auto-WLM** [5] | *Scope:* Redshift only. *Decision space:* queueing / short-query acceleration / cluster scaling. *Objective:* latency- and memory-prediction features for admission, not a per-query routing cost. *Constraints:* admission control, not SLO infeasibility. |
| **RHEEM / Wayang** [6] | *Objective:* single scalar (runtime + data movement); no memory tier, no $\kappa(\cdot)$. *Constraints:* none. |
| **Strausz et al.** [7] | *Objective:* single-objective predicted runtime; no memory tier, no $\kappa(\cdot)$. *Constraints:* $\arg\min$ over engines; no SLO infeasibility. |
| **BRAD** [8] | *Granularity:* workload-level blueprint. *Decision space:* engine set + table placement + provisioning + routing policy. *Decision-time info:* assumes reprovisioning rights over managed cloud warehouses (Aurora/Redshift/Athena) whose memory governance is opaque and non-invertible; provisioning is extrapolated analytically (Amdahl-law scaling) rather than measured. *Objective:* monetary cost s.t. p99 latency; no per-query memory tier, no $\kappa(p_{\text{SF}})$. |

## Synthesis

The mismatches partition into three classes.

**Single-engine MOQO and tuning** (T&K [1,2]; Georgoulakis Misegiannis et al. [3]; UDAO/UDAO-Spark [4]) supply-constrained and Pareto-optimal machinery but optimize physical plans or runtime knobs within a single engine, with no cross-engine decision variable and no memory-tier objective. **Cross-platform routers** (RHEEM/Wayang [6]; Strausz et al. [7]) match scope and decision space but are single-objective over runtime, lacking both a memory-tier objective and infeasibility-aware constraints. **Workload-level blueprint search** (BRAD [8]) is the closest partial match but solves a different problem: it optimizes engine set, table placement, provisioning, and routing policy over reprovisionable managed warehouses whose memory governance is non-invertible, whereas we route per-query across static lakehouse engines with explicit memory-tier modeling and per-query infeasibility-aware SLOs.

Beyond the scope mismatch, plan-space MOQO does not transfer *algorithmically*. Classical MOQO/MPQ searches within the intra-engine physical-plan space via dynamic programming or Pareto approximation over plan frontiers. Our decision space is instead the small discrete cross-product of routing tuples $(\text{engine}, \text{worker count})$, over which multi-objective selection reduces to skyline enumeration on predicted metrics — which the router already performs. The dynamic-programming and approximation machinery that constitutes plan-space MOQO is therefore inapplicable, independent of scope.

Adapting any partial match to our setting would require importing this paper's core contributions — the engine adapters, per-query peak-memory prediction, the billing model with $p_{\text{SF}}$ and $\kappa(\cdot)$, and the explicit SLO-infeasibility formalization. Accordingly, we position these systems as related work rather than as directly comparable empirical baselines.

## References

1. I. Trummer and C. Koch. *Approximation Schemes for Many-Objective Query Optimization.* SIGMOD 2014, pp. 1299–1310. DOI: [10.1145/2588555.2610527](https://doi.org/10.1145/2588555.2610527). (arXiv:1404.0046)
2. I. Trummer and C. Koch. *Multi-Objective Parametric Query Optimization.* PVLDB 8(3):221–232, 2015; extended in The VLDB Journal 26:107–124, 2017. DOI: [10.1007/s00778-016-0439-0](https://doi.org/10.1007/s00778-016-0439-0).
3. M. Georgoulakis Misegiannis, V. Kantere, and L. d'Orazio. *Multi-objective query optimization in Spark SQL.* IDEAS '22, pp. 70–74. DOI: [10.1145/3548785.3548800](https://doi.org/10.1145/3548785.3548800).
4. C. Lyu, Q. Fan, P. Guyard, and Y. Diao. *A Spark Optimizer for Adaptive, Fine-Grained Parameter Tuning* (UDAO-Spark). PVLDB 17(11):3565–3579, 2024. DOI: [10.14778/3681954.3682021](https://doi.org/10.14778/3681954.3682021). (arXiv:2403.00995). System lineage: K. Zaouk et al., *UDAO*, 2019; F. Song et al., *Spark-based Cloud Data Analytics using Multi-Objective Optimization*, ICDE 2021.
5. G. Saxena, M. Rahman, N. Chainani, C. Lin, G. Caragea, F. Chowdhury, R. Marcus, T. Kraska, I. Pandis, and B. Narayanaswamy. *Auto-WLM: Machine Learning Enhanced Workload Management in Amazon Redshift.* SIGMOD Companion 2023, pp. 225–237. DOI: [10.1145/3555041.3589677](https://doi.org/10.1145/3555041.3589677).
6. D. Agrawal, S. Chawla, B. Contreras-Rojas, A. Elmagarmid, Y. Idris, Z. Kaoudi, S. Kruse, J. Lucas, E. Mansour, M. Ouzzani, P. Papotti, J.-A. Quiané-Ruiz, N. Tang, S. Thirumuruganathan, and A. Troudi. *RHEEM: Enabling Cross-Platform Data Processing — May The Big Data Be With You!* PVLDB 11(11):1414–1427, 2018. DOI: [10.14778/3236187.3236195](https://doi.org/10.14778/3236187.3236195).
7. A. Strausz, N. Pardon, and I. Giurgiu. *A Learned Cost Model-based Cross-engine Optimizer for SQL Workloads.* VLDB 2025 Workshop: Third International Workshop on Composable Data Management Systems (CDMS). arXiv:[2506.02802](https://arxiv.org/abs/2506.02802).
8. G. X. Yu et al. *Blueprinting the Cloud: Unifying and Automatically Optimizing Cloud Data Infrastructures with BRAD.* PVLDB, 2024. DOI: [10.14778/3681954.3682026](https://doi.org/10.14778/3681954.3682026). (arXiv:2407.15363)

---

*Notes on scope.* RHEEM/Wayang is represented by the cross-platform optimizer of
Agrawal et al. [6] (the RHEEMix cost-based optimizer, *VLDB Journal* 2020, is the
optimizer-specific extension of the same system). Cost-Intelligent Data Analytics in
the Cloud (Zhang, CIDR 2024; arXiv:2308.09569) is a vision paper, not a system, and is
therefore excluded as a baseline candidate.
