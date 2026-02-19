# Appendix A1 — Queries and Measurements from the Motivating Example

This appendix provides the full SQL text and per-engine measurements for the two TPC-H queries used in the motivating example of Section 1 (Figure 1). Both queries were synthetically generated over the TPC-H schema (SF10) using the method of Nidd et al. (2025). They were executed on four engine configurations:

| Label in paper | Engine   | Workers |
|----------------|----------|---------|
| Eng1           | PrestoDB | 1       |
| Eng2           | PrestoDB | 4       |
| Eng3           | Spark    | 1       |
| Eng4           | Spark    | 4       |

Cost is reported in **base-pod-seconds** (execution time × per-worker cost multiplier κ), where κ is determined by the quantized memory tier assigned to the worker pod. See Section 2 and Appendix A3 for the full cost formulation and engine configurations.

---

## Query 1

*Characterized by a classic Pareto frontier: no single engine dominates on both latency and cost.*

```sql
SELECT
    AVG(LENGTH(NATION.N_NAME)),
    AVG(LENGTH(ORDERS.O_COMMENT)),
    AVG(LENGTH(ORDERS.O_ORDERPRIORITY)),
    AVG(LENGTH(CUSTOMER.C_ADDRESS)),
    AVG(LENGTH(CUSTOMER.C_PHONE)),
    AVG(CUSTOMER.C_ACCTBAL),
    SUM(SUPPLIER.S_ACCTBAL),
    AVG(LENGTH(LINEITEM.L_SHIPMODE)),
    AVG(LENGTH(LINEITEM.L_RETURNFLAG)),
    AVG(LENGTH(LINEITEM.L_COMMENT)),
    AVG(LENGTH(REGION.R_NAME)),
    LINEITEM.L_SHIPMODE AS L_SHIPMODE
FROM
    NATION
    JOIN REGION ON REGION.R_REGIONKEY = NATION.N_REGIONKEY,
    ORDERS,
    CUSTOMER,
    SUPPLIER,
    LINEITEM
WHERE
    CUSTOMER.C_NATIONKEY = NATION.N_NATIONKEY
    AND LINEITEM.L_SUPPKEY = ORDERS.O_ORDERKEY
    AND SUPPLIER.S_NATIONKEY = CUSTOMER.C_CUSTKEY
    AND LINEITEM.L_SUPPKEY = SUPPLIER.S_SUPPKEY
GROUP BY
    L_SHIPMODE
ORDER BY
    AVG(LENGTH(CUSTOMER.C_ADDRESS))
```

### Measurements

| Metric | Eng1 (Presto-w1) | Eng2 (Presto-w4) | Eng3 (Spark-w1) | Eng4 (Spark-w4) |
|---|---|---|---|---|
| **Execution time (s)** | 27.69 | 14.69 | 8.67 | 7.40 |
| **Native memory (MB)** | 167.45 | 86.60 | 4292.00 | 1172.00 |
| **MVP pod size (GB)** | 0.33 | 0.17 | 6.47 | 2.00 |
| **Quantized pod tier (MB)** | 2048 | 2048 | 8192 | 4096 |
| **Cost multiplier κ** | 1× | 1× | 4× | 2× |
| **Cost (base-pod-seconds)** | 27.69 | 58.76 | 34.68 | 59.23 |

**Takeaway:** Eng1 (Presto-w1) is the most cost-efficient (27.69) but the slowest (27.69 s). Eng4 (Spark-w4) achieves the lowest latency (7.40 s) at a 2.14× cost premium. No engine simultaneously minimises both objectives — a classic Pareto trade-off.

---

## Query 2

*Characterized by a "silver bullet" engine that is simultaneously the fastest and the least expensive.*

```sql
SELECT
    NATION.N_NAME,
    REGION.R_NAME,
    AVG(ORDERS.O_TOTALPRICE)          AS AvgOrderPrice,
    COUNT(DISTINCT ORDERS.O_ORDERKEY) AS TotalOrders,
    AVG(CUSTOMER.C_ACCTBAL)           AS AvgCustomerBalance,
    COUNT(DISTINCT SUPPLIER.S_SUPPKEY) AS TotalSuppliers,
    SUM(LINEITEM.L_EXTENDEDPRICE)     AS TotalExtendedPrice,
    AVG(LENGTH(NATION.N_NAME))        AS AvgNationNameLength,
    AVG(LENGTH(REGION.R_NAME))        AS AvgRegionNameLength
FROM
    NATION
    INNER JOIN REGION   ON REGION.R_REGIONKEY   = NATION.N_REGIONKEY
    INNER JOIN CUSTOMER ON NATION.N_NATIONKEY   = CUSTOMER.C_NATIONKEY
    INNER JOIN ORDERS   ON CUSTOMER.C_CUSTKEY   = ORDERS.O_CUSTKEY
    INNER JOIN LINEITEM ON ORDERS.O_ORDERKEY    = LINEITEM.L_ORDERKEY
    INNER JOIN SUPPLIER ON LINEITEM.L_SUPPKEY   = SUPPLIER.S_SUPPKEY
GROUP BY
    NATION.N_NAME,
    REGION.R_NAME
```

### Measurements

| Metric | Eng1 (Presto-w1) | Eng2 (Presto-w4) | Eng3 (Spark-w1) | Eng4 (Spark-w4) |
|---|---|---|---|---|
| **Execution time (s)** | 96.00 | 17.12 | 9.97 | 12.24 |
| **Native memory (MB)** | 2785.28 | 824.66 | 4356.00 | 1092.00 |
| **MVP pod size (GB)** | 5.49 | 1.62 | 6.56 | 1.89 |
| **Quantized pod tier (MB)** | 8192 | 2048 | 8192 | 2048 |
| **Cost multiplier κ** | 4× | 1× | 4× | 1× |
| **Cost (base-pod-seconds)** | 384.00 | 68.48 | 39.88 | 48.95 |

**Takeaway:** Eng3 (Spark-w1) is both the fastest (9.97 s) and the cheapest (39.88 base-pod-seconds), making it the Pareto-optimal "silver bullet". This arises because Spark-w1 fits its working set within the base memory tier (κ = 4×, but extremely short runtime), while the high native memory of Presto-w1 forces a larger pod tier (κ = 4×) with no speed advantage. A provisioning-unaware router optimising for latency alone would still find Eng3, but would not reliably generalise this reasoning to unseen queries — the provisioning-aware formulation of QueryPARCEL does.