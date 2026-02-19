# Appendix A6 — Cluster and Execution Environment

All experiments run on a private OpenShift orchestrated cluster located in the IBM Research Zurich data center.

## Hardware

| Resource | Specification |
|----------|---------------|
| Nodes | 16 × Dell PowerEdge FC630 |
| CPU | Dual-socket Intel Xeon E5-2683 v4 @ 2.10 GHz (32 cores/node) |
| RAM | 768 GB DDR4 per node |
| Network | 1 × 25 Gbps Mellanox ConnectX-4 per node |
| Local storage | 120 GB SATA SSD per node |
| Total cluster RAM | 12.3 TiB |
| Total cluster cores | 512 physical cores |

Three nodes run OpenShift master VMs (control plane only); the remaining 15 are pure worker nodes.

## Storage

| Layer | System | Capacity | Role |
|-------|--------|----------|------|
| Object store | ZC2 Ceph S3 | Shared, >100 TB/s aggregate BW | Benchmark datasets (Parquet) |
| Persistent volumes | IBM Storage Scale (GPFS) on FlashSystem 9200 | 50 TiB | OpenShift PVCs, engine logs |

Benchmark datasets (TPC-H, TPC-DS at SF10; IMDB) reside in Ceph S3 as Apache Parquet files, accessed via an Apache Iceberg/Hive catalog. Both Spark and Presto read identical files through the same catalog endpoint with equivalent predicate pushdown and column pruning; neither engine benefits from proprietary storage optimizations.

## Engine Deployment

Spark and Presto engines run as containerized workloads on OpenShift pods, managed by IBM Analytics Engine (IAE) and IBM watsonx.data respectively. Worker pods are scheduled to dedicated nodes to eliminate co-tenancy effects. All four configurations (Spark-w1, Spark-w4, Presto-w1, Presto-w4) are deployed with fixed pod counts (`dynamicAllocation.enabled = false`, `query.max-concurrent-queries = 1`), ensuring no resource contention between concurrent queries or between engine instances.
