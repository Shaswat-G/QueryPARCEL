# Appendix A5 — Datasets and Workloads

## Training and Evaluation Corpora (TPC-H and TPC-DS)

Both benchmarks are instantiated at **Scale Factor 10** (SF10), stored as **Apache Parquet** on S3-compatible object storage, read via a shared Hive metastore/catalog.

| Dataset | Tables | FK Relations | Rows (M) | Raw Size (GB) | Parquet Size (GB) | Queries |
|---------|--------|-------------|----------|--------------|-------------------|---------|
| TPC-H   | 8      | 8           | 86.6     | 10.0         | 3.2               | 1,960   |
| TPC-DS  | 24     | 102         | 191.5    | 10.0         | 4.2               | 4,388   |
| **Total** | —    | —           | —        | —            | —                 | **6,348** |

Query counts reflect successfully executed queries completing within the 125 s timeout across all four engine configurations. Queries that fail on any configuration (OOM, parse error, or timeout) are excluded from the corpus.

### TPC-H

Eight normalized tables (star schema with a central `lineitem` fact table) connected by 8 foreign-key constraints. The schema covers a supply-chain OLAP workload with dominant operators being multi-way hash joins, aggregations, and sort-based top-k. The canonical 22 TPC-H templates produce fewer than 100 distinct queries; we augment with synthetically recombined queries following the template-recombination method of Nidd et al. (2025), yielding 1,960 structurally diverse queries.

### TPC-DS

Twenty-four tables (snowflake schema with multiple fact tables: `store_sales`, `catalog_sales`, `web_sales`, and their returns counterparts) connected by 102 foreign-key constraints. The schema covers a retail decision-support workload with complex multi-join subqueries, window functions, and correlated predicates. Query augmentation uses LLM-generated variants from the SQLStorm benchmark (Schmidt et al., 2025), yielding 4,388 queries.

### Schema Complexity Contrast

TPC-DS presents substantially higher schema complexity than TPC-H: 3× more tables, 12.75× more FK relations, and 2.2× more rows. This translates directly into larger plan graphs (more join nodes, wider operator trees) and a more demanding test of the GNN's ability to handle structural diversity.

### Data Splits

The 6,348-query corpus is split via stratified sampling:

| Split | Fraction | Queries |
|-------|----------|---------|
| Train | 80%      | 5,078   |
| Validation | 10% | 635     |
| Test  | 10%      | 635     |

Stratification ensures TPC-H contributes ~31% and TPC-DS ~69% to each split, preserving the overall schema complexity distribution. The validation set is used solely for hyperparameter selection and early-stopping checkpoint determination; the test set is untouched during all optimization stages.

---

## Query Complexity Distribution

| Dataset | Avg. Joins | Max Joins | Avg. GroupBy Cols | Max GroupBy Cols | Avg. Operators |
|---------|-----------|-----------|-------------------|------------------|----------------|
| TPC-H   | 2.45      | 8         | 1.46              | 12               | 7.73           |
| TPC-DS  | 2.20      | 20        | 3.56              | 34               | 3.23           |

Operator counts include all relational and expression nodes in optimized Substrait plans. The two benchmarks exercise complementary cost dimensions:

- **TPC-H** stresses join enumeration and cardinality estimation: higher average operator count (7.73 vs. 3.23) reflects deeper plan trees from join-heavy star-schema navigation; aggregations are simple (avg. 1.46 group-by columns).
- **TPC-DS** stresses aggregation memory footprints: multi-fact snowflake schema extends joins to depth 20 and group-by to 34 columns, with complex multi-dimensional rollups and correlated subqueries distributing complexity into aggregation rather than join depth.

Together the two benchmarks exercise heterogeneous cost dimensions required for robust routing model training.

---

## Zero-Shot Transfer Corpus (IMDB JOB)

The Join Order Benchmark (JOB) over the Internet Movie Database (IMDB) is used exclusively for the few-shot adaptation experiment (RQ6). It is **not** used in any training or evaluation of the main model.

| Dataset | Tables | FK Relations | Rows (M) | Raw Size (GB) | Parquet Size (GB) | Queries |
|---------|--------|-------------|----------|--------------|-------------------|---------|
| IMDB    | 23     | 17          | 74.3     | 3.6          | 1.8               | 728     |

The IMDB schema covers 23 tables with 17 FK relations describing movies, actors, directors, genres, companies, and their associations. The 728 JOB queries consist of complex multi-join chains (up to 17 joins), with heavy selectivity variation across predicates — a distribution shift from the TPC benchmarks in both schema topology and predicate structure.

### Few-Shot Split

| Split | Queries |
|-------|---------|
| Fine-tuning pool ($n$ labeled samples, $n \in \{0, 50, 100, \ldots, 500\}$) | 600 |
| Test (held out) | 128 |

$n = 0$ corresponds to **zero-shot transfer** (model trained on TPC-H/DS, evaluated directly on IMDB without any adaptation).

---

## Storage Format

All datasets are stored as **column-oriented Apache Parquet** files with Snappy compression on S3-compatible object storage, accessed via a shared Iceberg/Hive metastore. Table statistics (row counts, column NDVs, min/max values) are maintained in the metastore and consulted by both Spark's and Presto's optimizers during logical plan decoration. The Parquet representation yields a 2.4–3.1× size reduction over raw CSV owing to columnar encoding and compression, but does not alter the logical schema or query semantics.
