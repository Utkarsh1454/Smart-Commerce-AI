# Empirical Benchmarking Methodology & Scalability Analysis
**Course:** INT315 – Distributed and Cluster Computing (CA2)  
**Project:** Smart Commerce AI: A Distributed Personalized Product and Advertisement Recommendation System Using Apache Spark  
**Lead Author / Student:** Utkarsh Pandey  
**Institution:** Lovely Professional University  

---

## 1. Executive Summary & Academic Clarification

A central requirement of Course INT315 is rigorous, reproducible cluster computing evaluation and honest distinction between execution paradigms. In commercial cloud and high-performance computing literature, benchmarks frequently conflate **multi-threading on single-node shared-memory machines** with **distributed multi-node physical clusters**.

To uphold strict academic integrity:
1. **Execution Environment:** All benchmarks in this project were executed on an 8-logical-core x86_64 host under PySpark 3.5.1 on Java 17 using local multi-threaded parallelism (`local[*]`).
2. **Partition Scalability vs. Physical Multi-Node Clusters:** The experiments vary the Apache Spark RDD/DataFrame partition count ($p \in \{1, 2, 4, 8\}$) to evaluate task parallelization, scheduling overhead, thread contention, and shuffle serialization costs. We **do not** claim physical multi-node network cluster execution.
3. **Statistical Replication:** In accordance with standard systems benchmarking practices, all partition scalability measurements were repeated three independent times ($N=3$) after JIT JVM warmup and cache pre-materialization to calculate sample mean runtimes and standard deviations ($\sigma$).

---

## 2. Experimental Environment & Hardware Specifications

| Component | Specification |
|---|---|
| **Host Operating System** | Microsoft Windows 11 Home (x86_64 Architecture) |
| **Physical Processor** | AMD / Intel Multi-Core Processor (8 Logical SMT Threads) |
| **Host Memory (RAM)** | 16.0 GB DDR4 |
| **Storage Subsystem** | NVMe Solid-State Drive (PCIe Gen 3x4) |
| **Java Virtual Machine (JVM)** | OpenJDK Runtime Environment (build 17.0.10+7) |
| **Apache Spark Distribution** | Apache Spark 3.5.1 (Pre-built with Hadoop 3.3.6) |
| **Python Driver & Worker** | Python 3.13.1 (64-bit) |
| **Spark Master Mode** | `local[*]` (Spawns 8 parallel worker task threads) |
| **Driver Heap Memory Allocation** | 4.0 GB (`spark.driver.memory = "4g"`) |
| **Shuffle Partitions** | Dynamic / Evaluated across $p \in \{1, 2, 4, 8\}$ |

---

## 3. Benchmarking Framework & Pipeline Stages

The benchmark suite instruments every stage of the Smart Commerce AI distributed pipeline via wall-clock timing (`time.time()`) and Spark UI stage execution metrics:

```
[Input Raw CSVs: 57,000 Records]
          │
          ▼
Stage 1: Distributed Preprocessing & Temporal Feature Engineering (Parquet Storage)
   • Point-in-time temporal filtering (time_stamp < raw_sample.time_stamp)
   • Broadcast Hash Joins for ad_feature (400 items) and user_profile (1,200 users)
          │
          ├─────────────────────────────────────────────────┐
          ▼                                                 ▼
Stage 2A: Implicit-Feedback ALS Recommender       Stage 2B: Imbalance-Balanced CTR Classifiers
   • Matrix Factorization (Rank=12, Alpha=20)        • ElasticNet Logistic Regression vs. Random Forest
   • Top-10 Recs vs. Popularity Baseline             • Prior-Probability Baseline & Confusion Matrix
          │                                                 │
          └───────────────────────┬─────────────────────────┘
                                  ▼
Stage 3: Auction Serving & Ranking Engine
   • Point-in-time scoring & frequency capping (max 2 ads/campaign)
   • Single-file CSV coalescing & parquet exports
          │
          ▼
Stage 4: Controlled Partition Scalability Experiment (3 Replications, p ∈ {1, 2, 4, 8})
```

---

## 4. Controlled Scalability Experiment Methodology

To study how Spark's Catalyst engine and TaskScheduler scale on multi-core hardware, a standardized aggregation workload was executed on the preprocessed 15,000-record CTR feature dataset:
$$\text{Workload} = \sigma_{\text{price}>0}\left(\Pi_{\text{cate\_id}}\left(\text{COUNT}(*), \text{AVG}(\text{price}), \text{SUM}(\text{clk})\right)\right)$$

### Experimental Protocol
1. **Pre-computation & Isolation:** The dataset is partitioned into $p \in \{1, 2, 4, 8\}$ partitions and materialized into memory using `.cache()` and `.count()` before timing commences.
2. **Warmup & Replication:** For each partition count $p$, the aggregation pipeline is executed across three independent iterations:
   $$\bar{T}_p = \frac{1}{3}\sum_{r=1}^{3} T_{p, r}$$
   $$\sigma_p = \sqrt{\frac{1}{3}\sum_{r=1}^{3} (T_{p, r} - \bar{T}_p)^2}$$
3. **Speedup ($S_p$) and Parallel Efficiency ($E_p$):**
   $$S_p = \frac{\bar{T}_1}{\bar{T}_p}, \quad E_p = \frac{S_p}{p} \times 100\%$$

### Empirical Results Table

| Partitions ($p$) | Mean Time $\bar{T}_p$ (s) | Std Dev $\sigma_p$ (s) | Speedup $S_p$ | Parallel Efficiency $E_p$ (%) | Dominant Cost / Bottleneck |
|:---:|:---:|:---:|:---:|:---:|---|
| **1** | **0.0849** | $\pm 0.0222$ | **1.000×** | **100.0%** | Baseline (Single-thread in-memory scan) |
| **2** | **0.1311** | $\pm 0.0378$ | **0.648×** | **32.4%** | Task scheduling, thread handoff |
| **4** | **0.1000** | $\pm 0.0126$ | **0.849×** | **21.2%** | Intermediate aggregation speedup |
| **8** | **0.1305** | $\pm 0.0111$ | **0.651×** | **8.1%** | Micro-task fragmentation & shuffle merge |

### Academic Analysis: Amdahl's Law and Task Granularity
In distributed computing, **Amdahl's Law** states that the theoretical speedup is constrained by the sequential portion of the workload ($s$):
$$S_p = \frac{1}{s + \frac{1-s}{p}}$$

For small-to-moderate in-memory datasets ($\sim 15,000$ rows, $\sim 1.5$ MB Parquet), the task execution time ($\approx 10$–$20$ ms) is comparable to the Spark scheduler overhead ($\sim 15$–$30$ ms for task packaging, thread synchronization, and shuffle exchange). As a result:
- Increasing partitions from 1 to 8 introduces **micro-task fragmentation**, where the fixed overhead of scheduling 8 separate tasks exceeds the parallel compute benefit.
- When scaling to enterprise data volumes ($> 10^7$ rows), the compute time dominates scheduling overhead, yielding near-linear speedups up to the physical CPU core limit.

---

## 5. End-to-End Pipeline Stage Benchmarks

The full training and evaluation suite measured the following execution times and data throughputs:

| Pipeline Stage | Records Processed | Active Partitions | Execution Time (s) | Throughput (Records/s) | Status |
|---|:---:|:---:|:---:|:---:|:---:|
| **ALS Model Training (Implicit Feedback)** | 31,978 | 1 | 4.692 s | 6,815.6 rec/s | COMPLETED |
| **Top-10 Organic Recommendations (ALS)** | 1,200 users | 8 | 1.825 s | 657.5 users/s | COMPLETED |
| **CTR Training: Logistic Regression (ElasticNet)** | 10,617 | 1 | 4.709 s | 2,254.4 rec/s | COMPLETED |
| **CTR Inference: Logistic Regression** | 2,225 | 1 | 0.391 s | 5,696.9 rec/s | COMPLETED |
| **CTR Training: Random Forest (60 Trees, Depth 8)** | 10,617 | 1 | 3.937 s | 2,696.9 rec/s | COMPLETED |
| **CTR Inference: Random Forest** | 2,225 | 1 | 0.451 s | 4,931.9 rec/s | COMPLETED |
| **Auction Serving & Capping Engine** | 15,000 | 1 | 10.230 s | 1,466.3 rec/s | COMPLETED |

---

## 6. Verification of Spark Execution Plans (Broadcast Hash Joins)

To prevent severe distributed shuffle bottlenecks during multi-table joins, the preprocessing pipeline applies `F.broadcast()` to the smaller metadata dimensions (`ad_feature.csv` with 400 rows and `user_profile.csv` with 1,200 rows).

The physical execution plan was formally inspected via `.explain()`:
```
== Physical Plan ==
AdaptiveSparkPlan isFinalPlan=false
+- Project [clk#..., cate_id#..., brand_id#..., price#..., user_category_events#...]
   +- BroadcastHashJoin [cms_group_id#...], [cms_group_id#...], Inner, BuildRight
      :- BroadcastHashJoin [adgroup_id#...], [adgroup_id#...], Inner, BuildRight
      :  :- Filter (time_stamp#... > 0)
      :  :  +- Scan parquet [raw_sample]
      :  +- BroadcastExchange HashedRelationBroadcastMode
      :     +- Filter (price#... > 0.0)
      :        +- Scan parquet [ad_feature]
      +- BroadcastExchange HashedRelationBroadcastMode
         +- Scan parquet [user_profile]
```
**Conclusion:** The physical plan confirms `BroadcastHashJoin` with `BuildRight`, completely eliminating shuffle read/write steps on cluster worker nodes during dimension enrichment.

---

## 7. Instructions for Reproducing Benchmarks

To execute the entire empirical benchmark suite and regenerate all summary CSVs and JSON reports:

```powershell
# 1. Activate project virtual environment
cd "Smart_Commerce_AI"
.\.venv\Scripts\Activate.ps1

# 2. Run automated dataset integrity verification
python scripts/validate_dataset_integrity.py

# 3. Run temporal feature engineering (guaranteeing zero data leakage)
python -m src.preprocess

# 4. Run model training and 3x statistical partition benchmarking
python -m src.train

# 5. Run serving engine and export frequency-capped recommendations
python -m src.recommend
```
All empirical benchmark outputs are automatically exported to:
- `Smart_Commerce_AI/results/benchmark_summary.json`
- `Smart_Commerce_AI/results/cluster_scalability_experiment.csv`
- `Smart_Commerce_AI/results/cluster_performance_benchmarks.csv`
