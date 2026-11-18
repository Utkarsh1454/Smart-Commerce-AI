# Smart Commerce AI 🛍️⚡
### A Distributed Personalized Product & Advertisement Recommendation System Using Apache Spark
**Course Project:** INT315 — Distributed and Cluster Computing (Continuous Assessment 2)  
**Author:** Utkarsh Pandey  
**Institution:** School of Computer Science & Engineering, Lovely Professional University  
**Cluster Architecture:** Apache Spark 3.5.1 on 8 Logical Cores (`local[*]`), Java 17 LTS, Apache Parquet with Snappy Compression.

---

## 🌟 Executive Summary

**Smart Commerce AI** is a production-grade distributed recommendation and real-time advertisement auction platform inspired by the architectures of Alibaba Taobao, Amazon, and Flipkart. Engineered specifically for **Course INT315 – Distributed and Cluster Computing**, the project addresses the dual computational workloads of modern digital commerce:

1. **Organic Product Personalization:** Generates high-affinity recommendations from implicit browsing signals using **Spark MLlib Implicit Alternating Least Squares (ALS)** with confidence weighting ($c_{ui} = 1 + \alpha r_{ui}$), evaluated against a **Popularity-Based Baseline**.
2. **Sponsored Advertisement Auction Engine:** Forecasts Click-Through Rates (CTR) under acute positive class imbalance (13.0% click rate) using distributed **ElasticNet Logistic Regression** and **Random Forest (60 trees, depth 8)** with inverse class weighting and validation-tuned decision thresholds, benchmarked against a **Prior-Probability Baseline**.
3. **Strict Point-in-Time Temporal Filtering (Zero Data Leakage):** Eliminates future-event leakage by enforcing $t_{\text{behavior}} < t_{\text{impression}}$, validated via automated unit tests.
4. **Cluster Scalability & Benchmarking:** Conducts statistically replicated ($N=3$) partition experiments across $p \in \{1, 2, 4, 8\}$ partitions, measuring runtimes, standard deviations, speedup, and parallel efficiency under Amdahl's Law.
5. **Commercial Safeguards:** Enforces ad frequency capping (maximum 2 impressions per campaign per user) and inventory price sanity filtering ($price > 0$).

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    subgraph DataLayer ["Data Storage & Ingestion Layer"]
        A["Educational Alibaba Taobao Stream<br/>(42,000 Behavior Events)"]
        B["Raw Ad Impressions<br/>(15,000 Ad Samples)"]
        C["User Demographic Profiles<br/>(1,200 Customers)"]
        D["Ad Creative Metadata<br/>(400 Items)"]
    end

    subgraph SparkCluster ["Apache Spark 3.5.1 Distributed Cluster (local[*], 8 Cores)"]
        E["Point-in-Time Temporal Filter<br/>(time_stamp < raw_sample.time_stamp)"]
        F["Catalyst Broadcast Hash Joins<br/>F.broadcast(ad_feature, user_profile)"]
        G["Spark MLlib Implicit ALS<br/>(Rank=12, α=20.0, regParam=0.08)"]
        H["CTR Binary Classifiers<br/>(Class Weighting + Threshold Optimization)"]
        I["Commercial Safeguards<br/>Frequency Capping & Inventory Filter"]
    end

    subgraph StorageLayer ["High-Performance Storage Layer"]
        J[("Snappy-Compressed<br/>Columnar Apache Parquet")]
        K[("Single-File CSV Exports<br/>results/product_recommendations_csv<br/>results/ad_recommendations_csv")]
    end

    subgraph PresentationLayer ["Dual Serving Frontends"]
        L["🛍️ Interactive E-Commerce Storefront<br/>(index.html: Port 3000, Light Palette, Live Search, Cart)"]
        M["📊 Streamlit Cluster Analytics Dashboard<br/>(app.py: Port 8501, Empirical Metrics & Amdahl Curves)"]
    end

    A --> E
    B --> E
    E --> F
    C --> F
    D --> F
    F --> G
    F --> H
    G --> I
    H --> I
    I --> J
    I --> K
    K --> L
    J --> M
```

---

## 📊 Comprehensive Empirical Benchmark Results

All benchmarks below were empirically measured on an 8-core host running **Apache Spark 3.5.1** on **Java 17 LTS**:

### 1. Product Recommendation: Implicit ALS vs. Popularity Baseline
*Evaluated on held-out test split ($N=7,980$ interactions) across 1,920 candidate items:*

| Evaluation Metric | Popularity Baseline | Implicit ALS Recommender | Absolute Gain ($\Delta$) | Relative Gain (%) |
|---|:---:|:---:|:---:|:---:|
| **NDCG@5** | 0.00319 | **0.01591** | +0.01272 | **+398.7%** |
| **Precision@5** | 0.00317 | **0.01501** | +0.01184 | **+373.5%** |
| **Recall@5** | 0.00252 | **0.01152** | +0.00900 | **+357.1%** |
| **MAP@5** | 0.00146 | **0.00799** | +0.00653 | **+447.3%** |
| **NDCG@10** | 0.00388 | **0.01967** | +0.01579 | **+406.9% (5.07×)** |
| **Precision@10** | 0.00284 | **0.01468** | +0.01184 | **+416.9% (5.17×)** |
| **Recall@10** | 0.00484 | **0.02286** | +0.01802 | **+372.3%** |
| **MAP@10** | 0.00141 | **0.00761** | +0.00620 | **+439.7%** |
| **Catalog Coverage (%)** | **0.52%** (10 items) | **52.50%** (1,008 items) | +51.98% | **+10,096% (101×)** |
| **Training Duration** | 0.050 s | 4.692 s | — | Throughput: **6,815.6 rec/s** |
| **Serving Latency** | 0.100 s | 1.825 s | — | Throughput: **657.5 users/s** |

### 2. Ad CTR Prediction: Baselines & Calibration Comparison
*Evaluated on independent test split ($N=2,225$, 13% positive click rate) with inverse weighting ($w_{\text{pos}}=3.886, w_{\text{neg}}=0.574$):*

| Metric | Prior-Probability Baseline | Logistic Regression (ElasticNet) | Random Forest (60 Trees, Depth 8) | Best Performing Model |
|---|:---:|:---:|:---:|:---:|
| **ROC-AUC** | 0.50000 | **0.58298** | 0.58217 | Logistic Regression (+0.1%) |
| **PR-AUC** | 0.12867 | 0.14333 | **0.14528** | **Random Forest (+1.4%)** |
| **Classification Accuracy** | 0.87133 | 0.20764 | **0.30112** | Prior (Trivial majority class) |
| **Positive Precision** | 0.00000 | 0.12879 | **0.14006** | **Random Forest (+8.7%)** |
| **Positive Recall** | 0.00000 | **0.95926** | 0.92593 | **Logistic Regression (+3.6%)** |
| **Positive F1-Score** | 0.00000 | 0.22709 | **0.24331** | **Random Forest (+7.1%)** |
| **Log-Loss (Cross-Entropy)** | 0.38385 | 0.68564 | **0.59627** | **Random Forest (-13.0%)** |
| **Tuned Threshold ($\tau^*$)** | 0.50 | 0.40 | 0.30 | Tuned on validation split |
| **True Positives (TP)** | 0 | 259 | 250 | Out of 270 test clicks |
| **False Positives (FP)** | 0 | 1,752 | 1,535 | Out of 1,955 non-clicks |
| **True Negatives (TN)** | 1,938 | 203 | 420 | Out of 1,955 non-clicks |
| **False Negatives (FN)** | 286 | 11 | 20 | Out of 270 test clicks |

### 3. Controlled 3-Repetition Partition Scalability Benchmark
*Standardized aggregation workload on 15,000 records across 3 independent replications ($N=3$):*

| Partitions ($p$) | Mean Time $\bar{T}_p$ (s) | Std Dev $\sigma_p$ (s) | Speedup $S_p$ | Efficiency $E_p$ (%) | Distributed Performance Analysis |
|:---:|:---:|:---:|:---:|:---:|---|
| **1** | **0.0849 s** | $\pm 0.0222$ s | **1.000×** | **100.0%** | Single-partition contiguous in-memory scan |
| **2** | 0.1311 s | $\pm 0.0378$ s | 0.648× | 32.4% | Task scheduling and thread context switching |
| **4** | 0.1000 s | $\pm 0.0126$ s | 0.849× | 21.2% | Parallel aggregation gain vs. coordination |
| **8** | 0.1305 s | $\pm 0.0111$ s | 0.651× | 8.1% | Micro-task fragmentation & shuffle merge overhead |

---

## 📁 Repository Structure & Audited Codebase

```
Smart-Commerce-AI/
├── Smart_Commerce_AI/
│   ├── data/
│   │   ├── raw/                       # Educational CSV datasets (42k behaviors, 15k impressions)
│   │   ├── processed/                 # Parquet feature stores (ctr_features, interactions)
│   │   └── README.md                  # Dataset provenance, schema, and license documentation
│   ├── src/
│   │   ├── config.py                  # PySpark runtime configuration & path resolution
│   │   ├── preprocess.py              # Temporal filtering & Broadcast Hash Joins
│   │   ├── train.py                   # Distributed ALS, LogReg, Random Forest, & Scalability suite
│   │   └── recommend.py               # Serving engine with frequency capping & CSV exports
│   ├── tests/
│   │   └── test_data_leakage.py       # Automated unit test verifying point-in-time filtering
│   ├── scripts/
│   │   ├── validate_dataset_integrity.py  # Automated schema & data validation
│   │   └── download_taobao_dataset.py     # RecZoo TaobaoAd_x1 downloader
│   ├── results/                       # Exported benchmark CSVs, JSON, and CSV recommendations
│   ├── models/                        # Saved Spark MLlib model artifacts
│   ├── research_paper/
│   │   ├── Research_Paper_Draft.md    # Springer-formatted revised research paper
│   │   └── PRISMA_Systematic_Review_Paper.md  # PRISMA 2020 systematic review
│   ├── app.py                         # Interactive Streamlit Cluster Analytics Dashboard
│   ├── requirements.txt               # Locked production dependencies
│   └── vercel.json                    # Serverless cloud deployment configuration
├── index.html                         # Interactive Storefront (Tailwind CSS, Light Theme)
├── AUDIT_REPORT.md                    # Comprehensive 11-point audit & remediation report
├── DATASET_DOCUMENTATION.md           # Dataset provenance & schema verification
├── BENCHMARK_METHODOLOGY.md           # Benchmarking protocols, local[*] vs. cluster, Amdahl analysis
├── EXPERIMENT_RESULTS.md              # Full empirical tables & baseline comparisons
├── TEST_REPORT.md                     # Comprehensive automated test execution record
├── RUN_GUIDE.md                       # Step-by-step operational setup & execution guide
└── README.md                          # Root academic and technical overview
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- **Python 3.10+** (tested on 3.13)
- **Java 11 or 17** (OpenJDK 17 recommended; verify with `java -version`)
- **Git**

### 2. Setup & Virtual Environment
```powershell
cd Smart-Commerce-AI/Smart_Commerce_AI
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3. Execute Complete Audited Pipeline
```powershell
# Step A: Validate dataset schema & integrity
python scripts/validate_dataset_integrity.py

# Step B: Run temporal preprocessing (zero data leakage)
python -m src.preprocess

# Step C: Run temporal leakage unit test
python -m tests.test_data_leakage

# Step D: Train ALS recommender, CTR models & run 3x scalability benchmark
python -m src.train

# Step E: Serve frequency-capped recommendations & export CSVs
python -m src.recommend
```

### 4. Launch Frontends
- **Interactive Storefront (Port 3000):**
  ```powershell
  python -m http.server 3000
  ```
  Open: **`http://localhost:3000`**
- **Streamlit Analytics Dashboard (Port 8501):**
  ```powershell
  streamlit run app.py --server.port 8501
  ```
  Open: **`http://localhost:8501`**

---

## 📜 Academic Compliance & Course Attribution

- **Course:** INT315 – Distributed and Cluster Computing  
- **Assessment:** Continuous Assessment 2 (CA2)  
- **Author / Student:** Utkarsh Pandey  
- **Institution:** School of Computer Science & Engineering, Lovely Professional University  
- **GitHub Repository:** [https://github.com/Utkarsh1454/Smart-Commerce-AI](https://github.com/Utkarsh1454/Smart-Commerce-AI)
