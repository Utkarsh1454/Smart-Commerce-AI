# Automated Test & Validation Report
**Course:** INT315 – Distributed and Cluster Computing (CA2)  
**Project:** Smart Commerce AI: A Distributed Personalized Product and Advertisement Recommendation System Using Apache Spark  
**Lead Author / Student:** Utkarsh Pandey  
**Institution:** Lovely Professional University  

---

## 1. Test Summary Overview

| Test Identifier | Category / Component | Target Script / Module | Expected Outcome | Actual Result | Status |
|---|---|---|---|---|:---:|
| **TST-01** | Dataset Schema & Integrity | `scripts/validate_dataset_integrity.py` | Verify 4 CSV files, schema columns, non-null IDs, valid price | All 4 datasets valid, 100% verified | **PASS** |
| **TST-02** | Temporal Data Leakage | `tests/test_data_leakage.py` | Verify future and concurrent events have ZERO effect on feature counts | Assertions pass, exit code 0 | **PASS** |
| **TST-03** | Catalyst Plan & Joins | `src/preprocess.py` | Verify Broadcast Hash Join on metadata dimensions via `.explain()` | `BroadcastHashJoin [BuildRight]` confirmed | **PASS** |
| **TST-04** | Distributed ALS Recommender | `src/train.py` | Train Implicit ALS, generate Top-10 recs, compare Popularity | NDCG=0.0197, Coverage=52.5% vs 0.52% | **PASS** |
| **TST-05** | Imbalanced CTR Prediction | `src/train.py` | Train LogReg & Random Forest with class weighting & tuned threshold | Log-Loss=0.5963, Pos F1=0.2433 | **PASS** |
| **TST-06** | 3x Statistical Benchmark | `src/train.py` | 3 replications across $p \in \{1, 2, 4, 8\}$ with std dev & Amdahl speedup | Mean runtimes and speedups exported | **PASS** |
| **TST-07** | Serving & Frequency Capping | `src/recommend.py` | Serve 1,200 users, cap max 2 ads/campaign, price > 0 filter | 12,000 recs + 5,972 ads exported to CSV | **PASS** |
| **TST-08** | Storefront Web Application | `index.html` (Port 3000) | Serve light-palette responsive UI, load dynamic recommendations | HTTP 200 OK, dynamic search & cart | **PASS** |
| **TST-09** | Streamlit Analytics Dashboard| `app.py` (Port 8501) | Visualize genuine empirical benchmark results & ROC/Coverage | HTTP 200 OK, dynamic metrics display | **PASS** |
| **TST-10** | Cloud Deployment Entrypoint | Root `app.py` / `vercel.json` | Comply with Vercel serverless HTTP entrypoint specifications | `class handler(BaseHTTPRequestHandler)` | **PASS** |

---

## 2. Test Execution Details

### TST-01: Dataset Schema & Integrity Validation
- **Command:** `python scripts/validate_dataset_integrity.py`
- **Output:**
  ```
  ======================================================================
  SMART COMMERCE AI - DATASET INTEGRITY & AUTHENTICITY VALIDATOR
  ======================================================================
  Dataset Provenance: Synthetic Demonstration Dataset (AliMama-compatible schema)
  Notice: This dataset models the schema of Alibaba Taobao ad displays for educational
          and cluster computing evaluation in Course INT315.
  ----------------------------------------------------------------------
  Checking: raw_sample.csv
    -> Records: 15,000 | Columns: ['user_id', 'time_stamp', 'adgroup_id', 'pid', 'noclk', 'clk']
    -> Verified: Non-empty, expected columns present, no null IDs, valid binary clk labels.
  Checking: ad_feature.csv
    -> Records: 400 | Columns: ['adgroup_id', 'cate_id', 'campaign_id', 'customer_id', 'brand_id', 'price']
    -> Verified: Non-empty, expected columns present, no null IDs, valid positive prices.
  Checking: user_profile.csv
    -> Records: 1,200 | Columns: ['user_id', 'cms_segid', 'cms_group_id', 'final_gender_code', 'age_level', 'pvalue_level', 'shopping_level', 'occupation', 'new_user_class_level']
    -> Verified: Non-empty, expected columns present, no null IDs.
  Checking: user_behavior.csv
    -> Records: 42,000 | Columns: ['user_id', 'item_id', 'cate_id', 'btag', 'time_stamp']
    -> Verified: Non-empty, expected columns present, no null IDs, valid btag actions.
  ======================================================================
  SUMMARY: 4/4 datasets passed all schema, type, and integrity checks!
  ======================================================================
  ```

---

### TST-02: Temporal Data Leakage Unit Test
- **Command:** `python -m tests.test_data_leakage`
- **Mechanism:** Injects synthetic past ($t=500$), concurrent ($t=1,000$), and future ($t=1,500$) user events relative to an ad impression at $t=1,000$.
- **Validation Criteria:** Feature engineering logic must enforce `behavior.time_stamp < raw_sample.time_stamp`. The resulting feature vector must show exactly 1 event (past) and 0 future/concurrent events.
- **Output:**
  ```
  Testing point-in-time temporal filtering logic...
  Simulated Ad Impression at time_stamp: 1000
  Injected Behaviors:
    - User 9999, Category 101, Action 'pv', Timestamp 500  (PAST)
    - User 9999, Category 101, Action 'buy', Timestamp 1000 (CONCURRENT)
    - User 9999, Category 101, Action 'cart', Timestamp 1500 (FUTURE)
  Computed Feature Row:
    - user_category_events: 1
    - user_category_buys: 0
    - user_category_carts: 0
  ✅ ASSERTION PASSED: Future and concurrent actions were strictly excluded! Zero data leakage!
  ```

---

### TST-03: Spark Catalyst Execution Plan & Broadcast Hash Join Inspection
- **Script:** `src/preprocess.py`
- **Verification Method:** Calls `df.explain()` on enriched impression DataFrame.
- **Result:**
  - `BroadcastHashJoin` confirmed for `ad_feature` dimension table.
  - `BroadcastHashJoin` confirmed for `user_profile` dimension table.
  - Verification: No `SortMergeJoin` or shuffle exchange occurs for dimension enrichment, eliminating inter-node network data movement.

---

### TST-04 & TST-05: Model Training, Baseline Comparisons, and Imbalance Handling
- **Command:** `python -m src.train`
- **Implicit ALS Evaluation:**
  - Trained on 31,978 implicit interactions in 4.692 seconds.
  - Tested on 7,980 held-out interactions:
    - ALS: NDCG@10 = `0.01967`, Precision@10 = `0.01468`, Catalog Coverage = `52.50%` (1,008 items).
    - Popularity Baseline: NDCG@10 = `0.00388`, Precision@10 = `0.00284`, Catalog Coverage = `0.52%` (10 items).
    - **Conclusion:** ALS demonstrates a 5.07× ranking gain and 101× catalog diversity gain over baseline.
- **CTR Classification:**
  - Class Imbalance: 13.0% positive click rate.
  - Inverse weights applied: $w_1 = 3.886, w_0 = 0.574$.
  - Optimal threshold selected on independent validation split: $\tau^* = 0.40$ (LogReg), $\tau^* = 0.30$ (Random Forest).
  - Test Set ($N=2,225$):
    - Random Forest: ROC-AUC = `0.58217`, Log-Loss = `0.59627`, Pos F1 = `0.24331` (TP=250, FP=1,535, TN=420, FN=20).
    - Logistic Regression: ROC-AUC = `0.58298`, Log-Loss = `0.68564`, Pos F1 = `0.22709` (TP=259, FP=1,752, TN=203, FN=11).
    - Prior Baseline: Accuracy = `0.87133`, Pos F1 = `0.00000`, TP = 0.

---

### TST-06: Controlled 3x Partition Scalability Replication
- **Script:** `src/train.py`
- **Output:**
  - $p=1$: $\bar{T}_1 = 0.0849$ s ($\sigma = 0.0222$ s) $\to$ Speedup: 1.000× (Efficiency: 100.0%)
  - $p=2$: $\bar{T}_2 = 0.1311$ s ($\sigma = 0.0378$ s) $\to$ Speedup: 0.648× (Efficiency: 32.4%)
  - $p=4$: $\bar{T}_4 = 0.1000$ s ($\sigma = 0.0126$ s) $\to$ Speedup: 0.849× (Efficiency: 21.2%)
  - $p=8$: $\bar{T}_8 = 0.1305$ s ($\sigma = 0.0111$ s) $\to$ Speedup: 0.651× (Efficiency: 8.1%)
  - **Conclusion:** Demonstrates scheduling overhead on fine-grained in-memory tasks under Amdahl's Law.

---

### TST-07: Serving Engine & Frequency Capping Validation
- **Command:** `python -m src.recommend`
- **Output:**
  - Product Recommendations: 12,000 rows generated for 1,200 distinct users.
  - Sponsored Ads: 5,972 ad placements generated.
  - Verified: No user receives more than 2 ads from the exact same advertising campaign.
  - Verified: All recommended products and ads have non-zero positive prices (`price > 0`).

---

### TST-08, TST-09, TST-10: Application & Deployment Health
- **Local Web Server (Port 3000):** Serving responsive light-palette UI (`index.html`) with interactive live search, dynamic category filtering, cart management, and model recommendation cards.
- **Streamlit Analytics Dashboard (Port 8501):** Interactive dashboard successfully loading and displaying all real metrics from `results/benchmark_summary.json`.
- **Vercel Serverless Architecture:** Root `app.py` exposes standard `class handler(BaseHTTPRequestHandler)` to guarantee seamless deployment on Vercel's Python serverless runtime.
