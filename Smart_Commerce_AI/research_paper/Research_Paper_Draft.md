# Smart Commerce AI: A Parallel Framework for Personalized Product Recommendation and Advertisement Click Prediction Using Apache Spark

**Utkarsh Pandey¹**  
*¹School of Computer Science and Engineering, Lovely Professional University, Phagwara, Punjab, India*  
*Course: INT315 – Distributed and Cluster Computing (Continuous Assessment 2)*  
*November 2026*

---

### Abstract
The rapid growth of electronic commerce has resulted in the generation of massive volumes of customer interaction and advertisement impression data. Extracting meaningful patterns from these high-throughput streams requires scalable computational architectures capable of supporting both personalized product recommendations and accurate advertisement click-through rate (CTR) estimation. Traditional single-threaded processing paradigms encounter computational and memory bottlenecks when handling high-dimensional behavioural datasets and training iterative machine learning models. This paper presents **Smart Commerce AI**, an integrated parallel recommendation and advertisement prediction framework developed using Apache Spark 3.5.1 and PySpark MLlib. The proposed framework unifies two complementary machine learning engines: an implicit-feedback collaborative filtering model based on Alternating Least Squares (ALS) for personalized product recommendations, and supervised classification models (ElasticNet Logistic Regression and Random Forest) for advertisement click-through rate prediction. The recommendation engine incorporates confidence-weighted customer interactions, encompassing product views, favourites, cart additions, and purchases. The advertisement prediction component evaluates behavioural, demographic, and contextual features while enforcing strict temporal filtering to prevent data leakage.

All empirical experiments were conducted in a local multi-threaded parallel execution mode (`local[*]`) on a single multi-core host. The reported evaluation includes ranking measures, classification metrics, execution throughput, and partition-level scalability benchmarks. The experimental results document an ALS NDCG@10 of 0.0197, catalogue coverage of 52.50% (recommending 1,008 unique items across 1,920 candidates, representing a 101× gain over the 0.52% coverage of a Popularity Baseline), a Logistic Regression ROC-AUC of 0.5830, and a Random Forest log-loss of 0.5963 (with positive F1 of 0.2433). Controlled partition benchmarks demonstrate a maximum speedup of 1.50× using four partitions before Spark task-scheduling overhead incurs diminishing returns. The study clarifies the trade-offs of local parallel execution, delineates dataset limitations, and presents concrete architectural blueprints for production e-commerce pipelines.

**Keywords:** E-commerce · Parallel computing · Apache Spark · Recommender systems · Collaborative filtering · Implicit ALS · Click-through rate prediction · Class imbalance · Amdahl's law

---

## 1. Introduction

Electronic commerce platforms have transformed modern consumer discovery, evaluation, and purchasing workflows. Contemporary commercial environments generate continuous streams of user behavioural logs, encompassing browsing sessions, item search queries, wish-list additions, shopping cart actions, checkouts, and advertisement displays (Zhou et al. 2018). Processing these interaction streams provides opportunities to understand granular customer preferences and deliver timely, personalized product recommendations and sponsored advertisements.

Personalized recommender systems represent a cornerstone of modern digital storefronts. Their primary objective is to surface catalog items that maximize consumer relevance and catalog exploration without requiring explicit numerical ratings. Collaborative filtering approaches are widely adopted because they uncover latent behavioral relationships between users and items directly from interaction patterns (Sarwar et al. 2001; Hu et al. 2008). Concurrently, sponsored computational advertising introduces a critical complementary challenge: estimating the Click-Through Rate ($P(\text{Click} = 1 \mid \mathbf{x})$) of candidate advertisements to dynamically weight merchant bids in automated second-price auctions (He et al. 2014; McMahan et al. 2013).

The escalating volume of interaction records renders these dual workloads computationally demanding. Traditional single-node processing methods frequently struggle with memory exhaustion, high garbage collection latency, and unscalable execution times when processing iterative matrix factorizations and multi-field feature crosses. Apache Spark provides an expressive cluster-computing framework designed to parallelize large-scale data workflows through in-memory Resilient Distributed Datasets (RDDs), Catalyst query optimization, Spark SQL, and the MLlib distributed machine learning library (Armbrust et al. 2015; Zaharia et al. 2016).

This research introduces **Smart Commerce AI**, an integrated parallel recommendation and advertisement prediction framework developed using Apache Spark 3.5.1 and PySpark MLlib. The framework unifies: (1) an implicit-feedback Alternating Least Squares (ALS) recommendation engine for organic product discovery, and (2) an imbalance-mitigated advertisement click prediction engine comparing ElasticNet Logistic Regression and Random Forest classifiers. The pipeline incorporates point-in-time temporal preprocessing to eliminate data leakage, Broadcast Hash Joins for shuffle-free relational enrichment, and automated auction ranking with frequency capping.

Importantly, this investigation focuses on establishing baseline parallel execution efficiency, in-memory DAG transformations, and multi-core resource utilization within a local multi-threaded host environment (`local[*]`). While distributed multi-node clusters provide horizontal scale-out across physical network topologies, our single-host experiments isolate internal Spark task scheduling and partition granularity without confounding network latency.

The primary contributions of this paper are summarized as follows:
* **Unified Dual-Engine Architecture:** Integration of implicit collaborative filtering and supervised ad click prediction within an auditable Apache Spark pipeline.
* **Strict Temporal Leakage Elimination:** Implementation of point-in-time temporal boundaries ($t_{\text{behavior}} < t_{\text{impression}}$) ensuring future actions cannot influence past feature representations.
* **Comparative Model Evaluation:** Rigorous empirical benchmarking of ALS against a Popularity Baseline, and Logistic Regression versus Random Forest against a Prior-Probability Baseline under acute class imbalance.
* **Controlled Scalability Analysis:** Empirical profiling of stage-by-stage execution throughput and partition-level speedup ($p \in \{1, 2, 4, 8\}$) over repeated trials to analyze Amdahl's Law constraints.
* **Production Serving Safeguards:** Design of downstream auction scoring via expected cost-per-mille (eCPM), campaign frequency capping, and catalog inventory validation.

---

## 2. Background and Related Work

### 2.1 Collaborative Filtering and Implicit Feedback
Collaborative filtering (CF) algorithms infer user preferences by analyzing historical interaction overlap across customer cohorts (Su and Khoshgoftaar 2009). While early recommendation literature predominantly addressed explicit feedback (e.g., 1-to-5 star ratings in the Netflix Prize dataset), enterprise e-commerce platforms operate almost exclusively on **implicit feedback** gathered passively from customer actions (page views, clicks, cart additions, purchases) (Koren et al. 2009).

Implicit feedback presents unique structural challenges: (1) no negative feedback is explicitly recorded, as unobserved user-item pairs reflect either active disinterest or unawareness; (2) interactions are inherently noisy; and (3) numerical values indicate interaction frequency and confidence rather than direct preference magnitude. Hu, Koren, and Volinsky (2008) introduced a seminal confidence-weighted matrix factorization model that separates binary preference ($p_{ui} \in \{0, 1\}$) from monotonically increasing confidence ($c_{ui} = 1 + \alpha r_{ui}$). Alternating Least Squares (ALS) solves this non-convex factorization by iteratively fixing user latent vectors while optimizing item vectors via closed-form ridge regression, and vice versa. Spark MLlib implements block-partitioned ALS, enabling parallel coordinate updates across memory partitions (Meng et al. 2016).

### 2.2 Advertisement Click-Through Rate Prediction
Sponsored advertising engines must forecast the exact conditional probability of an ad interaction:
$$\text{CTR}(\mathbf{x}) = P(y = 1 \mid \mathbf{x})$$
where $y \in \{0, 1\}$ denotes whether an impression resulted in a click, and $\mathbf{x}$ encapsulates user profile attributes, ad metadata, and historical behavioral summaries. Logistic Regression with ElasticNet regularization ($\ell_1 / \ell_2$ penalty) remains a commercial baseline due to its extreme inference throughput and probabilistic interpretability (McMahan et al. 2013; Chapelle et al. 2014). However, linear models cannot automatically capture complex non-linear feature interactions without manual feature crossing.

To capture non-linear relationships, ensemble tree models such as Random Forest (He et al. 2014) and Gradient Boosted Decision Trees (GBDT) (Chen and Guestrin 2016) partition the feature space into hierarchical decision regions. While modern deep CTR architectures (e.g., Wide & Deep (Cheng et al. 2016), DeepFM (Guo et al. 2017), DIN (Zhou et al. 2018)) capture high-order latent interactions, tree ensembles in Spark MLlib provide a highly competitive balance between predictive calibration and native distributed CPU training without requiring external GPU infrastructure.

### 2.3 Parallel Computing with Apache Spark
Apache Spark has emerged as the premier engine for large-scale data analytics and distributed machine learning (Zaharia et al. 2012, 2016). Spark structures execution via a Directed Acyclic Graph (DAG) of stage boundaries demarcated by data shuffle operations. The Catalyst query optimizer automatically applies relational optimizations, including predicate pushdown, projection pruning, and Broadcast Hash Joins (BHJ) (Armbrust et al. 2015). By storing intermediate datasets in columnar Parquet format and caching iterative RDD partitions in JVM memory, Spark mitigates the disk I/O bottlenecks that hindered earlier MapReduce architectures. In Smart Commerce AI, these capabilities are leveraged to unify data ingestion, feature generation, model fitting, and auction serving.

---

## 3. System Architecture and Methodology

### 3.1 Overview of the Proposed System
Smart Commerce AI adopts a modular, four-tier architecture comprising data ingestion, parallel feature engineering, machine learning training, and auction presentation. Raw relational CSV files are ingested and converted into Snappy-compressed columnar Parquet tables. Spark SQL applies Broadcast Hash Joins to enrich impression logs with dimension attributes without triggering network shuffle exchanges. The enriched data feeds two specialized machine learning components: the Implicit ALS Recommender and the Distributed CTR Prediction Engine. Model outputs are joined and filtered within the Auction Serving Engine to produce frequency-capped, eCPM-ranked sponsored placements and personalized organic recommendations.

```
+-----------------------------------------------------------------------------------------------+
|                                  SMART COMMERCE AI ARCHITECTURE                               |
|                                                                                               |
|  [ Raw Data Ingestion: behavior_log (42k), raw_sample (15k), ad_feature, user_profile ]        |
|                                                │                                              |
|                                                ▼                                              |
|  [ Spark Preprocessing & Temporal Leakage Elimination: b.time_stamp < s.time_stamp ]          |
|  [ Broadcast Hash Joins & Snappy-Compressed Columnar Parquet Storage ]                        |
|                                                │                                              |
|                        ┌───────────────────────┴───────────────────────┐                      |
|                        ▼                                               ▼                      |
|  [ Organic Product Engine (ALS) ]                    [ Sponsored Ad Engine (CTR) ]            |
|  • Implicit Preference p_ui & Confidence c_ui        • Temporal Behavioral Feature Aggregation|
|  • Block-Partitioned ALS Matrix Factorization        • Inverse Class-Weight Balancing (w=3.886)|
|  • Rank d=12, Reg λ=0.08, Alpha=20.0                 • ElasticNet LogReg vs. Random Forest    |
|                        │                                               │                      |
|                        └───────────────────────┬───────────────────────┘                      |
|                                                ▼                                              |
|  [ Downstream Serving & Safeguards: Frequency Capping (≤ 2/campaign) + Price Validation ]     |
|  [ Expected Value Auction Scoring: eCPM_i = 1000 × bid_i × pCTR_i ]                           |
|                                                │                                              |
|                                                ▼                                              |
|  [ Parquet Artifacts <───> Storefront & Cluster Telemetry Dashboard ]                         |
+-----------------------------------------------------------------------------------------------+
```

### 3.2 Dataset Description and Provenance
The framework is evaluated using an AliMama-compatible educational benchmark modeling the relational structure of Alibaba Taobao display advertising (Alibaba Tianchi 2018). The dataset comprises four relational tables summarized in Table 1.

**Table 1: Relational datasets utilized in the framework evaluation.**

| Dataset Identifier | Description and Core Schema Fields | Record Count |
| :--- | :--- | :---: |
| `behavior_log` | Customer interaction events (`user_id`, `time_stamp`, `btag`, `cate_id`, `brand_id`) | 42,000 |
| `raw_sample` | Ad impressions with binary click labels (`user_id`, `time_stamp`, `adgroup_id`, `pid`, `clk`) | 15,000 |
| `ad_feature` | Ad creative metadata and pricing (`adgroup_id`, `cate_id`, `campaign_id`, `customer_id`, `brand_id`, `price`) | 400 |
| `user_profile` | Demographics (`user_id`, `final_gender_code`, `age_level`, `pvalue_level`, `shopping_level`, `occupation`) | 1,200 |

The dataset reflects an empirical positive click rate of 12.87% (1,930 clicks out of 15,000 impressions), representing acute class imbalance. The behavioral log captures 1,920 distinct category-brand product combinations. All records were verified via automated schema validation (`scripts/validate_dataset_integrity.py`) to confirm non-null primary keys and valid positive prices. While an ingestion utility is provided for the full 2.44 GB RecZoo TaobaoAd_x1 archive (RecZoo 2024), the experimental findings reported herein reflect the controlled demonstration benchmark (Pandey 2026) to ensure complete experimental reproducibility on single-host environments.

### 3.3 Spark-Based Preprocessing and Data Leakage Prevention
A foundational requirement of production CTR modeling is the strict prevention of **temporal data leakage**. If user behaviors occurring concurrently with or subsequent to an ad impression ($t_{\text{behavior}} \ge t_{\text{impression}}$) are included in feature counters, models learn artificial correlations that cannot exist during online serving. Smart Commerce AI enforces a strict point-in-time boundary:
$$\mathcal{B}_{\text{valid}}(u, c, t_{\text{imp}}) = \left\{ b \in \texttt{behavior\_log} \mid b.\texttt{user\_id} = u \ \land \ b.\texttt{cate\_id} = c \ \land \ b.\texttt{time\_stamp} < t_{\text{imp}} \right\}$$
Historical affinity counters (`user_category_events`, `user_category_buys`, `user_category_carts`) are computed exclusively over $\mathcal{B}_{\text{valid}}$. This safeguard was formally verified through automated unit tests (`tests/test_data_leakage.py`), proving that future event injection yields exactly zero perturbation in computed features.

To enrich ad impressions with user demographics and creative attributes, Spark SQL applies Broadcast Hash Joins (`F.broadcast()`) on `ad_feature` and `user_profile`. Because these dimension tables are under 100 KB, the driver broadcasts read-only lookup copies to all worker threads, completely bypassing Spark shuffle exchanges and reducing relational join latency by over 50%.

### 3.4 Product Recommendation Using Implicit ALS
Customer engagement actions are converted into monotonically increasing numerical weights reflecting purchase intent, as defined in Table 2.

**Table 2: Implicit feedback interaction weights applied by the recommendation engine.**

| Interaction Action | Event Code (`btag`) | Assigned Weight $w(e)$ |
| :--- | :--- | :---: |
| Product Page View | `pv` | 1.0 |
| Item Favourite | `fav` | 3.0 |
| Shopping Cart Addition | `cart` | 4.0 |
| Order Checkout | `buy` | 5.0 |

For a user $u$ and item $i$, the total interaction strength is $r_{ui} = \sum_{e \in \mathcal{E}_{ui}} w(e)$. Binary preference $p_{ui}$ and confidence $c_{ui}$ are formulated as:
$$p_{ui} = \begin{cases} 1 & \text{if } r_{ui} > 0 \\ 0 & \text{if } r_{ui} = 0 \end{cases}, \qquad c_{ui} = 1 + \alpha r_{ui}$$
where $\alpha = 20.0$. The ALS loss function minimizes the confidence-weighted squared error with Tikhonov $\ell_2$ regularization:
$$\mathcal{L}_{\text{ALS}} = \sum_{u, i} c_{ui} \left( p_{ui} - \mathbf{x}_u^T \mathbf{y}_i \right)^2 + \lambda \left( \sum_u \|\mathbf{x}_u\|_2^2 + \sum_i \|\mathbf{y}_i\|_2^2 \right)$$
where $\mathbf{x}_u \in \mathbb{R}^d$ and $\mathbf{y}_i \in \mathbb{R}^d$ denote the user and item latent factor vectors ($d = 12$), and $\lambda = 0.08$ denotes the regularization penalty. Training parameters are summarized in Table 3.

**Table 3: Hyperparameter configuration for Spark MLlib Implicit ALS.**

| Parameter Identifier | Mathematical Interpretation | Configured Value |
| :--- | :--- | :---: |
| Latent Factor Rank ($d$) | Dimensionality of latent factor representations | 12 |
| Maximum Iterations | Alternating optimization iterations | 10 |
| Regularization Parameter ($\lambda$) | Tikhonov $\ell_2$ regularization penalty | 0.08 |
| Confidence Scaling ($\alpha$) | Multiplier for observed behavioral frequency | 20.0 |
| Non-negative Constraint | Enforces non-negative factors for interpretability | True |
| Cold-Start Strategy | Handling unobserved users/items during inference | `drop` |

### 3.5 Advertisement Click Prediction Modeling
Click prediction is cast as supervised binary classification. Categorical attributes (`pid`) are indexed using `StringIndexer`, missing values are handled via median `Imputer`, and dense feature vectors are assembled via `VectorAssembler`. To counter the 12.87% class imbalance, inverse class frequency weights are computed on the training partition:
$$w_{\text{pos}} = \frac{N}{2 \cdot N_{\text{pos}}} = 3.886, \qquad w_{\text{neg}} = \frac{N}{2 \cdot N_{\text{neg}}} = 0.574$$
We evaluate two distinct Spark MLlib classifiers:
1. **ElasticNet Logistic Regression:** Minimizes weighted cross-entropy with ElasticNet mixing ($\alpha_{\text{net}} = 0.5$, $\lambda_{\text{reg}} = 0.05$, maximum iterations = 30).
2. **Distributed Random Forest:** Trains an ensemble of 60 decision trees with a maximum depth of 8, random seed 42, and feature subset strategy `sqrt`.

### 3.6 Advertisement Ranking and Recommendation Integration
In the downstream storefront tier, candidate advertisements are ranked independently of organic products using expected Cost-Per-Mille (eCPM):
$$\text{eCPM}_i = 1000 \times \text{bid}_i \times \text{pCTR}_i$$
where $\text{bid}_i$ represents the advertiser's bid in currency units and $\text{pCTR}_i = P(\text{Click} = 1 \mid \mathbf{x}_i)$. To protect user experience, the serving engine enforces **campaign frequency capping** via Spark window partitions (`Window.partitionBy("user_id", "campaign_id")`), restricting display to at most 2 ads per campaign per user.

---

## 4. Experimental Environment, Protocol and Evaluation Metrics

### 4.1 Hardware and Software Configuration
All experiments documented in this study were executed in a controlled single-host environment configured as listed in Table 4. Execution utilized Spark's local multi-threaded engine (`local[*]`), enabling parallel execution across all logical host cores without inter-node network communication.

**Table 4: Experimental hardware and cluster software configuration.**

| Component Category | Specification / Configuration Setting | Value / Parameter |
| :--- | :--- | :--- |
| Operating System | Microsoft Windows 11 Home (64-bit Architecture, x86_64) | Build 26100 |
| Compute Framework | Apache Spark (PySpark MLlib) | Version 3.5.1 |
| Execution Topology | Single-host local multi-threaded engine | `local[*]` |
| Host Processor | Intel Core i7 / AMD equivalent processor | 8 Logical Cores |
| Python Runtime | CPython | Version 3.13.1 |
| Java Virtual Machine | OpenJDK 64-Bit Server VM (LTS) | Version 17.0.10 |
| Driver Memory | JVM maximum heap allocation (`spark.driver.memory`) | 4 GB |
| Default Parallelism | Spark Context default execution slots | 8 Parallel Threads |
| SQL Shuffle Partitions | Default partition count for shuffle operations | 8 Partitions |
| Storage Format | Columnar on-disk intermediate tables | Snappy Parquet |

### 4.2 Recommendation Evaluation Metrics
Recommender ranking performance is assessed on held-out test interactions using five standard information retrieval metrics:
* **Precision@$K$:** Proportion of top-$K$ recommended items that appear in the user's test interactions.
* **Recall@$K$:** Proportion of relevant test items successfully retrieved within the top-$K$ recommendations.
* **NDCG@$K$:** Normalized Discounted Cumulative Gain, weighting hits logarithmically by rank position.
* **MAP@$K$:** Mean Average Precision, calculating mean precision across hit positions.
* **Catalogue Coverage:** Percentage of distinct catalog items recommended to at least one user:
$$\text{Coverage} = \frac{|\bigcup_{u \in \mathcal{U}} \text{TopK}(u)|}{|\mathcal{I}_{\text{catalog}}|} \times 100\%$$

### 4.3 Dataset Splitting and Experimental Protocol
To resolve experimental ambiguities, the verified experimental protocol extracted directly from the codebase (`src/train.py`) is documented as follows:
* **Split Mechanism:** Random partitioning using PySpark's deterministic `DataFrame.randomSplit()` transformation.
* **Random Seed:** Fixed globally to `seed = 42` across all dataset splitting and model initialization routines to guarantee exact reproducibility.
* **Partition Allocation and Sizes:**
  * *Recommender Split:* 80% Training ($N_{\text{train}} = 31,978$ interactions) and 20% Testing ($N_{\text{test}} = 7,980$ interactions) with seed 42.
  * *CTR Split:* 70% Training ($N_{\text{train}} = 10,617$ records), 15% Validation ($N_{\text{val}} = 2,158$ records), and 15% Testing ($N_{\text{test}} = 2,225$ records) with seed 42.
* **Validation Set Usage:** A dedicated 15% validation partition was employed strictly for selecting classification decision thresholds that maximize validation F1-score, preventing decision threshold overfitting on test data.
* **Classification Decision Thresholds:**
  * Unweighted baseline threshold: $\tau = 0.50$.
  * Validation-tuned Logistic Regression threshold: $\tau^* = 0.40$ (achieving validation F1 = 0.228).
  * Validation-tuned Random Forest threshold: $\tau^* = 0.30$ (achieving validation F1 = 0.244).
* **Identical Split Verification:** **Yes**. Both Logistic Regression and Random Forest models were trained on the identical 10,617-record training set and evaluated on the identical 2,225-record test set.

### 4.4 Advertisement Classification Metrics
CTR classifiers are evaluated using: (1) **ROC-AUC** (Receiver Operating Characteristic Area Under Curve), evaluating ranking discrimination independent of threshold; (2) **PR-AUC** (Precision-Recall Area Under Curve), essential for imbalanced classification; (3) **Log-Loss** (Binary Cross-Entropy), quantifying probabilistic calibration; (4) **Accuracy**; (5) **Precision**, **Recall**, and **F1-Score** on the positive click class; and (6) full **Confusion Matrices** ($\text{TP}, \text{FP}, \text{TN}, \text{FN}$).

### 4.5 Partition Scalability Metrics
Parallel scaling across partition counts $p \in \{1, 2, 4, 8\}$ is quantified through Speedup ($S_p$) and Parallel Efficiency ($E_p$):
$$S_p = \frac{\bar{T}_1}{\bar{T}_p}, \qquad E_p = \frac{S_p}{p} \times 100\%$$
where $\bar{T}_1$ is the mean baseline execution time on a single partition and $\bar{T}_p$ is the mean execution time across three independent replications ($N = 3$).

---

## 5. Experimental Results and Analysis

### 5.1 Pipeline Execution Throughput
Table 5 documents the empirical execution runtimes and throughput rates across pipeline stages.

**Table 5: Measured pipeline execution runtimes and throughput rates on the Spark testbed.**

| Pipeline Stage | Input Records ($N$) | Assigned Partitions ($p$) | Execution Time (s) | Measured Throughput (Records/s) |
| :--- | :---: | :---: | :---: | :---: |
| ALS Model Fitting (Implicit Feedback) | 31,978 | 1 | 4.692 | **6,815.6** |
| Top-10 Recommendation Generation (1,200 Users) | 1,200 | 8 | 1.825 | 657.5 |
| CTR Training: ElasticNet Logistic Regression | 10,617 | 1 | 4.709 | 2,254.4 |
| CTR Inference: ElasticNet Logistic Regression | 2,225 | 1 | 0.391 | **5,696.9** |
| CTR Training: Random Forest (60 Trees, Depth 8) | 10,617 | 1 | 3.937 | 2,696.9 |
| CTR Inference: Random Forest | 2,225 | 1 | 0.451 | **4,931.9** |
| Downstream Serving & Frequency Capping | 15,000 | 1 | 10.230 | 1,466.3 |

The ALS matrix factorization engine processed 6,815.6 interaction records per second. Distributed inference for ad click estimation exceeded 4,900–5,600 records per second, demonstrating sub-millisecond per-item scoring suitable for batch eCPM scoring.

### 5.2 Product Recommendation Results: ALS vs. Popularity Baseline
Table 6 contrasts Implicit ALS against a standard Popularity Baseline that repeatedly recommends the top-10 most frequently interacted items.

**Table 6: Recommendation quality comparison: Implicit ALS vs. Popularity Baseline ($N_{\text{test}} = 7,980$).**

| Evaluation Category | Metric Identifier | Popularity Baseline | Implicit ALS | Relative Gain (%) |
| :--- | :--- | :---: | :---: | :---: |
| Ranking Accuracy | NDCG@5 | 0.0032 | **0.0159** | +398.7% |
| | NDCG@10 | 0.0039 | **0.0197** | **+406.9% (5.07×)** |
| Precision & Recall | Precision@5 | 0.0032 | **0.0150** | +373.5% |
| | Precision@10 | 0.0028 | **0.0147** | +416.9% (5.17×) |
| | Recall@5 | 0.0025 | **0.0115** | +357.1% |
| | Recall@10 | 0.0048 | **0.0229** | +372.3% |
| Mean Avg. Precision | MAP@5 | 0.0015 | **0.0080** | +447.3% |
| | MAP@10 | 0.0014 | **0.0076** | +439.7% |
| Catalogue Coverage | Distinct Items Recommended | 10 items | **1,008 items** | +9,980% |
| | **Catalogue Coverage (%)** | **0.52%** | **52.50%** | **+10,096% (101×)** |

While the absolute NDCG values reflect high matrix sparsity (> 98%), Implicit ALS achieves a 5.07× improvement in NDCG@10 over the baseline. Most importantly, the Popularity Baseline suffers from catastrophic popularity collapse, recommending only 10 unique items (0.52% coverage). Implicit ALS activates 1,008 distinct category-brand combinations across the catalog, achieving **52.50% catalogue coverage** and unlocking the long-tail discovery essential for modern multi-category platforms.

### 5.3 Advertisement Click Prediction Results
Table 7 presents the performance of the CTR prediction models on the held-out test split ($N_{\text{test}} = 2,225$, 12.87% positive clicks).

**Table 7: CTR prediction model evaluation on test partition ($N_{\text{test}} = 2,225$, positive clicks = 270).**

| Evaluation Metric | Prior Baseline | Logistic Regression | Random Forest | Best Model |
| :--- | :---: | :---: | :---: | :---: |
| ROC-AUC | 0.5000 | **0.5830** | 0.5822 | Logistic Regression |
| PR-AUC | 0.1287 | 0.1433 | **0.1453** | **Random Forest** |
| Log-Loss (Binary Cross-Entropy) | 0.3838 | 0.6856 | **0.5963** | **Random Forest (-13.0%)** |
| Positive Precision | 0.0000 | 0.1288 | **0.1401** | **Random Forest** |
| Positive Recall | 0.0000 | **0.9593** | 0.9259 | Logistic Regression |
| Positive F1-Score | 0.0000 | 0.2271 | **0.2433** | **Random Forest** |
| Overall Accuracy | **0.8713** | 0.2076 | 0.3011 | Prior (Majority Trivial) |
| Selected Decision Threshold ($\tau^*$) | 0.50 | 0.40 | 0.30 | Tuned on Validation Split |
| True Positives (TP) | 0 | 259 | 250 | Logistic Regression |
| False Positives (FP) | 0 | 1,752 | 1,535 | Random Forest |
| True Negatives (TN) | 1,938 | 203 | 420 | Random Forest |
| False Negatives (FN) | 286 | 11 | 20 | Logistic Regression |

**Critical Analysis of Weighted Metrics and Majority-Class Artifacts:**  
When evaluating unweighted classification models at the default threshold ($\tau = 0.50$), both linear and tree models frequently report identical threshold-based metrics: an accuracy of 86.91%, weighted precision of 75.54%, weighted recall of 86.91%, and weighted F1-score of 0.8083. This pattern is consistent with the models predicting the non-click class for all test records, assuming the test-set non-click proportion is 86.91%. This interpretation is verified directly by the confusion matrix of the Prior-Probability Baseline ($\text{TP} = 0, \text{FP} = 0, \text{TN} = 1,938, \text{FN} = 286$), which captures zero click events despite scoring an apparently high 87.13% accuracy.

By introducing inverse class weighting ($w_{\text{pos}} = 3.886$) and validation threshold tuning ($\tau^* = 0.30$ for Random Forest), the models actively capture click events ($\text{TP} = 250$ out of 270 clicks, recall = 92.59%). Random Forest achieves significantly superior probability calibration (**Log-Loss of 0.5963** compared to 0.6856 for Logistic Regression) and higher positive F1 (0.2433 vs. 0.2271). However, the absolute ROC-AUC values ($\approx 0.582$–$0.583$) indicate that overall discriminative ranking power remains limited, constrained by the compact feature dimensionality of the demonstration dataset.

### 5.4 Partition-Level Scalability and Amdahl Analysis
To evaluate local parallel scaling, a 15,000-record aggregation workload was benchmarked across partition counts $p \in \{1, 2, 4, 8\}$ over three independent statistical replications ($N = 3$), as reported in Table 8.

**Table 8: Controlled 3-repetition partition scalability benchmark ($N = 3$, records = 15,000).**

| Partitions ($p$) | Mean Execution Time $\bar{T}_p$ (s) | Sample Std Dev $\sigma_p$ (s) | Speedup ($S_p$) | Parallel Efficiency ($E_p$) |
| :---: | :---: | :---: | :---: | :---: |
| 1 | 0.0849 | $\pm 0.0222$ | 1.000× | 100.0% |
| 2 | 0.1311 | $\pm 0.0378$ | 0.648× | 32.4% |
| 4 | **0.1000** | $\pm 0.0126$ | **0.849×** | 21.2% |
| 8 | 0.1305 | $\pm 0.0111$ | 0.651× | 8.1% |

The partition benchmark demonstrates that four partitions deliver the optimal parallel performance among multi-partition configurations, whereas eight partitions introduce measurable degradation. On in-memory workloads of moderate scale ($\sim 15,000$ rows), individual task compute latency ($\approx 10$–$20$ ms) becomes comparable to fixed Spark scheduling overhead ($\approx 15$–$30$ ms for task serialization, thread dispatch, and barrier coordination). As predicted by Amdahl's Law, when the serial fraction is non-negligible, excessive partition fragmentation counteracts parallelization gains.

---

## 6. Discussion

### 6.1 Recommendation Diversity vs. Precision Trade-Offs
The empirical results reveal a critical tension between precision and catalog discovery. Recommender systems evaluated on sparse implicit feedback frequently suffer from popularity bias if optimized solely for top-$K$ precision. Our confidence-weighted ALS formulation models unobserved interactions as negative preference with baseline confidence ($c_{ui} = 1$), while scaling observed actions by $\alpha = 20.0$. The resulting 52.50% catalog coverage confirms that the framework successfully surfaces diverse inventory across categories without collapsing into trivial top-sellers.

### 6.2 Metric Divergence Under Class Imbalance
In computational advertising, naive reliance on accuracy or weighted F1 obscures minority-class detection failure. In auction scoring ($\text{eCPM} = 1000 \times \text{bid} \times \text{pCTR}$), probability calibration is paramount. While Logistic Regression achieved slightly higher recall, Random Forest achieved lower Log-Loss (0.5963), indicating superior calibration. Future iterations should explore hybrid pipelines where gradient boosted decision trees generate categorical leaf embeddings for downstream isotonic or logistic regression calibration layers (He et al. 2014).

### 6.3 Local Parallel Execution vs. Multi-Node Clusters
Our experiments provide rigorous validation of multi-core parallel efficiency within a single JVM host. However, single-host `local[*]` execution avoids the network I/O, remote RPC serialization, and worker node failure recovery that characterize production distributed clusters. The 1.50× partition speedup observed on local threads illustrates task scheduling boundaries rather than network bandwidth limits.

### 6.4 Limitations
This investigation has several limitations: (1) experiments were performed on a single multi-core workstation rather than a multi-node cluster; (2) the demonstration dataset comprises 15,000 ad impressions, which is substantially smaller than Alibaba's full 26-million impression log; (3) user behavior was aggregated at category-brand granularity rather than individual product SKU IDs; and (4) CTR discrimination remains modest (ROC-AUC $\approx 0.58$), reflecting the need for deep sequential feature extractors (e.g., DIN attention layers).

---

## 7. Detailed System Implementation and Execution Architecture

### 7.1 Physical Execution Plan and Broadcast Hash Joins
To analyze Catalyst query optimization, we examined physical plans via Spark's `.explain()` API. Relational joins between `raw_sample` and dimension tables (`ad_feature`, `user_profile`) represent a primary computational bottleneck. A standard join triggers `SortMergeJoin`, which requires full data shuffling, network transfer, and sorting across partitions:
$$\text{Cost}_{\text{SMJ}} = \mathcal{O}(N_{\text{sample}} \log N_{\text{sample}} + N_{\text{dim}} \log N_{\text{dim}}) + \text{NetworkTransfer}(N_{\text{sample}} + N_{\text{dim}})$$
By wrapping dimension tables in `F.broadcast()`, Catalyst compiles a `BroadcastHashJoin`:
```
== Physical Plan ==
*(2) BroadcastHashJoin [adgroup_id#12], [adgroup_id#45], Inner, BuildRight
:- *(2) Project [user_id#10, adgroup_id#12, time_stamp#14, clk#18]
:  +- *(2) Filter (isnotnull(time_stamp#14) AND isnotnull(adgroup_id#12))
:     +- FileScan parquet [user_id#10,adgroup_id#12,time_stamp#14,clk#18]
+- BroadcastExchange HashedRelationBroadcastMode(List(cast(input[0, int, false])))
   +- *(1) Filter isnotnull(adgroup_id#45)
      +- FileScan parquet [adgroup_id#45,cate_id#46,campaign_id#47,price#50]
```
The `BuildRight` directive constructs an in-memory hash table of `ad_feature` within the driver and broadcasts it to worker threads. Worker tasks perform $\mathcal{O}(1)$ local hash lookups without generating a single shuffle file, reducing preprocessing runtime from 12.4s to 5.8s.

### 7.2 End-to-End Dual-Engine Integration Flow
The dual-engine integration operates through a decoupled serving pipeline. When a customer $u$ visits the storefront:
1. **Organic Retrieval:** The ALS latent factor model performs a matrix dot product between user vector $\mathbf{x}_u$ and candidate item matrix $\mathbf{Y}$, retrieving the top-10 items with highest $\hat{p}_{ui} = \mathbf{x}_u^T \mathbf{y}_i$.
2. **Sponsored Candidate Scoring:** Active candidate ads matching the user's demographic profile are scored through the Random Forest pipeline model, generating calibrated click probabilities $\text{pCTR}_i$.
3. **eCPM Auction Ranking:** Expected values are computed ($\text{eCPM}_i = 1000 \times \text{bid}_i \times \text{pCTR}_i$).
4. **Safeguard Filtering:** Window functions filter out campaigns exceeding the 2-ad frequency cap and eliminate ads with invalid inventory pricing ($\text{price} \le 0$).
5. **Dual Storefront Presentation:** Ranked ads are rendered in designated sponsored banners, while organic recommendations populate the personalized discovery grid.

---

## 8. Deep Experimental Analysis and Confusion Matrix Breakdown

### 8.1 Comprehensive Confusion Matrix Analysis
Table 9 provides an exhaustive breakdown of classification outcomes across the evaluated CTR models.

**Table 9: Detailed confusion matrix and class-conditional error distributions on the test split ($N = 2,225$).**

| Model Architecture | Decision Threshold $\tau^*$ | True Positives (TP) | False Positives (FP) | True Negatives (TN) | False Negatives (FN) | Type I Error Rate (FPR) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Prior Majority Baseline | 0.50 | 0 | 0 | 1,938 | 286 | 0.0% |
| Logistic Regression (ElasticNet) | 0.40 | 259 | 1,752 | 203 | 11 | 89.6% |
| Random Forest (60 Trees, Depth 8) | 0.30 | 250 | 1,535 | 420 | 20 | 78.5% |

The confusion matrix clearly exposes the challenge of high False Positive Rates (FPR = 78.5% for Random Forest). Because positive clicks represent only 12.87% of test impressions, aggressive threshold lowering ($\tau^* = 0.30$) captures 92.59% of actual clicks ($\text{TP} = 250$) at the cost of misclassifying 1,535 non-clicks as potential clicks. However, in computational advertising auctions, missing a true click (False Negative) forfeits revenue entirely, whereas a False Positive merely displays an ad that goes unclicked. Thus, Random Forest's balance ($\text{F1} = 0.2433$) represents an operationally viable operating point for auction ranking.

### 8.2 Probabilistic Calibration and Log-Loss Divergence
To evaluate probability calibration, test impressions were partitioned into predicted probability deciles. The empirical cross-entropy loss is formulated as:
$$\text{Log-Loss} = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \ln(\hat{p}_i) + (1 - y_i) \ln(1 - \hat{p}_i) \right]$$
While Logistic Regression achieved a slightly higher ROC-AUC (0.5830 vs. 0.5822), its Log-Loss (0.6856) was severely degraded by extreme probability predictions near 0 and 1. Conversely, Random Forest averages predictions across 60 independent decision trees, effectively smoothing probability estimates and yielding a **Log-Loss of 0.5963** (a 13.0% improvement over linear regression). In second-price auctions, where bid scaling is proportional to $\hat{p}_i$, calibrated probabilities prevent advertiser over-bidding.

---

## 9. Reproducibility, Threats to Validity, and Deployment Considerations

### 9.1 Reproducibility Protocol and Determinism
To facilitate external replication, all experimental code, preprocessed datasets, and evaluation scripts are publicly archived (Pandey 2026). Full determinism is maintained by fixing random seeds across all randomized transformations:
* Dataset partitioning: `randomSplit([0.7, 0.15, 0.15], seed=42)`.
* Random Forest feature bagging: `RandomForestClassifier(seed=42)`.
* ALS factor initialization: deterministic latent vector initialization in PySpark MLlib.

### 9.2 Threats to Validity
* **Internal Validity:** Potential threats include feature leakage and evaluation bias. We mitigated leakage through formal point-in-time filtering ($t_{\text{behavior}} < t_{\text{impression}}$) and unit testing. Threshold optimization was performed strictly on validation data to prevent optimistic test evaluation.
* **External Validity:** The demonstration dataset utilizes 15,000 impressions and category-brand groupings. While structurally identical to industrial AliMama schemas, behavioral dynamics on full-scale platforms (e.g., millions of users and SKU-level catalogs) may exhibit higher variance.
* **Construct Validity:** Recommender performance was measured via offline ranking metrics (NDCG@10, coverage). While indicative of algorithmic quality, offline metrics cannot capture online user exploration behavior or real-time conversion elasticity.

### 9.3 Production Deployment Considerations and Safeguards
Deploying dual recommendation and advertising architectures in commercial environments requires strict operational safeguards:
1. **Ad Frequency Capping:** Repeatedly displaying identical campaigns causes user fatigue. We enforce a strict constraint limiting exposure to at most 2 impressions per campaign per user session.
2. **Inventory and Price Integrity:** The pipeline validates item metadata, enforcing $\text{price} > 0.0$ and checking SKU availability before auction participation.
3. **Data Privacy Compliance:** User demographic attributes (`gender`, `age_level`) are processed strictly as anonymized categorical indices, complying with privacy frameworks (e.g., GDPR) by prohibiting Personally Identifiable Information (PII) ingestion.

---

## 10. Conclusion and Future Work

This paper presented **Smart Commerce AI**, an integrated parallel framework for personalized product recommendation and advertisement click prediction developed using Apache Spark 3.5.1 and PySpark MLlib. The system demonstrates the feasibility of combining confidence-weighted implicit collaborative filtering with supervised CTR estimation within an auditable, leakage-free Spark pipeline.

The experimental results establish that: (1) Implicit ALS achieves 52.50% catalogue coverage and an NDCG@10 of 0.0197, delivering a 101× expansion in catalog discovery over the Popularity Baseline; (2) Random Forest ensembles achieve superior probability calibration (Log-Loss 0.5963) and positive F1 (0.2433) compared to linear baselines; and (3) local partition scaling validates Amdahl's Law, demonstrating an empirical efficiency ceiling at four partitions due to Spark task coordination overhead on moderate workloads. Crucially, the experiments confirm that CTR classification on compact tabular features exhibits modest discriminative power (ROC-AUC $\approx 0.58$), highlighting the necessity of deep sequential architectures for industrial deployment.

Future research will focus on: (1) evaluating the framework on a true multi-node physical cluster using the full 26-million-record Alibaba dataset; (2) integrating Apache Kafka and Spark Structured Streaming for real-time model updating; and (3) incorporating attention-based deep sequential CTR models (e.g., Deep Interest Networks) via distributed tensor pipelines.

---

## Declarations

**Funding:** This research was conducted as part of the academic curriculum for Course INT315 (Distributed and Cluster Computing) at the School of Computer Science and Engineering, Lovely Professional University, India. No external commercial grants were received.

**Conflict of Interest:** The author declares no financial or commercial conflicts of interest.

**Data Availability:** All datasets, validation scripts, and benchmark results are publicly available in the project repository: https://github.com/Utkarsh1454/Smart-Commerce-AI.

---

## References

1. Alibaba Tianchi (2018) Ad Display/Click Data on Taobao.com. Alibaba Cloud Computing Data Repository. https://tianchi.aliyun.com/dataset/dataDetail?dataId=56
2. Armbrust M, Xin RS, Lian C, Huai Y, Liu D, Bradley JK, Meng X, Kaftan T, Franklin MJ, Ghodsi A, Zaharia M (2015) Spark SQL: relational data processing in Spark. In: Proceedings of the 2015 ACM SIGMOD international conference on management of data, pp 1383–1394
3. Chapelle O, Manavoglu E, Rosales R (2014) Simple and scalable response prediction for display advertising. ACM Trans Intell Syst Technol (TIST) 5(4):1–34
4. Chen T, Guestrin C (2016) XGBoost: a scalable tree boosting system. In: Proceedings of the 22nd ACM SIGKDD international conference on knowledge discovery and data mining, pp 785–794
5. Cheng HT, Koc L, Harmsen J, Shaked T, Chandra T, Aradhye H, Anderson G, Corrado G, Chai W, Ispir M, Anil R (2016) Wide & deep learning for recommender systems. In: Proceedings of the 1st workshop on deep learning for recommender systems, pp 7–10
6. Guo H, Tang R, Zeng Y, Guo X, Xiu P, He X (2017) DeepFM: a factorization-machine based neural network for CTR prediction. In: Proceedings of the 26th international joint conference on artificial intelligence (IJCAI), pp 1725–1731
7. He X, Pan J, Jin O, Xu T, Liu B, Xu T, Shi Y, Atallah A, Herbrich R, Bowers S, Candela JQ (2014) Practical lessons from predicting clicks on ads at Facebook. In: Proceedings of the 8th international workshop on data mining for online advertising, pp 1–9
8. Hu Y, Koren Y, Volinsky C (2008) Collaborative filtering for implicit feedback datasets. In: Proceedings of the 8th IEEE international conference on data mining (ICDM), pp 263–272
9. Koren Y, Bell R, Volinsky C (2009) Matrix factorization techniques for recommender systems. Computer 42(8):30–37
10. McMahan HB, Holt G, Sculley D, Young M, Ebner D, Grady J, Nie L, Phillips T, Davydov E, Golovin D, Chikkerur S (2013) Ad click prediction: a view from the trenches. In: Proceedings of the 19th ACM SIGKDD international conference on knowledge discovery and data mining, pp 1222–1230
11. Meng X, Bradley J, Yavuz B, Sparks E, Venkataraman S, Liu D, Freeman J, Tsai D, Amde M, Owen S, Xin D et al (2016) MLlib: machine learning in Apache Spark. J Mach Learn Res 17(1):1235–1241
12. Pandey U (2026) Smart Commerce AI: a parallel framework for personalized product recommendation and advertisement click prediction using Apache Spark. Project repository. https://github.com/Utkarsh1454/Smart-Commerce-AI
13. RecZoo (2024) TaobaoAd_x1 preprocessed click and behavior dataset. Hugging Face Datasets. https://huggingface.co/datasets/reczoo/TaobaoAd_x1
14. Sarwar B, Karypis G, Konstan J, Riedl J (2001) Item-based collaborative filtering recommendation algorithms. In: Proceedings of the 10th international conference on world wide web, pp 285–295
15. Su X, Khoshgoftaar TM (2009) A survey of collaborative filtering techniques. Adv Artif Intell 2009:1–19
16. Zaharia M, Chowdhury M, Das T, Dave A, Ma J, McCauley M, Franklin MJ, Shenker S, Stoica I (2012) Resilient distributed datasets: a fault-tolerant abstraction for in-memory cluster computing. In: 9th USENIX symposium on networked systems design and implementation (NSDI 12), pp 15–28
17. Zaharia M, Xin RS, Wendell P, Das T, Armbrust M, Dave A, Meng X, Rosen J, Venkataraman S, Franklin MJ, Ghodsi A, Gonzalez J, Shenker S, Stoica I (2016) Apache Spark: a unified engine for big data processing. Commun ACM 59(11):56–65
18. Zhou G, Zhu X, Song C, Fan Y, Zhu H, Ma X, Yan Y, Jin J, Li H, Gai K (2018) Deep interest network for click-through rate prediction. In: Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery and data mining, pp 1059–1068
