# Complete Execution & Operational Run Guide
**Course:** INT315 – Distributed and Cluster Computing (CA2)  
**Project:** Smart Commerce AI: A Distributed Personalized Product and Advertisement Recommendation System Using Apache Spark  
**Lead Author / Student:** Utkarsh Pandey  
**Institution:** Lovely Professional University  

---

## 1. System Prerequisites

Before running the project, verify that the following runtimes are installed on your workstation:
1. **Python:** Version 3.10, 3.11, 3.12, or 3.13 (64-bit).
2. **Java Development Kit (JDK):** OpenJDK 11 or 17 (Java 17 recommended; verify with `java -version`).
3. **Git:** For repository cloning and branch tracking.
4. **Hardware Minimums:** 8 GB RAM (16 GB recommended), 4 CPU cores, 5 GB free disk space.

---

## 2. Environment Setup

Clone the repository and set up an isolated virtual environment:

```powershell
# 1. Clone repository
git clone https://github.com/Utkarsh1454/Smart-Commerce-AI.git
cd Smart-Commerce-AI/Smart_Commerce_AI

# 2. Create isolated virtual environment
python -m venv .venv

# 3. Activate virtual environment
# On Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# On Linux / macOS bash:
source .venv/bin/activate

# 4. Install production dependencies
pip install -r requirements.txt
```

---

## 3. Step-by-Step Pipeline Execution

Execute the pipeline in the exact sequence outlined below:

### Step 1: Automated Dataset Schema & Integrity Validation
```powershell
python scripts/validate_dataset_integrity.py
```
*Expected Outcome:* Validates all 4 CSV datasets (`raw_sample.csv`, `ad_feature.csv`, `user_profile.csv`, `user_behavior.csv`), confirming non-null identifiers, valid action types, and non-empty rows.

### Step 2: Temporal Point-in-Time Preprocessing & Feature Engineering
```powershell
python -m src.preprocess
```
*Expected Outcome:*
- Enforces strict point-in-time filtering (`behavior.time_stamp < raw_sample.time_stamp`) to eliminate data leakage.
- Executes Broadcast Hash Joins on ad metadata and user profile dimensions.
- Generates Parquet feature stores:
  - `data/processed/user_item_interactions.parquet` (39,958 rows)
  - `data/processed/ctr_features.parquet` (15,000 rows)
  - `data/processed/user_id_map.parquet` & `item_id_map.parquet`

### Step 3: Run Automated Temporal Leakage Unit Test
```powershell
python -m tests.test_data_leakage
```
*Expected Outcome:* Injects simulated concurrent and future actions into a synthetic test impression. Verifies with assertion exit code 0 that future actions have zero influence on feature counters.

### Step 4: Distributed Model Training & 3x Scalability Benchmarking
```powershell
python -m src.train
```
*Expected Outcome:*
- Trains Implicit ALS recommender and evaluates NDCG@10, Precision@10, and Catalog Coverage against a Popularity Baseline.
- Fits ElasticNet Logistic Regression and Random Forest CTR models with inverse class weighting ($w_1 = 3.886, w_0 = 0.574$).
- Optimizes classification decision thresholds on the independent validation split ($\tau^* = 0.40$ for LogReg, $\tau^* = 0.30$ for Random Forest).
- Runs 3-repetition scalability benchmark across $p \in \{1, 2, 4, 8\}$ partitions.
- Saves model artifacts to `models/` and exports metrics to `results/`.

### Step 5: Recommendation & Ad Ranking Serving Engine
```powershell
python -m src.recommend
```
*Expected Outcome:*
- Generates Top-10 organic recommendations for 1,200 users (12,000 recommendations) mapped to catalog items.
- Scores candidate ads via Random Forest, applies frequency capping ($\le 2$ ads per campaign per user), and selects Top-5 sponsored ads (5,972 ad placements).
- Exports single-file CSV artifacts to `results/product_recommendations_csv` and `results/ad_recommendations_csv`.

---

## 4. Running the Applications

### Interactive Storefront (HTML / Tailwind CSS / Vanilla JS)
Serve the light-palette e-commerce storefront on port 3000:
```powershell
# From repository root or Smart_Commerce_AI/
python -m http.server 3000
```
- Open browser at: **`http://localhost:3000`**
- Features: Live product catalog search, category filtering, cart management, organic ALS recommendations, and frequency-capped sponsored ads.

### Streamlit Distributed Analytics Dashboard
Launch the real-time cluster metrics dashboard on port 8501:
```powershell
streamlit run app.py --server.port 8501
```
- Open browser at: **`http://localhost:8501`**
- Features: Real benchmark summary, ALS vs. Popularity catalog coverage comparison, CTR ROC curves, confusion matrix breakdown, and cluster partition speedup charts.

---

## 5. Vercel Cloud Deployment

The repository includes pre-configured cloud deployment settings:
- `vercel.json`: Specifies static builds and Python serverless function routing.
- `app.py`: Implements `class handler(BaseHTTPRequestHandler)` compliant with Vercel Python runtime.
- `.vercelignore`: Excludes heavy Spark virtual environments, local model checkpoints, and temp directories.

To deploy via Vercel CLI:
```powershell
npm install -g vercel
vercel deploy --prod
```
