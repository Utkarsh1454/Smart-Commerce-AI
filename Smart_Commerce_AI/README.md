# Smart Commerce AI
## Distributed Personalized Product & Advertisement Recommendation using Apache Spark

### What is included
- PySpark preprocessing pipeline
- ALS implicit-feedback recommender over category-brand interactions
- Logistic Regression and Random Forest CTR models
- Streamlit demo dashboard
- Synthetic Taobao-shaped sample data so the project can run immediately
- Research paper draft and data dictionary

### Data provenance — read this first
The included CSVs are synthetic demonstration data generated for software testing and coursework development. They are NOT original Alibaba/Taobao data. Do not claim results from them as results on Alibaba's real dataset.

Official / research data sources:
1. Alibaba Tianchi Taobao Display Ad Click dataset: https://tianchi.aliyun.com/dataset/dataDetail?dataId=56
2. Alibaba MaxCompute public commerce dataset docs: https://www.alibabacloud.com/help/en/maxcompute/user-guide/overview-of-public-datasets
3. RecZoo TaobaoAd_x1 preprocessed joined dataset: https://huggingface.co/datasets/reczoo/TaobaoAd_x1 (25,029,426 rows, 2.44 GB; preprocessed joined format, not the original four raw files)

For original raw exports, expected canonical filenames:
- raw_sample.csv
- ad_feature.csv
- user_profile.csv
- behavior_log.csv

The preprocessing script normalizes common source variants (`user`/`user_id`, `userid`/`user_id`, `cate`/`cate_id`, `brand`/`brand_id`).

### Project requirements
- Python 3.9–3.11 recommended
- Java 11 or Java 17
- Apache Spark / PySpark 3.5.1
- 8 GB RAM minimum for the included sample; more is recommended for real Taobao data

### Setup (Windows PowerShell)
```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Run
```powershell
python scripts/inspect_data.py
python -m src.preprocess
python -m src.train
python -m src.recommend
streamlit run app.py
```

### Model design
**Product recommender:** implicit ALS trained on weighted category-brand interactions from behavior_log. Interaction weights are project design choices: view=1, favourite=3, cart=4, purchase=5. Since the source behaviour table records category and brand rather than product IDs, the ALS item is a category-brand pair, not an individual SKU.

**Ad model:** binary click prediction using joined impression, ad, profile and historical category-affinity features. The pipeline compares Logistic Regression and Random Forest using ROC-AUC. Candidate ad ranking is demonstrated from ads present in the sampled impression history; a production deployment should score a current eligible ad inventory and enforce consent, frequency caps and opt-out.

### Important research limitations
- The demo data is synthetic; replace it with genuine source data before reporting real-world findings.
- The provided model evaluation uses a random train/test split for a reproducible demo. For the final paper, implement a strict chronological split and ensure no future behaviour leaks into earlier impression features.
- ALS RMSE on implicit feedback is only a diagnostic. Prefer Recall@K, Precision@K and NDCG@K for final ranking evaluation.
- The source Taobao behaviour log does not provide individual SKU identifiers; category-brand recommendation is the defensible formulation.
- Avoid sensitive demographic targeting in the ad-ranking policy. User profile fields are included for dataset familiarization; test a model that excludes sensitive or unnecessary fields and report the difference.

### Outputs
- `data/processed/*.parquet`
- `models/als_recommender/`
- `models/logistic_regression/`
- `models/random_forest/`
- `results/ctr_metrics.csv`
- `results/top_product_recommendations.json`
- `results/product_recommendations_csv/`
- `results/ad_recommendations_csv/`

### Reproducibility
Random seed is fixed at 42 for synthetic generation and model splitting. Record OS, CPU, RAM, Java version and Spark version in your final paper.
