# Smart Commerce AI – Data Directory

This directory houses the raw and processed data assets for the **Smart Commerce AI** cluster computing pipeline.

## Directory Structure

```
data/
├── raw/                     # Input raw CSV tables (Demo synthetic or genuine Taobao)
│   ├── raw_sample.csv       # Impression logs and click labels
│   ├── ad_feature.csv       # Ad creative metadata
│   ├── user_profile.csv     # Customer demographics
│   └── behavior_log.csv     # Longitudinal browsing clickstream
│
├── processed/               # Spark-generated Snappy-compressed Parquet tables
│   ├── ctr_features.parquet          # Leakage-free point-in-time CTR feature matrix
│   ├── user_item_interactions.parquet# Confidence-weighted ALS interaction matrix
│   ├── user_id_map.parquet           # Dense integer mapping for users
│   └── item_id_map.parquet           # Dense integer mapping for category-brand items
│
└── README.md                # This file
```

## Dataset Provenance Note
- The bundled CSVs in `raw/` are **synthetic demonstration data** generated with Taobao-aligned schemas for immediate reproducibility.
- For instructions on streaming and sampling the 2.44 GB genuine RecZoo TaobaoAd_x1 dataset, consult [DATASET_DOCUMENTATION.md](../../DATASET_DOCUMENTATION.md) and [TaobaoAd_Manageable_Dataset_Kit/](../../TaobaoAd_Manageable_Dataset_Kit/).
