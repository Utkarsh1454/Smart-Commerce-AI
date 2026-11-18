# Smart Commerce AI – Comprehensive Dataset Documentation

**Project:** Smart Commerce AI: A Distributed Personalized Product & Advertisement Recommendation System  
**Course:** INT315 – Distributed and Cluster Computing  
**Dataset Architecture & Provenance Specification**

---

## 1. Executive Summary & Provenance Statement

To ensure academic transparency and reproducible cluster evaluation, this project clearly distinguishes between two dataset tiers:

1. **Synthetic Demonstration Dataset (Bundled):** Included within `Smart_Commerce_AI/data/raw/` (and root `data/raw/`). Synthesized to match the exact schema, relational joins, data types, and temporal distributions of the Alibaba Taobao Ad Click dataset. This enables immediate, zero-download replication of the PySpark pipeline, ALS model training, and cluster benchmarks.
2. **Genuine Alibaba Taobao Datasets (External):**
   - **Alibaba Tianchi Ad Display / Click Dataset (dataId=56):** 26 million impression logs, 1.14 million user profiles, 846,808 ad creatives, and 700+ million behavioral events recorded between May 6 and May 13, 2017.
   - **RecZoo TaobaoAd_x1 (Hugging Face):** A 2.44 GB compressed joined representation containing 25,029,426 rows across official `train.csv` (21,929,911 rows) and `test.csv` (3,099,515 rows).

> [!IMPORTANT]
> The demonstration files bundled in this repository are **synthetically generated demonstration records**. Under no circumstances should they be represented as genuine proprietary Alibaba telemetry. When genuine data is downloaded via the provided `TaobaoAd_Manageable_Dataset_Kit`, the pipeline dynamically ingests and scales to those genuine records.

---

## 2. Relational Schema & Entity Specifications

The distributed e-commerce architecture relies on four foundational relational tables:

```
+-----------------------------------------------------------------------------------------------+
|                               RELATIONAL ENTITY RELATIONSHIP                                  |
+-----------------------------------------------------------------------------------------------+
|  user_profile (1,200 rows)                                                                    |
|  [PK] user_id ────┐                                                                           |
|                   │ (1:N)                                                                     |
|                   ▼                                                                           |
|  raw_sample (15,000 rows) ◄──── [PK] adgroup_id ──── ad_feature (400 rows)                   |
|  • user_id              (N:1)                        • adgroup_id                             |
|  • adgroup_id                                        • cate_id ─────┐                         |
|  • time_stamp (Cutoff)                               • brand_id     │                         |
|  • clk (Target 0/1)                                  • price        │                         |
|         │                                                           │                         |
|         ▼                                                           │                         |
|  behavior_log (42,000 rows)                                         │                         |
|  • user_id ◄────────────────────────────────────────────────────────┘ (Point-in-Time Join)    |
|  • cate_id                                                                                    |
|  • brand_id                                                                                   |
|  • btag (pv, fav, cart, buy)                                                                  |
|  • time_stamp (Strictly: behavior.time_stamp < raw_sample.time_stamp)                         |
+-----------------------------------------------------------------------------------------------+
```

### Table 2.1: `raw_sample.csv` (Ad Display Logs)
Contains impression-level ad display events with binary click outcomes.
- **Row Count:** 15,000 records
- **Primary Key:** Synthetic synthetic event identifier `(user_id, adgroup_id, time_stamp)`
- **Fields:**
  - `user_id` (Integer): Anonymized user identifier (100,001 to 101,200).
  - `adgroup_id` (Integer): Advertisement unit creative ID (50,001 to 50,400).
  - `time_stamp` (BigInt): Unix epoch timestamp of ad display (range: 1,495,564,080 to 1,496,045,400; May 2017).
  - `pid` (String): Ad slot position classification (e.g., `430548_1007`, `430539_1007`).
  - `nonclk` (Integer): 1 if impression was not clicked; 0 otherwise.
  - `clk` (Integer): **Target label.** 1 if impression was clicked; 0 otherwise (Positive class rate: ~10.0%).

### Table 2.2: `ad_feature.csv` (Ad Metadata Catalog)
Contains merchant and creative attributes for candidate advertisements.
- **Row Count:** 400 records
- **Primary Key:** `adgroup_id`
- **Fields:**
  - `adgroup_id` (Integer): Unique ad group creative identifier.
  - `cate_id` (Integer): Category hierarchy identifier (e.g., 101 to 130).
  - `campaign_id` (Integer): Commercial advertising campaign identifier.
  - `customer_id` (Integer): Merchant advertiser entity identifier.
  - `brand_id` (Integer): Brand catalog identifier (e.g., 1001 to 1080).
  - `price` (Double): Display price in currency units ($5.00 to $999.00).

### Table 2.3: `user_profile.csv` (Demographic Segments)
Contains customer demographic and behavioral cohort features.
- **Row Count:** 1,200 records
- **Primary Key:** `user_id`
- **Fields:**
  - `user_id` (Integer): Unique customer entity identifier.
  - `cms_segid` (Integer): Micro-segment cluster ID.
  - `cms_group_id` (Integer): Macro demographic group ID.
  - `final_gender_code` (Integer): 1 (Male), 2 (Female).
  - `age_level` (Integer): Categorical age bin (1: <18, 2: 18-24, 3: 25-34, 4: 35-49, 5: 50+).
  - `pvalue_level` (Integer): Purchasing power tier (1: Low, 2: Medium, 3: High).
  - `shopping_level` (Integer): Shopping engagement frequency grade (1 to 3).
  - `occupation` (Integer): 0 (General/Other), 1 (College student).
  - `new_user_class_level` (Integer): City-tier residency level (1 to 4).

### Table 2.4: `behavior_log.csv` (Implicit Browsing Stream)
Contains longitudinal customer clickstream events used to build matrix factorization signals and historical affinities.
- **Row Count:** 42,000 records
- **Fields:**
  - `user_id` (Integer): Customer entity identifier.
  - `time_stamp` (BigInt): Unix epoch timestamp of interaction (range: 1,493,643,660 to 1,495,363,440).
  - `btag` (String): Interaction type:
    - `pv` (Page View): Implicit baseline engagement (Weight = 1.0).
    - `fav` (Favorite/Wishlist): Moderate preference signal (Weight = 3.0).
    - `cart` (Add to Shopping Cart): Strong purchase intent (Weight = 4.0).
    - `buy` (Completed Purchase): Definitive positive preference (Weight = 5.0).
  - `cate_id` (Integer): Category of viewed/purchased item.
  - `brand_id` (Integer): Brand of viewed/purchased item.

---

## 3. Data Cleaning, Preprocessing & Type Handling

1. **Deduplication:** Raw tables are deduplicated on their logical keys (`user_id`, `adgroup_id`, `time_stamp`).
2. **Missing Value Imputation:**
   - Numerical demographic indicators (`age_level`, `pvalue_level`, `shopping_level`) are imputed with the median.
   - Missing behavioral interaction aggregates default to `0` via SQL `COALESCE`.
   - Missing placement slot IDs are assigned to an `"unknown"` category token.
3. **Data Type Casting:** Explicit casting to DoubleType (for Spark MLlib feature vectors) and IntegerType (for indexing and ALS factors).

---

## 4. Sampling & Manageable Dataset Toolkit

For cluster environments with constrained hardware (e.g., student laptops, single-node clusters), processing the full 2.44 GB / 25M row dataset can cause JVM heap overflow.
The `TaobaoAd_Manageable_Dataset_Kit/` provides a streaming chunk-based down-sampler:
```bash
python TaobaoAd_Manageable_Dataset_Kit/scripts/download_and_sample.py --train-samples 50000 --test-samples 10000 --seed 42
```
- **Sampling Methodology:** Stratified reservoir sampling across chunk blocks preserving the ~5% CTR imbalance ratio.
- **Integrity Verification:** MD5 checksum validation against official Hugging Face hashes.

---

## 5. Licensing and Terms of Use

- **Alibaba Tianchi Dataset:** Available under the Tianchi Open Data License Agreement for academic research and educational evaluation. Commercial redistribution is prohibited.
- **RecZoo TaobaoAd_x1:** Distributed under Creative Commons Attribution-NonCommercial-ShareAlike (CC BY-NC-SA 4.0).
- **Synthetic Demonstration Package:** Authored specifically for INT315 CA2 and licensed under the MIT License.
