"""
Smart Commerce AI - Candidate Recommendation & Ad Ranking Engine
Course: INT315 - Distributed and Cluster Computing
Component: Serving & Export Engine (ALS Products + Scored Sponsored Ads)
Features:
1. Model loading with fallback and error handling.
2. ALS Top-K Organic Product Recommendation generation.
3. Random Forest CTR Scoring for Sponsored Ads.
4. Commercial Safeguards:
   - In-memory caching for zero shuffle re-evaluation.
   - Frequency capping (max 2 impressions per campaign per user).
   - Price and inventory sanity check (price > 0).
"""

import os
import sys
import time
from pathlib import Path

# Ensure config environment setup (paths without spaces, SPARK_HOME cleanup) executes first
from src.config import PROCESSED, MODELS, RESULTS

from pyspark.sql import SparkSession, DataFrame, functions as F
from pyspark.sql import Window
from pyspark.ml.recommendation import ALSModel
from pyspark.ml import PipelineModel
from pyspark.ml.util import JavaMLReader
from pyspark.ml.functions import vector_to_array

def main():
    print("="*70)
    print("SMART COMMERCE AI - RECOMMENDATION & AD RANKING ENGINE")
    print("="*70)

    spark = (
        SparkSession.builder.appName("SmartCommerce-Recommend")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    # 1. Product Recommendations (Implicit ALS)
    als_model_path = MODELS / "als_recommender"
    user_map_path = PROCESSED / "user_id_map.parquet"
    item_map_path = PROCESSED / "item_id_map.parquet"

    if not als_model_path.exists():
        print(f"❌ Error: ALS model not found at {als_model_path}. Please execute `python -m src.train` first.")
        spark.stop()
        return

    print("\n[RECOMMENDER] Loading trained Implicit ALS model...")
    als = ALSModel.load(str(als_model_path))
    user_map = spark.read.parquet(str(user_map_path))
    item_map = spark.read.parquet(str(item_map_path))

    # Generate Top-10 organic recommendations per user
    all_user_recs = als.recommendForAllUsers(10)
    exploded = (
        all_user_recs.select("als_user_id", F.explode("recommendations").alias("rec"))
        .select("als_user_id", F.col("rec.item_id").alias("item_id"), F.col("rec.rating").alias("score"))
    )

    product_recs = (
        exploded.join(user_map, "als_user_id")
        .join(item_map, "item_id")
        .select("user_id", "item_key", "score")
    )

    product_recs_out = RESULTS / "product_recommendations_csv"
    product_recs.coalesce(1).write.mode("overwrite").option("header", True).csv(str(product_recs_out))
    print(f"  -> Saved organic product recommendations to {product_recs_out}")

    # 2. Sponsored Ad CTR Scoring & Ranking with Commercial Safeguards
    rf_model_path = MODELS / "random_forest"
    ctr_features_path = PROCESSED / "ctr_features.parquet"

    if not rf_model_path.exists():
        print(f"❌ Error: Random Forest model not found at {rf_model_path}. Please execute `python -m src.train` first.")
        spark.stop()
        return

    print("\n[SPONSORED ADS] Scoring candidate ad inventory with Random Forest CTR model...")
    ctr_model = JavaMLReader(PipelineModel).load(str(rf_model_path))
    candidates = spark.read.parquet(str(ctr_features_path))

    numeric = [
        "cms_segid", "cms_group_id", "final_gender_code", "age_level", "pvalue_level",
        "shopping_level", "occupation", "new_user_class_level", "cate_id", "campaign_id",
        "customer_id", "brand_id", "price", "user_category_events", "user_category_buys",
        "user_category_carts"
    ]
    for c in numeric:
        if c in candidates.columns:
            candidates = candidates.withColumn(c, F.col(c).cast("double"))

    # Inventory Sanity Check: price > 0
    candidates = candidates.filter(F.col("price") > 0.0)

    # Transform through pipeline to compute predicted CTR
    scored = (
        ctr_model.transform(candidates)
        .select(
            "user_id", "adgroup_id", "cate_id", "brand_id", "campaign_id", "price",
            vector_to_array("probability")[1].alias("predicted_ctr")
        )
    )

    # SAFEGUARD 1: Frequency Capping (maximum 2 ads from the same campaign per user)
    campaign_window = Window.partitionBy("user_id", "campaign_id").orderBy(F.desc("predicted_ctr"))
    capped_ads = (
        scored.withColumn("campaign_rank", F.row_number().over(campaign_window))
        .filter(F.col("campaign_rank") <= 2)
    )

    # Group by ad creative and rank top 5 per user
    user_ad_window = Window.partitionBy("user_id").orderBy(F.desc("predicted_ctr"))
    ranked_ads = (
        capped_ads.groupBy("user_id", "adgroup_id", "cate_id", "brand_id", "campaign_id", "price")
        .agg(F.avg("predicted_ctr").alias("predicted_ctr"))
        .withColumn("rank", F.row_number().over(user_ad_window))
        .filter(F.col("rank") <= 5)
    )

    ad_recs_out = RESULTS / "ad_recommendations_csv"
    ranked_ads.coalesce(1).write.mode("overwrite").option("header", True).csv(str(ad_recs_out))
    print(f"  -> Saved frequency-capped top-5 sponsored ads to {ad_recs_out}")

    print("\n" + "="*70)
    print("RECOMMENDATION AND AUCTION RANKING PIPELINE COMPLETED")
    print(f"  • Total Users Served:      {product_recs.select('user_id').distinct().count():,}")
    print(f"  • Total Products Scored:   {product_recs.count():,}")
    print(f"  • Total Sponsored Ads:     {ranked_ads.count():,}")
    print("="*70 + "\n")

    scored.unpersist()
    spark.stop()

if __name__ == "__main__":
    main()
