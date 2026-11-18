"""
Smart Commerce AI - Candidate Recommendation & Ad Ranking
Component: Serving & Export Engine (ALS Products + Scored Sponsored Ads)
"""

import time
from pathlib import Path
from pyspark.sql import SparkSession, DataFrame, functions as F
from pyspark.sql import Window
from pyspark.ml.recommendation import ALSModel
from pyspark.ml.functions import vector_to_array

from src.config import PROCESSED, MODELS, RESULTS

def main():
    spark = (
        SparkSession.builder.appName("SmartCommerce-Recommend")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    # 1. Product Recommendations (Implicit ALS)
    print("Generating top product recommendations from ALS model...")
    als = ALSModel.load(str(MODELS / "als_recommender"))
    user_map = spark.read.parquet(str(PROCESSED / "user_id_map.parquet"))
    item_map = spark.read.parquet(str(PROCESSED / "item_id_map.parquet"))

    users = als.recommendForAllUsers(10)
    exploded = (
        users.select("als_user_id", F.explode("recommendations").alias("rec"))
        .select("als_user_id", F.col("rec.item_id").alias("item_id"), F.col("rec.rating").alias("score"))
    )
    product_recs = (
        exploded.join(user_map, "als_user_id")
        .join(item_map, "item_id")
        .select("user_id", "item_key", "score")
    )
    product_recs.write.mode("overwrite").option("header", True).csv(str(RESULTS / "product_recommendations_csv"))
    print("Saved product recommendations to results/product_recommendations_csv")

    # 2. Sponsored Ad CTR Scoring (Random Forest via JVM)
    print("Scoring candidate impressions with Random Forest CTR model...")
    j_model = spark._jvm.org.apache.spark.ml.PipelineModel.load(str(MODELS / "random_forest"))
    candidates = spark.read.parquet(str(PROCESSED / "ctr_features.parquet"))

    numeric = [
        "cms_segid", "cms_group_id", "final_gender_code", "age_level", "pvalue_level",
        "shopping_level", "occupation", "new_user_class_level", "cate_id", "campaign_id",
        "customer_id", "brand_id", "price", "user_category_events", "user_category_buys",
        "user_category_carts"
    ]
    for c in numeric:
        if c in candidates.columns:
            candidates = candidates.withColumn(c, F.col(c).cast("double"))

    scored_jdf = j_model.transform(candidates._jdf)
    scored = DataFrame(scored_jdf, spark).select(
        "user_id", "adgroup_id", "cate_id", "brand_id", "price",
        vector_to_array("probability")[1].alias("predicted_ctr")
    )

    ranked_ads = (
        scored.groupBy("user_id", "adgroup_id", "cate_id", "brand_id", "price")
        .agg(F.avg("predicted_ctr").alias("predicted_ctr"))
        .withColumn(
            "rank",
            F.row_number().over(Window.partitionBy("user_id").orderBy(F.desc("predicted_ctr")))
        )
        .filter(F.col("rank") <= 5)
    )

    ranked_ads.write.mode("overwrite").option("header", True).csv(str(RESULTS / "ad_recommendations_csv"))
    print("Saved top-5 ranked ad recommendations to results/ad_recommendations_csv")
    print("Recommendation generation complete.")

    spark.stop()

if __name__ == "__main__":
    main()
