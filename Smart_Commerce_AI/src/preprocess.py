"""
Smart Commerce AI - Distributed Data Preprocessing & Feature Engineering
Course: INT315 - Distributed and Cluster Computing
Component: PySpark Data Ingestion, Broadcast Joins & Leakage-Free Temporal Engineering
"""

import sys
import os
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import IntegerType, DoubleType
from pyspark.sql import Window
from src.config import RAW, PROCESSED, WEIGHTS

def main():
    print("="*70)
    print("SMART COMMERCE AI - DISTRIBUTED DATA PREPROCESSING (PYSPARK)")
    print("="*70)

    spark = (
        SparkSession.builder.appName("SmartCommerce-Preprocessing")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    # Ingest raw source CSVs
    raw_path = RAW / "raw_sample.csv"
    ads_path = RAW / "ad_feature.csv"
    users_path = RAW / "user_profile.csv"
    behavior_path = RAW / "behavior_log.csv"

    for p in [raw_path, ads_path, users_path, behavior_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required raw CSV: {p}")

    raw = spark.read.option("header", True).option("inferSchema", True).csv(str(raw_path))
    ads = spark.read.option("header", True).option("inferSchema", True).csv(str(ads_path))
    users = spark.read.option("header", True).option("inferSchema", True).csv(str(users_path))
    behavior = spark.read.option("header", True).option("inferSchema", True).csv(str(behavior_path))

    # Normalize common column name variants found in Taobao exports
    def rename_if_present(df, old, new):
        return df.withColumnRenamed(old, new) if old in df.columns and new not in df.columns else df

    raw = rename_if_present(raw, "user", "user_id")
    ads = rename_if_present(ads, "brand", "brand_id")
    ads = rename_if_present(ads, "customer", "customer_id")
    users = rename_if_present(users, "userid", "user_id")
    behavior = rename_if_present(behavior, "user", "user_id")
    behavior = rename_if_present(behavior, "nick", "user_id")
    behavior = rename_if_present(behavior, "cate", "cate_id")
    behavior = rename_if_present(behavior, "brand", "brand_id")

    # Data cleaning and deduplication
    raw = raw.dropna(subset=["user_id", "adgroup_id", "time_stamp", "clk"]).dropDuplicates()
    raw = raw.withColumn("clk", F.col("clk").cast(IntegerType()))
    raw = raw.filter(F.col("clk").isin(0, 1))

    ads = ads.dropna(subset=["adgroup_id", "cate_id", "brand_id"]).dropDuplicates(["adgroup_id"])
    ads = ads.withColumn("price", F.col("price").cast(DoubleType()))

    users = users.dropDuplicates(["user_id"])

    behavior = behavior.dropna(subset=["user_id", "btag", "cate_id", "brand_id"]).dropDuplicates()
    behavior = behavior.filter(F.col("btag").isin(list(WEIGHTS.keys())))

    # -------------------------------------------------------------------------
    # STAGE 1: PRODUCT RECOMMENDATION DATA PREPARATION (IMPLICIT ALS)
    # Formulate confidence-weighted user-item interactions
    # -------------------------------------------------------------------------
    print("\n[PREPROCESS] Constructing implicit-feedback ALS interaction matrix...")
    weight_expr = (
        F.when(F.col("btag") == "pv", WEIGHTS["pv"])
        .when(F.col("btag") == "fav", WEIGHTS["fav"])
        .when(F.col("btag") == "cart", WEIGHTS["cart"])
        .otherwise(WEIGHTS["buy"])
    )

    interactions = (
        behavior.withColumn("weight", weight_expr)
        .withColumn("item_key", F.concat_ws("_", F.col("cate_id").cast("string"), F.col("brand_id").cast("string")))
        .groupBy("user_id", "item_key")
        .agg(F.sum("weight").alias("rating"))
    )

    # Dense integer mapping for users and items (ALS factor requirements)
    user_map = (
        interactions.select("user_id").distinct()
        .withColumn("als_user_id", F.row_number().over(Window.orderBy("user_id")).cast("int") - 1)
    )
    item_map = (
        interactions.select("item_key").distinct()
        .withColumn("item_id", F.row_number().over(Window.orderBy("item_key")).cast("int") - 1)
    )

    interactions_dense = (
        interactions.join(user_map, "user_id")
        .join(item_map, "item_key")
        .select("als_user_id", "item_id", "rating")
    )

    user_map.write.mode("overwrite").parquet(str(PROCESSED / "user_id_map.parquet"))
    item_map.write.mode("overwrite").parquet(str(PROCESSED / "item_id_map.parquet"))
    interactions_dense.write.mode("overwrite").parquet(str(PROCESSED / "user_item_interactions.parquet"))
    print("  -> Exported ALS interaction tables to Parquet.")

    # -------------------------------------------------------------------------
    # STAGE 2: POINT-IN-TIME CTR FEATURE MATRIX (ZERO DATA LEAKAGE)
    # Strictly enforces: behavior.time_stamp < raw_sample.time_stamp
    # Broadcast Hash Joins eliminate wide shuffle on small dimension catalogs
    # -------------------------------------------------------------------------
    print("\n[PREPROCESS] Performing Broadcast Hash Joins & Leakage-Free Temporal Aggregations...")

    raw_ads = raw.join(F.broadcast(ads), on="adgroup_id", how="left")
    raw_full = raw_ads.join(F.broadcast(users), on="user_id", how="left")

    # Distinct impression keys for point-in-time behavior lookup
    imp_keys = (
        raw_full.select(
            F.col("user_id"),
            F.col("adgroup_id"),
            F.col("cate_id"),
            F.col("time_stamp").alias("imp_time")
        )
        .distinct()
    )

    # Inner join on user and category, strictly requiring past timestamps
    past_behavior = (
        imp_keys.join(
            behavior.select(
                F.col("user_id"),
                F.col("cate_id"),
                F.col("time_stamp").alias("beh_time"),
                F.col("btag")
            ),
            on=["user_id", "cate_id"],
            how="inner"
        )
        .filter(F.col("beh_time") < F.col("imp_time"))
    )

    # Historical affinity aggregates prior to the impression cutoff
    past_affinity = (
        past_behavior.groupBy("user_id", "adgroup_id", "cate_id", "imp_time")
        .agg(
            F.count("*").alias("user_category_events"),
            F.sum(F.when(F.col("btag") == "buy", 1).otherwise(0)).alias("user_category_buys"),
            F.sum(F.when(F.col("btag") == "cart", 1).otherwise(0)).alias("user_category_carts")
        )
    )

    # Left join point-in-time affinity back to impressions
    ctr_features = (
        raw_full.join(
            past_affinity,
            on=(
                (raw_full.user_id == past_affinity.user_id) &
                (raw_full.adgroup_id == past_affinity.adgroup_id) &
                (raw_full.cate_id == past_affinity.cate_id) &
                (raw_full.time_stamp == past_affinity.imp_time)
            ),
            how="left"
        )
        .drop(past_affinity.user_id)
        .drop(past_affinity.adgroup_id)
        .drop(past_affinity.cate_id)
        .drop(past_affinity.imp_time)
    )

    # Fill default zeros for users with no prior category history
    for c in ["user_category_events", "user_category_buys", "user_category_carts"]:
        ctr_features = ctr_features.withColumn(c, F.coalesce(F.col(c), F.lit(0)).cast("double"))

    ctr_features = ctr_features.withColumn("pid", F.coalesce(F.col("pid").cast("string"), F.lit("unknown")))

    # Verify physical execution plan to confirm BroadcastHashJoin
    print("\n--- Physical Execution Plan Inspection (BroadcastHashJoin Verification) ---")
    ctr_features.explain(mode="simple")

    ctr_features.write.mode("overwrite").parquet(str(PROCESSED / "ctr_features.parquet"))
    print("  -> Exported point-in-time CTR features to Parquet.")

    # Save cleaned source tables for auditability
    raw.write.mode("overwrite").parquet(str(PROCESSED / "clean_raw_sample.parquet"))
    ads.write.mode("overwrite").parquet(str(PROCESSED / "clean_ad_feature.parquet"))
    users.write.mode("overwrite").parquet(str(PROCESSED / "clean_user_profile.parquet"))
    behavior.write.mode("overwrite").parquet(str(PROCESSED / "clean_behavior_log.parquet"))

    print("\n" + "="*70)
    print("PREPROCESSING COMPLETED SUCCESSFULLY")
    print(f"  • Raw Impressions:     {raw.count():,}")
    print(f"  • Ads Dimension:       {ads.count():,}")
    print(f"  • User Profiles:       {users.count():,}")
    print(f"  • Behavior Logs:       {behavior.count():,}")
    print(f"  • ALS Interactions:    {interactions_dense.count():,}")
    print(f"  • Final CTR Features:  {ctr_features.count():,}")
    print("="*70 + "\n")

    spark.stop()

if __name__ == "__main__":
    main()
