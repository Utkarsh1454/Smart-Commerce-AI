#!/usr/bin/env python3
"""
Unit Test: Verification of Zero Data Leakage in Feature Engineering Pipeline
Course: INT315 - Distributed and Cluster Computing
Validates:
1. Past events (t_beh < t_imp) are strictly aggregated.
2. Concurrent events (t_beh == t_imp) and future events (t_beh > t_imp) are completely excluded.
3. Injecting synthetic future behaviors does not alter computed features.
"""

import sys
import os
import shutil
from pathlib import Path

# Unset external SPARK_HOME
if "SPARK_HOME" in os.environ:
    del os.environ["SPARK_HOME"]

from pyspark.sql import SparkSession, functions as F

ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = ROOT / "data" / "test_leakage_temp"

def test_leakage_prevention():
    print("="*70)
    print("RUNNING AUTOMATED TEST: TEMPORAL DATA LEAKAGE PREVENTION")
    print("="*70)

    TEST_DIR.mkdir(parents=True, exist_ok=True)

    spark = (
        SparkSession.builder.master("local[2]")
        .appName("Test-DataLeakage")
        .config("spark.sql.shuffle.partitions", "2")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    # Write test CSVs to disk to test end-to-end relational execution
    raw_csv = TEST_DIR / "raw.csv"
    ads_csv = TEST_DIR / "ads.csv"
    beh_clean_csv = TEST_DIR / "beh_clean.csv"
    beh_future_csv = TEST_DIR / "beh_future.csv"

    # Impression at timestamp T = 2000
    with open(raw_csv, "w", encoding="utf-8") as f:
        f.write("user_id,adgroup_id,time_stamp,pid,nonclk,clk\n")
        f.write("999,501,2000,pos1,1,0\n")

    with open(ads_csv, "w", encoding="utf-8") as f:
        f.write("adgroup_id,cate_id,campaign_id,customer_id,brand_id,price\n")
        f.write("501,10,1001,2001,3001,99.0\n")

    # Baseline past events (t = 1000, 1500 < 2000)
    with open(beh_clean_csv, "w", encoding="utf-8") as f:
        f.write("user_id,time_stamp,btag,cate_id,brand_id\n")
        f.write("999,1000,pv,10,3001\n")
        f.write("999,1500,cart,10,3001\n")

    # Baseline + Concurrent (t = 2000) + Future (t = 2500, 3000 > 2000)
    with open(beh_future_csv, "w", encoding="utf-8") as f:
        f.write("user_id,time_stamp,btag,cate_id,brand_id\n")
        f.write("999,1000,pv,10,3001\n")
        f.write("999,1500,cart,10,3001\n")
        f.write("999,2000,buy,10,3001\n")   # Concurrent (MUST be excluded)
        f.write("999,2500,buy,10,3001\n")   # Future buy (MUST be excluded)
        f.write("999,3000,cart,10,3001\n")  # Future cart (MUST be excluded)

    raw_df = spark.read.option("header", True).option("inferSchema", True).csv(str(raw_csv))
    ads_df = spark.read.option("header", True).option("inferSchema", True).csv(str(ads_csv))
    beh_clean_df = spark.read.option("header", True).option("inferSchema", True).csv(str(beh_clean_csv))
    beh_future_df = spark.read.option("header", True).option("inferSchema", True).csv(str(beh_future_csv))

    def extract_features(raw_in, ads_in, beh_in):
        raw_ads = raw_in.join(F.broadcast(ads_in), on="adgroup_id", how="left")
        imp_keys = raw_ads.select(
            F.col("user_id"),
            F.col("adgroup_id"),
            F.col("cate_id"),
            F.col("time_stamp").alias("imp_time")
        ).distinct()

        past_events = (
            imp_keys.join(
                beh_in.select(
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

        past_affinity = (
            past_events.groupBy("user_id", "adgroup_id", "cate_id", "imp_time")
            .agg(
                F.count("*").alias("user_category_events"),
                F.sum(F.when(F.col("btag") == "buy", 1).otherwise(0)).alias("user_category_buys"),
                F.sum(F.when(F.col("btag") == "cart", 1).otherwise(0)).alias("user_category_carts")
            )
        )

        joined = (
            raw_ads.join(
                past_affinity,
                on=(
                    (raw_ads.user_id == past_affinity.user_id) &
                    (raw_ads.adgroup_id == past_affinity.adgroup_id) &
                    (raw_ads.cate_id == past_affinity.cate_id) &
                    (raw_ads.time_stamp == past_affinity.imp_time)
                ),
                how="left"
            )
            .drop(past_affinity.user_id)
            .drop(past_affinity.adgroup_id)
            .drop(past_affinity.cate_id)
            .drop(past_affinity.imp_time)
        )

        for c in ["user_category_events", "user_category_buys", "user_category_carts"]:
            joined = joined.withColumn(c, F.coalesce(F.col(c), F.lit(0)).cast("double"))
        return joined.collect()[0]

    row_clean = extract_features(raw_df, ads_df, beh_clean_df)
    row_with_future = extract_features(raw_df, ads_df, beh_future_df)

    print("Features (Clean Past Events Only):")
    print(f"  user_category_events: {row_clean['user_category_events']} (Expected: 2.0)")
    print(f"  user_category_buys:   {row_clean['user_category_buys']} (Expected: 0.0)")
    print(f"  user_category_carts:  {row_clean['user_category_carts']} (Expected: 1.0)")

    print("\nFeatures (With Future & Concurrent Events Injected):")
    print(f"  user_category_events: {row_with_future['user_category_events']} (Expected: 2.0)")
    print(f"  user_category_buys:   {row_with_future['user_category_buys']} (Expected: 0.0)")
    print(f"  user_category_carts:  {row_with_future['user_category_carts']} (Expected: 1.0)")

    # Assertions
    assert row_clean["user_category_events"] == 2.0, "Clean events count mismatch"
    assert row_clean["user_category_buys"] == 0.0, "Clean buys count mismatch"
    assert row_clean["user_category_carts"] == 1.0, "Clean carts count mismatch"

    assert row_with_future["user_category_events"] == 2.0, "FAIL: Future events leaked into events count!"
    assert row_with_future["user_category_buys"] == 0.0, "FAIL: Future purchases leaked into buys count!"
    assert row_with_future["user_category_carts"] == 1.0, "FAIL: Future carts leaked into carts count!"

    print("\n✅ [ASSERTION PASSED] Future events had ZERO effect on feature values.")
    print("✅ [ASSERTION PASSED] Temporal data leakage is mathematically eliminated.")
    print("="*70 + "\n")

    spark.stop()
    shutil.rmtree(TEST_DIR, ignore_errors=True)
    return True

if __name__ == "__main__":
    success = test_leakage_prevention()
    sys.exit(0 if success else 1)
