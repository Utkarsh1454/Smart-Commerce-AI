from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import IntegerType, DoubleType
from pyspark.sql import Window
from pyspark.ml.feature import StringIndexer
from src.config import RAW, PROCESSED, WEIGHTS

def main():
    spark = (SparkSession.builder.appName("SmartCommerce-Preprocessing")
             .config("spark.sql.shuffle.partitions", "8").getOrCreate())
    spark.sparkContext.setLogLevel("WARN")
    raw = spark.read.option("header", True).option("inferSchema", True).csv(str(RAW/"raw_sample.csv"))
    ads = spark.read.option("header", True).option("inferSchema", True).csv(str(RAW/"ad_feature.csv"))
    users = spark.read.option("header", True).option("inferSchema", True).csv(str(RAW/"user_profile.csv"))
    behavior = spark.read.option("header", True).option("inferSchema", True).csv(str(RAW/"behavior_log.csv"))

    # Normalize common column-name variants found in Taobao exports.
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

    required = {
        "raw": (raw, ["user_id","adgroup_id","time_stamp","clk"]),
        "ads": (ads, ["adgroup_id","cate_id","brand_id","price"]),
        "users": (users, ["user_id"]),
        "behavior": (behavior, ["user_id","time_stamp","btag","cate_id","brand_id"])
    }
    for name,(df,cols) in required.items():
        missing=[c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"{name} is missing required columns: {missing}. Found: {df.columns}")

    raw = raw.dropna(subset=["user_id","adgroup_id","time_stamp","clk"]).dropDuplicates()
    raw = raw.withColumn("clk",F.col("clk").cast(IntegerType()))
    raw = raw.filter(F.col("clk").isin(0,1))
    ads = ads.dropna(subset=["adgroup_id","cate_id","brand_id"]).dropDuplicates(["adgroup_id"])
    ads = ads.withColumn("price",F.col("price").cast(DoubleType()))
    users = users.dropDuplicates(["user_id"])
    behavior = behavior.dropna(subset=["user_id","btag","cate_id","brand_id"]).dropDuplicates()
    behavior = behavior.filter(F.col("btag").isin(list(WEIGHTS.keys())))

    # Weighted category/brand interaction scores for the ALS recommender.
    weight_expr = F.when(F.col("btag")=="pv",1.0).when(F.col("btag")=="fav",3.0).when(F.col("btag")=="cart",4.0).otherwise(5.0)
    interactions = (behavior.withColumn("weight",weight_expr)
                    .withColumn("item_key",F.concat_ws("_",F.col("cate_id").cast("string"),F.col("brand_id").cast("string")))
                    .groupBy("user_id","item_key").agg(F.sum("weight").alias("rating")))
    # ALS requires integer IDs. Index user and category-brand keys rather than
    # arithmetic-casting source identifiers (which can overflow on real Taobao IDs).
    user_map = (interactions.select("user_id").distinct()
                .withColumn("als_user_id",F.row_number().over(Window.orderBy("user_id")).cast("int")-1))
    item_map = (interactions.select("item_key").distinct()
                .withColumn("item_id",F.row_number().over(Window.orderBy("item_key")).cast("int")-1))
    interactions = (interactions.join(user_map,"user_id").join(item_map,"item_key")
                    .select("als_user_id","item_id","rating"))
    user_map.write.mode("overwrite").parquet(str(PROCESSED/"user_id_map.parquet"))
    item_map.write.mode("overwrite").parquet(str(PROCESSED/"item_id_map.parquet"))
    interactions.write.mode("overwrite").parquet(str(PROCESSED/"user_item_interactions.parquet"))

    # CTR training frame. Behaviour-derived user/category affinity uses only historical behavior.
    affinity = (behavior.groupBy("user_id","cate_id").agg(
        F.count("*").alias("user_category_events"),
        F.sum(F.when(F.col("btag")=="buy",1).otherwise(0)).alias("user_category_buys"),
        F.sum(F.when(F.col("btag")=="cart",1).otherwise(0)).alias("user_category_carts")))
    joined = (raw.join(ads,"adgroup_id","left").join(users,"user_id","left")
              .join(affinity,(raw.user_id==affinity.user_id)&(ads.cate_id==affinity.cate_id),"left"))
    # Resolve duplicate join keys and fill missing behavior aggregates.
    for c in ["user_category_events","user_category_buys","user_category_carts"]:
        joined = joined.withColumn(c,F.coalesce(F.col(c),F.lit(0)))
    joined = joined.withColumn("pid",F.coalesce(F.col("pid").cast("string"),F.lit("unknown")))
    joined = joined.drop(affinity.user_id).drop(affinity.cate_id)
    joined.write.mode("overwrite").parquet(str(PROCESSED/"ctr_features.parquet"))

    # Also persist cleaned source tables for auditability.
    raw.write.mode("overwrite").parquet(str(PROCESSED/"clean_raw_sample.parquet"))
    ads.write.mode("overwrite").parquet(str(PROCESSED/"clean_ad_feature.parquet"))
    users.write.mode("overwrite").parquet(str(PROCESSED/"clean_user_profile.parquet"))
    behavior.write.mode("overwrite").parquet(str(PROCESSED/"clean_behavior_log.parquet"))
    print("Preprocessing complete.")
    for name,df in [("impressions",raw),("ads",ads),("users",users),("behaviors",behavior),("ALS interactions",interactions),("CTR features",joined)]:
        print(f"{name}: {df.count():,} rows")
    spark.stop()

if __name__ == "__main__":
    main()
