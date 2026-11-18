"""
Smart Commerce AI - Distributed Model Training & Cluster Performance Benchmarking
Course: INT315 - Distributed and Cluster Computing
Component: Model Training (Implicit ALS & CTR Classifiers) with System Profiling
"""

import os
import sys
import time
import json
import platform
from pathlib import Path
import math

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import DoubleType, IntegerType, ArrayType
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, Imputer
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
    RegressionEvaluator,
    RankingEvaluator
)
from pyspark.ml.functions import vector_to_array

from src.config import PROCESSED, MODELS, RESULTS

# Ensure results directory exists
RESULTS.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)


class ExecutionTimer:
    """Utility to measure execution time and compute throughput."""
    def __init__(self, task_name, record_count=0):
        self.task_name = task_name
        self.record_count = record_count
        self.start_time = None
        self.duration = 0.0

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.duration = time.time() - self.start_time
        throughput = (self.record_count / self.duration) if (self.duration > 0 and self.record_count > 0) else 0.0
        print(f"[BENCHMARK] {self.task_name}: {self.duration:.3f}s ({throughput:,.1f} records/sec)")


def collect_cluster_metadata(spark):
    """Collects distributed execution runtime metadata."""
    sc = spark.sparkContext
    metadata = {
        "os": f"{platform.system()} {platform.release()}",
        "cpu_cores": os.cpu_count(),
        "python_version": sys.version.split()[0],
        "spark_version": spark.version,
        "spark_master": sc.master,
        "default_parallelism": sc.defaultParallelism,
        "shuffle_partitions": spark.conf.get("spark.sql.shuffle.partitions", "200"),
        "app_id": sc.applicationId,
    }
    return metadata


def evaluate_ranking_metrics(recs_df, ground_truth_df, total_catalog_items, k_list=[5, 10]):
    """
    Computes precision@k, recall@k, ndcg@k, MAP@k, and catalog coverage using PySpark.
    recs_df: DataFrame with ['als_user_id', 'recommendations' (array of struct item_id, rating)]
    ground_truth_df: DataFrame with ['als_user_id', 'items' (array of item_id)]
    """
    # Extract predicted item_id list
    predictions_prepared = recs_df.withColumn(
        "prediction",
        F.expr("transform(recommendations, x -> cast(x.item_id as double))")
    ).select("als_user_id", "prediction")

    # Join predictions with ground truth
    ranking_data = predictions_prepared.join(ground_truth_df, "als_user_id", "inner")
    ranking_data = ranking_data.withColumn(
        "label",
        F.expr("transform(items, x -> cast(x as double))")
    ).select("prediction", "label")

    ranking_data.cache()
    metrics = {}

    for k in k_list:
        ndcg_eval = RankingEvaluator(metricName="ndcgAtK", k=k, predictionCol="prediction", labelCol="label")
        prec_eval = RankingEvaluator(metricName="precisionAtK", k=k, predictionCol="prediction", labelCol="label")
        rec_eval = RankingEvaluator(metricName="recallAtK", k=k, predictionCol="prediction", labelCol="label")
        map_eval = RankingEvaluator(metricName="meanAveragePrecisionAtK", k=k, predictionCol="prediction", labelCol="label")

        metrics[f"ndcg_at_{k}"] = float(ndcg_eval.evaluate(ranking_data))
        metrics[f"precision_at_{k}"] = float(prec_eval.evaluate(ranking_data))
        metrics[f"recall_at_{k}"] = float(rec_eval.evaluate(ranking_data))
        metrics[f"map_at_{k}"] = float(map_eval.evaluate(ranking_data))

    # Calculate Catalog Coverage: percentage of distinct items recommended across all users
    distinct_recs = (
        recs_df.select(F.explode("recommendations").alias("rec"))
        .select(F.col("rec.item_id").alias("item_id"))
        .distinct()
        .count()
    )
    metrics["catalog_coverage_pct"] = (distinct_recs / total_catalog_items * 100.0) if total_catalog_items > 0 else 0.0
    metrics["distinct_items_recommended"] = distinct_recs
    metrics["total_catalog_items"] = total_catalog_items

    ranking_data.unpersist()
    return metrics


def train_als(spark, benchmark_records):
    """
    Trains implicit-feedback ALS and evaluates ranking quality and cluster execution metrics.
    """
    print("\n" + "="*70)
    print("STAGE 1: DISTRIBUTED PRODUCT RECOMMENDER (IMPLICIT ALS)")
    print("="*70)

    interactions_path = PROCESSED / "user_item_interactions.parquet"
    if not interactions_path.exists():
        raise FileNotFoundError(f"Missing {interactions_path}. Run preprocess.py first.")

    interactions = spark.read.parquet(str(interactions_path))
    total_interactions = interactions.count()
    total_users = interactions.select("als_user_id").distinct().count()
    total_items = interactions.select("item_id").distinct().count()

    print(f"Loaded {total_interactions:,} interactions across {total_users:,} users and {total_items:,} category-brand items.")

    train, test = interactions.randomSplit([0.8, 0.2], seed=42)
    train_count = train.count()
    test_count = test.count()

    # Model training with timing
    als = ALS(
        userCol="als_user_id",
        itemCol="item_id",
        ratingCol="rating",
        rank=12,
        maxIter=10,
        regParam=0.08,
        coldStartStrategy="drop",
        nonnegative=True,
        implicitPrefs=True,
        alpha=20.0
    )

    t0 = time.time()
    model = als.fit(train)
    als_train_time = time.time() - t0
    benchmark_records.append({
        "stage": "ALS Model Training (Implicit Feedback)",
        "records_in": train_count,
        "partitions": train.rdd.getNumPartitions(),
        "execution_time_sec": round(als_train_time, 3),
        "throughput_records_sec": round(train_count / als_train_time, 1) if als_train_time > 0 else 0,
        "status": "COMPLETED"
    })

    # Diagnostic RMSE evaluation
    pred = model.transform(test)
    rmse_eval = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
    rmse = rmse_eval.evaluate(pred)

    # Save model
    model.write().overwrite().save(str(MODELS / "als_recommender"))

    # Top-10 Recommendations generation with timing
    t0 = time.time()
    all_recs = model.recommendForAllUsers(10)
    all_recs.cache()
    rec_count = all_recs.count()
    rec_gen_time = time.time() - t0
    benchmark_records.append({
        "stage": "Top-10 Recommendation Generation (All Users)",
        "records_in": total_users,
        "partitions": all_recs.rdd.getNumPartitions(),
        "execution_time_sec": round(rec_gen_time, 3),
        "throughput_records_sec": round(total_users / rec_gen_time, 1) if rec_gen_time > 0 else 0,
        "status": "COMPLETED"
    })

    # Save top recommendations JSON
    all_recs.write.mode("overwrite").json(str(RESULTS / "top_product_recommendations.json"))

    # Prepare ground-truth test items for ranking evaluation
    test_ground_truth = (
        test.groupBy("als_user_id")
        .agg(F.collect_list("item_id").alias("items"))
    )

    # Subsample users present in test set for ranking evaluation
    test_user_recs = all_recs.join(test_ground_truth.select("als_user_id"), "als_user_id", "inner")
    ranking_metrics = evaluate_ranking_metrics(test_user_recs, test_ground_truth, total_items, k_list=[5, 10])
    ranking_metrics["implicit_rmse_diagnostic"] = float(rmse)
    ranking_metrics["train_time_sec"] = round(als_train_time, 3)
    ranking_metrics["rec_gen_time_sec"] = round(rec_gen_time, 3)

    all_recs.unpersist()

    print("\n--- Product Recommendation Evaluation Results ---")
    for k, v in ranking_metrics.items():
        print(f"  {k}: {v if isinstance(v, int) else f'{v:.4f}'}")

    # Export recommender metrics CSV
    recommender_rows = [
        {"model": "Implicit ALS (Rank=12, Reg=0.08, Alpha=20.0)", "metric": k, "value": v}
        for k, v in ranking_metrics.items()
    ]
    with open(RESULTS / "recommender_metrics.csv", "w", encoding="utf-8") as f:
        f.write("model,metric,value\n")
        for row in recommender_rows:
            f.write(f'"{row["model"]}","{row["metric"]}",{row["value"]}\n')

    return ranking_metrics


def compute_log_loss(predictions_df, prob_col="probability", label_col="label", eps=1e-15):
    """Computes binary cross-entropy (Log Loss) from probability vector."""
    log_loss_df = predictions_df.withColumn(
        "p",
        vector_to_array(F.col(prob_col))[1]
    ).withColumn(
        "clipped_p",
        F.when(F.col("p") < eps, eps).when(F.col("p") > 1.0 - eps, 1.0 - eps).otherwise(F.col("p"))
    ).withColumn(
        "loss_term",
        -(F.col(label_col) * F.log(F.col("clipped_p")) + (1.0 - F.col(label_col)) * F.log(1.0 - F.col("clipped_p")))
    )
    result = log_loss_df.select(F.avg("loss_term")).collect()[0][0]
    return float(result) if result is not None else 0.0


def train_ctr(spark, benchmark_records):
    """
    Trains and compares distributed Logistic Regression and Random Forest CTR models,
    measuring full classification metrics (ROC-AUC, PR-AUC, Accuracy, Precision, Recall, F1, Log-Loss)
    and execution performance.
    """
    print("\n" + "="*70)
    print("STAGE 2: DISTRIBUTED ADVERTISEMENT CLICK PREDICTION (CTR)")
    print("="*70)

    ctr_path = PROCESSED / "ctr_features.parquet"
    if not ctr_path.exists():
        raise FileNotFoundError(f"Missing {ctr_path}. Run preprocess.py first.")

    df = spark.read.parquet(str(ctr_path))
    total_records = df.count()
    print(f"Loaded CTR dataset: {total_records:,} impression records.")

    numeric = [
        "cms_segid", "cms_group_id", "final_gender_code", "age_level", "pvalue_level",
        "shopping_level", "occupation", "new_user_class_level", "cate_id", "campaign_id",
        "customer_id", "brand_id", "price", "user_category_events", "user_category_buys",
        "user_category_carts"
    ]
    available = [c for c in numeric if c in df.columns]
    for c in available:
        df = df.withColumn(c, F.col(c).cast("double"))
    df = df.withColumn("label", F.col("clk").cast("double"))

    imputer = Imputer(inputCols=available, outputCols=[c + "_imp" for c in available]).setStrategy("median")
    pid_index = StringIndexer(inputCol="pid", outputCol="pid_idx", handleInvalid="keep")
    assembler = VectorAssembler(
        inputCols=[c + "_imp" for c in available] + ["pid_idx"],
        outputCol="features",
        handleInvalid="keep"
    )

    train, test = df.randomSplit([0.8, 0.2], seed=42)
    train.cache()
    test.cache()
    train_count = train.count()
    test_count = test.count()

    roc_eval = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
    pr_eval = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderPR")
    acc_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    f1_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
    prec_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
    rec_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall")

    classifiers = [
        ("logistic_regression", "Distributed Logistic Regression (ElasticNet)", LogisticRegression(
            labelCol="label", featuresCol="features", maxIter=30, regParam=0.05
        )),
        ("random_forest", "Distributed Random Forest (60 Trees, Depth 8)", RandomForestClassifier(
            labelCol="label", featuresCol="features", numTrees=60, maxDepth=8, seed=42
        ))
    ]

    ctr_results = []

    for model_key, display_name, clf in classifiers:
        print(f"\nTraining {display_name}...")
        pipeline = Pipeline(stages=[imputer, pid_index, assembler, clf])

        # Benchmark training time
        t0 = time.time()
        fitted_model = pipeline.fit(train)
        train_time = time.time() - t0

        benchmark_records.append({
            "stage": f"CTR Training: {display_name}",
            "records_in": train_count,
            "partitions": train.rdd.getNumPartitions(),
            "execution_time_sec": round(train_time, 3),
            "throughput_records_sec": round(train_count / train_time, 1) if train_time > 0 else 0,
            "status": "COMPLETED"
        })

        # Benchmark inference / scoring time
        t0 = time.time()
        predictions = fitted_model.transform(test)
        predictions.cache()
        test_eval_count = predictions.count()
        inference_time = time.time() - t0

        benchmark_records.append({
            "stage": f"CTR Inference: {display_name}",
            "records_in": test_count,
            "partitions": test.rdd.getNumPartitions(),
            "execution_time_sec": round(inference_time, 3),
            "throughput_records_sec": round(test_count / inference_time, 1) if inference_time > 0 else 0,
            "status": "COMPLETED"
        })

        # Evaluate all metrics
        roc_auc = float(roc_eval.evaluate(predictions))
        pr_auc = float(pr_eval.evaluate(predictions))
        acc = float(acc_eval.evaluate(predictions))
        f1 = float(f1_eval.evaluate(predictions))
        prec = float(prec_eval.evaluate(predictions))
        rec = float(rec_eval.evaluate(predictions))
        logloss = compute_log_loss(predictions, prob_col="probability", label_col="label")

        # Save model artifact
        fitted_model.write().overwrite().save(str(MODELS / model_key))

        model_summary = {
            "model_key": model_key,
            "model_name": display_name,
            "roc_auc": round(roc_auc, 5),
            "pr_auc": round(pr_auc, 5),
            "accuracy": round(acc, 5),
            "precision": round(prec, 5),
            "recall": round(rec, 5),
            "f1_score": round(f1, 5),
            "log_loss": round(logloss, 5),
            "train_time_sec": round(train_time, 3),
            "inference_time_sec": round(inference_time, 3),
            "inference_records_sec": round(test_count / inference_time, 1) if inference_time > 0 else 0
        }
        ctr_results.append(model_summary)

        print(f"  ROC-AUC:  {roc_auc:.4f} | PR-AUC: {pr_auc:.4f} | F1: {f1:.4f} | LogLoss: {logloss:.4f}")
        print(f"  Train:    {train_time:.2f}s ({train_count/train_time:,.0f} rec/s)")
        print(f"  Inference:{inference_time:.2f}s ({test_count/inference_time:,.0f} rec/s)")

        predictions.unpersist()

    train.unpersist()
    test.unpersist()

    # Save detailed CTR metrics CSV
    with open(RESULTS / "ctr_metrics.csv", "w", encoding="utf-8") as f:
        headers = ["model_key", "model_name", "roc_auc", "pr_auc", "accuracy", "precision", "recall", "f1_score", "log_loss", "train_time_sec", "inference_time_sec"]
        f.write(",".join(headers) + "\n")
        for r in ctr_results:
            row_str = ",".join(str(r[h]) if not isinstance(r[h], str) or "," not in str(r[h]) else f'"{r[h]}"' for h in headers)
            f.write(row_str + "\n")

    return ctr_results


def run_scalability_experiment(spark, benchmark_records):
    """
    Cluster Scalability Experiment: Evaluates distributed execution time and speedup factor
    across varying shuffle partition counts (1, 2, 4, 8) to model cluster scaling efficiency.
    Directly provides data for Section 6/7 scalability graphs in the paper.
    """
    print("\n" + "="*70)
    print("STAGE 3: CLUSTER PARTITIONING & SPEEDUP SCALABILITY BENCHMARK")
    print("="*70)

    ctr_path = PROCESSED / "ctr_features.parquet"
    if not ctr_path.exists():
        return []

    base_df = spark.read.parquet(str(ctr_path))
    partition_configs = [1, 2, 4, 8]
    scalability_results = []
    base_time = None

    for p in partition_configs:
        repartitioned = base_df.repartition(p)
        repartitioned.cache()
        t0 = time.time()
        # Representative cluster workload: wide shuffle group-by aggregation
        agg_result = (
            repartitioned.groupBy("cate_id")
            .agg(
                F.count("*").alias("count"),
                F.avg("price").alias("avg_price"),
                F.sum("clk").alias("total_clicks")
            )
            .collect()
        )
        elapsed = time.time() - t0
        repartitioned.unpersist()

        if base_time is None or p == 1:
            base_time = elapsed
        speedup = base_time / elapsed if elapsed > 0 else 1.0
        efficiency_pct = (speedup / p) * 100.0

        res = {
            "partitions": p,
            "execution_time_sec": round(elapsed, 4),
            "speedup": round(speedup, 3),
            "efficiency_pct": round(efficiency_pct, 1),
            "throughput_rows_sec": round(len(agg_result) / elapsed, 1) if elapsed > 0 else 0
        }
        scalability_results.append(res)
        print(f"  Partitions: {p:2d} | Runtime: {elapsed:.4f}s | Speedup: {speedup:.2f}x | Efficiency: {efficiency_pct:.1f}%")

    with open(RESULTS / "cluster_scalability_experiment.csv", "w", encoding="utf-8") as f:
        f.write("partitions,execution_time_sec,speedup,efficiency_pct\n")
        for r in scalability_results:
            f.write(f"{r['partitions']},{r['execution_time_sec']},{r['speedup']},{r['efficiency_pct']}\n")

    return scalability_results


def export_full_benchmark_summary(cluster_meta, benchmark_records, als_metrics, ctr_metrics, scalability_results):
    """Exports structured benchmark summary JSON and CSV for paper insertion and dashboard."""
    # Write cluster_performance_benchmarks.csv
    with open(RESULTS / "cluster_performance_benchmarks.csv", "w", encoding="utf-8") as f:
        headers = ["stage", "records_in", "partitions", "execution_time_sec", "throughput_records_sec", "status"]
        f.write(",".join(headers) + "\n")
        for r in benchmark_records:
            f.write(f'"{r["stage"]}",{r["records_in"]},{r["partitions"]},{r["execution_time_sec"]},{r["throughput_records_sec"]},"{r["status"]}"\n')

    # Comprehensive JSON document
    summary = {
        "project": "Smart Commerce AI",
        "cluster_environment": cluster_meta,
        "execution_benchmarks": benchmark_records,
        "product_recommendation_als": als_metrics,
        "ctr_models": ctr_metrics,
        "scalability_speedup": scalability_results,
        "generated_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(RESULTS / "benchmark_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*70)
    print(f"ALL BENCHMARK TABLES SUCCESSFULLY EXPORTED TO {RESULTS}")
    print("Files created:")
    print(f"  1. {RESULTS / 'cluster_performance_benchmarks.csv'}")
    print(f"  2. {RESULTS / 'ctr_metrics.csv'}")
    print(f"  3. {RESULTS / 'recommender_metrics.csv'}")
    print(f"  4. {RESULTS / 'cluster_scalability_experiment.csv'}")
    print(f"  5. {RESULTS / 'benchmark_summary.json'}")
    print("="*70 + "\n")


def main():
    spark = (
        SparkSession.builder.appName("SmartCommerce-ModelTraining-Benchmarking")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    cluster_meta = collect_cluster_metadata(spark)
    print("Cluster Execution Environment:")
    for k, v in cluster_meta.items():
        print(f"  {k}: {v}")

    benchmark_records = []

    # 1. Product recommendation training & ranking evaluation
    als_metrics = train_als(spark, benchmark_records)

    # 2. CTR model training & binary classification evaluation
    ctr_metrics = train_ctr(spark, benchmark_records)

    # 3. Scalability speedup experiment across partition counts
    scalability_results = run_scalability_experiment(spark, benchmark_records)

    # 4. Export all benchmark tables
    export_full_benchmark_summary(cluster_meta, benchmark_records, als_metrics, ctr_metrics, scalability_results)

    spark.stop()


if __name__ == "__main__":
    main()
