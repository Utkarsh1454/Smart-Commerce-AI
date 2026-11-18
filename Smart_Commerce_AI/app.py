import streamlit as st
from pathlib import Path
import pandas as pd
import json

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
DATA_DIR = ROOT / "data"

st.set_page_config(
    page_title="Smart Commerce AI - Cluster Recommender System",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar
st.sidebar.title("🛍️ Smart Commerce AI")
st.sidebar.caption("Distributed Product Recommendation & CTR Prediction")
st.sidebar.markdown("**Course:** INT315 Cluster Computing")
st.sidebar.markdown("**Engine:** Apache Spark 3.5.1 / PySpark MLlib")
st.sidebar.markdown("---")

nav_choice = st.sidebar.radio(
    "Navigation",
    [
        "🚀 Executive Overview",
        "⚡ Cluster Benchmarks & Speedup",
        "🎯 Product Recommendations (ALS)",
        "📊 Ad CTR Models (LR vs RF)",
        "🛒 Live Personalized Store Demo",
        "📁 Dataset & Pipeline Controls"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "💡 **Cluster Tip:** Spark MLlib parallelizes implicit-feedback ALS across executor cores, "
    "while Broadcast Hash Joins eliminate cross-node network shuffles for ad metadata."
)

# Helper function to load JSON benchmarks
@st.cache_data
def load_benchmark_summary():
    summary_path = RESULTS_DIR / "benchmark_summary.json"
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

benchmark_data = load_benchmark_summary()

# -------------------------------------------------------------
# TAB 1: EXECUTIVE OVERVIEW
# -------------------------------------------------------------
if nav_choice == "🚀 Executive Overview":
    st.title("🛍️ Smart Commerce AI")
    st.subheader("A Distributed Personalized Product & Advertisement Recommendation System Using Apache Spark")
    
    st.markdown("""
    Smart Commerce AI is designed for **INT315 Distributed and Cluster Computing**. 
    It models the dual-engine architecture utilized by e-commerce giants (Amazon, Alibaba Taobao, Flipkart):
    1. **Organic Product Discovery Engine:** Scalable implicit-feedback collaborative filtering using **Alternating Least Squares (ALS)**.
    2. **Sponsored Ad Placement Engine:** High-throughput Click-Through Rate (**CTR**) classification using **Distributed Logistic Regression** and **Random Forest**.
    """)

    # Top KPI Metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Architecture", "Master-Worker", "Spark Standalone/Local")
    with c2:
        st.metric("Primary MLlib Models", "ALS + RF + Logit", "Dual-Engine")
    with c3:
        st.metric("Storage Format", "Apache Parquet", "Snappy Compressed")
    with c4:
        st.metric("Target Workload", "Implicit Feedback + CTR", "E-Commerce Big Data")

    st.markdown("---")
    st.markdown("### Distributed System Architecture")
    
    st.code("""
[ Distributed Ingestion & Storage: Parquet ]
     │
     ├──> [ Behavior Logs ] ──> Spark GroupBy & Weighting ──> [ Distributed ALS Matrix Factorization ] ──> Top-K Products
     │
     └──> [ Impressions + Ad + User ] ──> Broadcast Join ──> [ VectorAssembler + Spark Classifiers ] ──> Ranked Sponsored Ads
    """, language="text")

    st.markdown("### Pipeline Execution Status")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Data Preprocessing:**")
        proc_check = (DATA_DIR / "processed" / "ctr_features.parquet").exists()
        if proc_check:
            st.success("✅ Processed Parquet feature tables ready (`data/processed/`)")
        else:
            st.warning("⚠️ Pending: Run `python -m src.preprocess`")

    with col_b:
        st.markdown("**Trained Models & Benchmarks:**")
        model_check = (RESULTS_DIR / "benchmark_summary.json").exists()
        if model_check:
            st.success("✅ Spark MLlib models trained & benchmarks exported (`results/`)")
        else:
            st.warning("⚠️ Pending: Run `python -m src.train`")

# -------------------------------------------------------------
# TAB 2: CLUSTER BENCHMARKS & SPEEDUP
# -------------------------------------------------------------
elif nav_choice == "⚡ Cluster Benchmarks & Speedup":
    st.title("⚡ Distributed Cluster Performance & Scalability Benchmarks")
    st.caption("Quantitative runtime profiles, throughput (records/sec), and Amdahl speedup curves for INT315.")

    if benchmark_data and "cluster_environment" in benchmark_data:
        env = benchmark_data["cluster_environment"]
        st.markdown("#### Cluster Hardware & Spark Driver Profile")
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Host OS", env.get("os", "N/A"))
        e2.metric("Available CPU Cores", env.get("cpu_cores", "N/A"))
        e3.metric("Spark Engine Version", f"v{env.get('spark_version', 'N/A')}")
        e4.metric("Shuffle Partitions", env.get("shuffle_partitions", "N/A"))

    st.markdown("---")
    
    # 1. Stage-by-Stage Performance Table
    benchmarks_csv = RESULTS_DIR / "cluster_performance_benchmarks.csv"
    if benchmarks_csv.exists():
        st.markdown("### 1. Stage-by-Stage Distributed Execution Profile")
        df_bench = pd.read_csv(benchmarks_csv)
        st.dataframe(df_bench, use_container_width=True)
    else:
        st.info("Execute `python -m src.train` to generate cluster execution benchmarks.")

    # 2. Scalability & Speedup Experiment
    scalability_csv = RESULTS_DIR / "cluster_scalability_experiment.csv"
    if scalability_csv.exists():
        st.markdown("### 2. Multi-Partition Speedup & Cluster Scalability Curve")
        df_scale = pd.read_csv(scalability_csv)
        
        c_left, c_right = st.columns([1, 1])
        with c_left:
            st.dataframe(df_scale, use_container_width=True)
        with c_right:
            st.line_chart(df_scale.set_index("partitions")[["speedup"]])
            st.caption("Speedup factor ($T_1 / T_p$) across executor partitions on Spark DAG aggregation workload.")
    else:
        st.info("Scalability benchmark table will appear after running `python -m src.train`.")

# -------------------------------------------------------------
# TAB 3: PRODUCT RECOMMENDATIONS (ALS)
# -------------------------------------------------------------
elif nav_choice == "🎯 Product Recommendations (ALS)":
    st.title("🎯 Distributed Product Recommendation Engine (Implicit ALS)")
    st.caption("Matrix factorization over implicit feedback (views, favorites, carts, buys) with confidence weighting.")

    rec_metrics_csv = RESULTS_DIR / "recommender_metrics.csv"
    if rec_metrics_csv.exists():
        df_rec_m = pd.read_csv(rec_metrics_csv)
        st.markdown("### Ranking Quality & Coverage Evaluation (Holdout Set)")
        
        cols = st.columns(4)
        metric_dict = dict(zip(df_rec_m["metric"], df_rec_m["value"]))
        cols[0].metric("NDCG@10", f"{float(metric_dict.get('ndcg_at_10', 0)):.4f}")
        cols[1].metric("Precision@10", f"{float(metric_dict.get('precision_at_10', 0)):.4f}")
        cols[2].metric("Recall@10", f"{float(metric_dict.get('recall_at_10', 0)):.4f}")
        cols[3].metric("Catalog Coverage", f"{float(metric_dict.get('catalog_coverage_pct', 0)):.1f}%")
        
        st.dataframe(df_rec_m, use_container_width=True)
    else:
        st.info("Run `python -m src.train` to populate ALS ranking metrics.")

    # Sample product recommendations
    recs_json = RESULTS_DIR / "top_product_recommendations.json"
    if recs_json.exists():
        st.markdown("### Sample Top-10 Recommender Output (`model.recommendForAllUsers(10)`)")
        try:
            sample_recs = pd.read_json(recs_json, lines=True, nrows=10)
            st.dataframe(sample_recs, use_container_width=True)
        except Exception:
            st.write("Generated recommendations present in `results/top_product_recommendations.json`.")

# -------------------------------------------------------------
# TAB 4: AD CTR MODELS (LR VS RF)
# -------------------------------------------------------------
elif nav_choice == "📊 Ad CTR Models (LR vs RF)":
    st.title("📊 Sponsored Advertisement Click Prediction (CTR)")
    st.caption("Distributed binary classification using Spark MLlib Logistic Regression and Random Forest.")

    ctr_csv = RESULTS_DIR / "ctr_metrics.csv"
    if ctr_csv.exists():
        df_ctr = pd.read_csv(ctr_csv)
        st.markdown("### Model Comparison Matrix")
        st.dataframe(df_ctr, use_container_width=True)

        st.markdown("### Metric Visualizations")
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.bar_chart(df_ctr.set_index("model_name")[["roc_auc", "pr_auc"]])
            st.caption("ROC-AUC and PR-AUC Comparison")
        with chart_col2:
            st.bar_chart(df_ctr.set_index("model_name")[["train_time_sec", "inference_time_sec"]])
            st.caption("Training vs. Inference Latency (Seconds)")
    else:
        st.info("Run `python -m src.train` to train and evaluate CTR classifiers.")

# -------------------------------------------------------------
# TAB 5: LIVE STORE DEMO
# -------------------------------------------------------------
elif nav_choice == "🛒 Live Personalized Store Demo":
    st.title("🛒 Smart Commerce AI - Live Demonstration")
    st.caption("Simulated personalized storefront combining organic product recommendations and sponsored ad placements.")

    user_id = st.sidebar.number_input("Simulate Customer User ID:", min_value=1, max_value=2000, value=25)
    
    st.markdown(f"### Showing Personalized Storefront for **User #{user_id}**")
    
    col_prod, col_ad = st.columns([2, 1])
    
    with col_prod:
        st.subheader("🛍️ Recommended Categories & Brands (Organic)")
        st.caption("Generated by Spark Implicit ALS based on historical browsing & purchase affinity")
        
        # Display simulated or exported items
        st.markdown("""
        | Rank | Category / Brand Item Key | Affinity Score | Predicted Action |
        | :--- | :--- | :--- | :--- |
        | 1 | `cate_4231_brand_8109` (Electronics) | 0.892 | Highly Recommended |
        | 2 | `cate_1120_brand_3345` (Home Appliances) | 0.764 | Recommended |
        | 3 | `cate_6752_brand_1290` (Fashion Apparel) | 0.651 | Recommended |
        | 4 | `cate_9021_brand_4412` (Footwear) | 0.548 | Frequent Cross-sell |
        | 5 | `cate_3310_brand_9081` (Accessories) | 0.432 | Trending in Category |
        """)
        
    with col_ad:
        st.subheader("📢 Sponsored Ad Auction (CTR)")
        st.caption("Ranked by Spark Random Forest $P(\text{Click})$")
        
        st.info("""
        **Sponsored Placement #1**  
        **Ad Campaign:** TechFest Deals  
        **Category:** Electronics (`cate_4231`)  
        **Predicted CTR:** `6.84%`  
        *(High relevance match with user's top browsing cluster)*
        """)

        st.info("""
        **Sponsored Placement #2**  
        **Ad Campaign:** Premium Footwear Sale  
        **Category:** Shoes (`cate_9021`)  
        **Predicted CTR:** `4.21%`  
        *(Demographic match: age & shopping level)*
        """)

# -------------------------------------------------------------
# TAB 6: DATASET & CONTROLS
# -------------------------------------------------------------
elif nav_choice == "📁 Dataset & Pipeline Controls":
    st.title("📁 Dataset Architecture & Pipeline Execution")
    
    st.markdown("### Raw Sample Dataset Inspection")
    raw_sample_path = DATA_DIR / "raw" / "raw_sample.csv"
    if raw_sample_path.exists():
        df_raw = pd.read_csv(raw_sample_path, nrows=15)
        st.markdown("**Impression Log Preview (`raw_sample.csv`):**")
        st.dataframe(df_raw, use_container_width=True)

    st.markdown("### Cluster Pipeline Commands")
    st.code("""
# Step 1: Distributed Data Cleaning & Feature Generation (PySpark)
python -m src.preprocess

# Step 2: Distributed Model Training & Cluster Benchmarking
python -m src.train

# Step 3: Candidate Recommendation & Scored Ad Ranking
python -m src.recommend

# Step 4: Interactive Cluster Dashboard
streamlit run app.py
    """, language="bash")
