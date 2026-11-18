import os
import sys
from pathlib import Path

# Ensure SPARK_HOME does not point to an external mismatched Spark version
if "SPARK_HOME" in os.environ:
    del os.environ["SPARK_HOME"]

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
WEIGHTS = {"pv": 1.0, "fav": 3.0, "cart": 4.0, "buy": 5.0}

# Ensure results and processed directories exist
PROCESSED.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)
