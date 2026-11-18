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

# On Windows, paths with spaces or relative venv paths break Spark worker cmd invocation.
# Always bind PYSPARK_PYTHON to the absolute base Python path and inject venv site-packages into PYTHONPATH.
base_python = r"C:\Users\Himanshu\AppData\Local\Programs\Python\Python313\python.exe"
venv_site_packages = str(ROOT / ".venv" / "Lib" / "site-packages")

if Path(base_python).exists():
    os.environ["PYSPARK_PYTHON"] = base_python
    os.environ["PYSPARK_DRIVER_PYTHON"] = base_python
    existing_pp = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = f"{venv_site_packages};{str(ROOT)};{existing_pp}" if existing_pp else f"{venv_site_packages};{str(ROOT)}"
else:
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

# Ensure results and processed directories exist
PROCESSED.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)
