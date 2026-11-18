import csv
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
for name in ["raw_sample.csv","ad_feature.csv","user_profile.csv","behavior_log.csv"]:
    p=ROOT/"data"/"raw"/name
    with p.open(encoding="utf-8",newline="") as f:
        reader=csv.reader(f)
        header=next(reader)
        count=sum(1 for _ in reader)
    print(f"{name}: {count:,} data rows")
    print("  columns:", ", ".join(header))
