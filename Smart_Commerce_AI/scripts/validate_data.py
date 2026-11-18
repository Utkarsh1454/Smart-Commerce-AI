import csv
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
checks={
 "raw_sample.csv":["user_id","adgroup_id","time_stamp","pid","nonclk","clk"],
 "ad_feature.csv":["adgroup_id","cate_id","campaign_id","customer_id","brand_id","price"],
 "user_profile.csv":["user_id","cms_segid","cms_group_id","final_gender_code","age_level","pvalue_level","shopping_level","occupation","new_user_class_level"],
 "behavior_log.csv":["user_id","time_stamp","btag","cate_id","brand_id"]
}
for name,expected in checks.items():
 p=ROOT/"data"/"raw"/name
 with p.open(encoding="utf-8",newline="") as f:
  r=csv.DictReader(f)
  missing=[c for c in expected if c not in (r.fieldnames or [])]
  count=sum(1 for _ in r)
 if missing: raise SystemExit(f"{name}: missing columns {missing}")
 print(f"OK {name}: {count:,} rows")
print("All required demo files passed basic schema validation.")
