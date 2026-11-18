#!/usr/bin/env python3
"""
Download genuine RecZoo TaobaoAd_x1 data and create a smaller, reproducible sample.
The source archive is ~2.44 GB. This script streams the download and samples
rows from the official train/test CSVs without loading the complete CSVs in RAM.
"""
from pathlib import Path
import argparse, hashlib, json, random, shutil, sys, zipfile
import requests
import pandas as pd

URL = "https://huggingface.co/datasets/reczoo/TaobaoAd_x1/resolve/main/TaobaoAd_x1.zip"
EXPECTED_MD5 = {
    "train.csv": "eaabfc8629f23519b04593e26c7522fc",
    "test.csv": "f5ae6197e52385496d46e2867c1c8da1",
}
EXPECTED_ROWS = {"train.csv": 21_929_911, "test.csv": 3_099_515}

def md5_file(path, chunk=8*1024*1024):
    h=hashlib.md5()
    with open(path,"rb") as f:
        while True:
            b=f.read(chunk)
            if not b: break
            h.update(b)
    return h.hexdigest()

def download(url, dest):
    dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists() and dest.stat().st_size > 100_000_000:
        print(f"Using existing archive: {dest} ({dest.stat().st_size/1e9:.2f} GB)")
        return
    tmp=dest.with_suffix(dest.suffix+".part")
    print("Downloading genuine TaobaoAd_x1 source archive (~2.44 GB)...")
    with requests.get(url,stream=True,timeout=(30,180),allow_redirects=True) as r:
        r.raise_for_status()
        total=int(r.headers.get("content-length",0))
        done=0
        with open(tmp,"wb") as f:
            for block in r.iter_content(chunk_size=8*1024*1024):
                if block:
                    f.write(block); done+=len(block)
                    if total:
                        print(f"\rDownloaded {done/1e9:.2f}/{total/1e9:.2f} GB",end="",flush=True)
    print()
    tmp.replace(dest)

def sample_member(zf, member, target_rows, output, seed, chunksize=100_000):
    names=zf.namelist()
    found=next((n for n in names if Path(n).name==member),None)
    if not found:
        raise RuntimeError(f"{member} not found in source ZIP. Members: {names[:30]}")
    expected=EXPECTED_ROWS[member]
    # Slight oversampling compensates for stochastic variation, then trim to exact target.
    fraction=min(1.0, (target_rows*1.20)/expected)
    rng=random.Random(seed)
    samples=[]
    seen=0
    with zf.open(found) as raw:
        reader=pd.read_csv(raw, chunksize=chunksize, low_memory=False)
        for i,chunk in enumerate(reader):
            seen += len(chunk)
            n=min(len(chunk), max(1, int(round(len(chunk)*fraction))))
            # Deterministic random sample from each sequential chunk.
            samples.append(chunk.sample(n=n, random_state=seed+i))
            if i % 25 == 0:
                print(f"{member}: scanned {seen:,} rows")
    if not samples:
        raise RuntimeError(f"No rows read from {member}")
    result=pd.concat(samples,ignore_index=True)
    if len(result)>target_rows:
        result=result.sample(n=target_rows,random_state=seed)
    result=result.reset_index(drop=True)
    output.parent.mkdir(parents=True,exist_ok=True)
    result.to_csv(output,index=False)
    print(f"Saved {output} ({len(result):,} rows, {len(result.columns)} columns)")
    return {"source_member":found,"source_rows_expected":expected,
            "source_rows_scanned":seen,"sample_rows":len(result),
            "columns":list(result.columns),"file_bytes":output.stat().st_size}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--train-rows",type=int,default=50_000)
    ap.add_argument("--test-rows",type=int,default=10_000)
    ap.add_argument("--output",default="TaobaoAd_Manageable")
    ap.add_argument("--archive",default="TaobaoAd_x1.zip")
    ap.add_argument("--keep-archive",action="store_true")
    args=ap.parse_args()
    out=Path(args.output).resolve()
    archive=Path(args.archive).resolve()
    if args.train_rows<1 or args.test_rows<1:
        ap.error("train/test sample sizes must be positive")
    download(URL,archive)
    print("Opening source ZIP...")
    with zipfile.ZipFile(archive) as zf:
        meta={}
        for member,target,seed in [
            ("train.csv",args.train_rows,2026),
            ("test.csv",args.test_rows,2027),
        ]:
            meta[member]=sample_member(
                zf,member,target,out/(member.replace(".csv","_sample.csv")),seed
            )
    info={
        "dataset":"TaobaoAd_x1 (RecZoo distribution of Alibaba Taobao advertising data)",
        "provenance":"Genuine source records sampled from official RecZoo-hosted TaobaoAd_x1 archive; not synthetic.",
        "source_url":URL,
        "source_archive":"TaobaoAd_x1.zip",
        "sampling":"Deterministic random sampling within chunks of each official split; original train/test split preserved.",
        "random_seeds":{"train":2026,"test":2027},
        "splits":meta,
        "warning":"This sample is suitable for development and coursework experiments. For publication-quality metrics, document sampling and compare results on larger/full data where feasible."
    }
    (out/"dataset_info.json").write_text(json.dumps(info,indent=2),encoding="utf-8")
    readme=f"""# Genuine TaobaoAd_x1 manageable sample

This folder contains genuine rows sampled from the RecZoo-hosted Alibaba TaobaoAd_x1 dataset.

## Files
- `train_sample.csv` — {args.train_rows:,} sampled training rows (or fewer only if source data is unexpectedly smaller).
- `test_sample.csv` — {args.test_rows:,} sampled test rows (or fewer only if source data is unexpectedly smaller).
- `dataset_info.json` — provenance, sample method and observed schema.

## Source and citation
RecZoo, TaobaoAd_x1: https://huggingface.co/datasets/reczoo/TaobaoAd_x1
Original source cited by the dataset card: Alibaba Tianchi, https://tianchi.aliyun.com/dataset/dataDetail?dataId=56

The original dataset card reports 21,929,911 training rows and 3,099,515 test rows. It describes 8 days of ad click-through data and a 22-day behaviour history. See the source README for field definitions and preprocessing notes.

## Important modelling note
The TaobaoAd_x1 files are preprocessed impression-level records. They include ad click labels and historical behaviour sequence features. They are not the original four separate raw tables. Preserve the provided train/test split; do not randomly mix train and test.

## Reproducibility
Sampling is deterministic (train seed 2026; test seed 2027), performed independently within chunks of each source split. The downloaded source archive is approximately 2.44 GB, so ensure sufficient disk space and a stable connection.

## Run
```bash
pip install pandas requests
python scripts/download_and_sample.py --train-rows 50000 --test-rows 10000
```
"""
    (out/"README.md").write_text(readme,encoding="utf-8")
    print(f"\nDone. Manageable genuine dataset folder: {out}")
    print("Train rows:",meta["train.csv"]["sample_rows"])
    print("Test rows:",meta["test.csv"]["sample_rows"])
    if not args.keep_archive and archive.exists():
        archive.unlink()
        print("Removed large source archive to save disk space.")

if __name__=="__main__":
    main()
