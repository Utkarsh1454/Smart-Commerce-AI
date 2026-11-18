# TaobaoAd_x1 Manageable Dataset Kit

This kit creates a real, manageable sample from the genuine Alibaba TaobaoAd_x1 dataset.

**It does not contain synthetic records or pretend to contain the source data.** The script downloads the genuine 2.44 GB source ZIP from the RecZoo Hugging Face repository, then samples 50,000 training and 10,000 test rows by default while preserving the official train/test split.

## Requirements
- Python 3.9+
- Stable internet connection
- At least 3 GB free disk space during download/extraction
- `pip install pandas requests`

## Run (Windows PowerShell)
```powershell
py -m pip install pandas requests
py scripts\download_and_sample.py --train-rows 50000 --test-rows 10000
```

Output:
```text
TaobaoAd_Manageable/
  train_sample.csv
  test_sample.csv
  dataset_info.json
  README.md
```

The full source archive is deleted after sampling to save disk space. Add `--keep-archive` to retain it. Use `--train-rows` and `--test-rows` to change sample sizes.

## Dataset source
https://huggingface.co/datasets/reczoo/TaobaoAd_x1

Dataset card / schema:
https://huggingface.co/datasets/reczoo/TaobaoAd_x1/blob/main/README.md

Original Alibaba Tianchi source cited by dataset card:
https://tianchi.aliyun.com/dataset/dataDetail?dataId=56

## Provenance and methodology
- Genuine records are downloaded from the RecZoo-hosted TaobaoAd_x1 archive.
- The source card reports 21,929,911 training rows and 3,099,515 test rows.
- Sample selection is deterministic and random within sequential chunks.
- Train and test remain separate.
- The script checks for `train.csv` and `test.csv` in the source archive and records actual observed columns in `dataset_info.json`.
