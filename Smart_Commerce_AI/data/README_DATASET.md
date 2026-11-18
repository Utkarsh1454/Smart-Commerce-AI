# Smart Commerce AI – Dataset package

## Important provenance notice
The CSVs in this package are **synthetic demonstration data**, generated for this project using a Taobao-shaped schema. They are NOT original Alibaba/Taobao records and must not be described as real customer data.

They are designed to make the project runnable immediately. To use genuine records, obtain the Alibaba Taobao dataset from the official source and replace the raw files while preserving the documented columns.

## Files
- `raw_sample.csv`: synthetic ad impressions and click labels.
- `ad_feature.csv`: synthetic ad catalog attributes.
- `user_profile.csv`: synthetic anonymized-style user profile fields.
- `behavior_log.csv`: synthetic browsing/favourite/cart/purchase events.

## Approximate sizes
- Users: 1,200
- Ads: 400
- Ad impressions: 15,000
- Behaviour events: 42,000

## Source dataset references
- Alibaba Tianchi: https://tianchi.aliyun.com/dataset/dataDetail?dataId=56
- Alibaba MaxCompute public datasets: https://www.alibabacloud.com/help/en/maxcompute/user-guide/overview-of-public-datasets
- RecZoo TaobaoAd_x1 (preprocessed 25,029,426-row joined dataset): https://huggingface.co/datasets/reczoo/TaobaoAd_x1

The RecZoo version is a preprocessed joined dataset and is not the same as the four raw CSV files.

## Data dictionary
### raw_sample.csv
`user_id, adgroup_id, time_stamp, pid, nonclk, clk`

### ad_feature.csv
`adgroup_id, cate_id, campaign_id, customer_id, brand_id, price`

### user_profile.csv
`user_id, cms_segid, cms_group_id, final_gender_code, age_level, pvalue_level, shopping_level, occupation, new_user_class_level`

### behavior_log.csv
`user_id, time_stamp, btag, cate_id, brand_id`

`btag` values: `pv` (view), `fav` (favourite), `cart` (add to cart), `buy` (purchase).

## Usage
Unzip this package into the project `data/raw/` directory, or use the matching files already included in the complete project ZIP.

## Additional genuine-data access
- Preprocessed TaobaoAd_x1 (25,029,426 rows, 2.44 GB): https://huggingface.co/datasets/reczoo/TaobaoAd_x1
- Zenodo mirror containing three original files (about 1.1 GB): https://zenodo.org/records/8088629
The bundled demonstration files remain synthetic. These external files are not bundled because they are large and hosted externally.
