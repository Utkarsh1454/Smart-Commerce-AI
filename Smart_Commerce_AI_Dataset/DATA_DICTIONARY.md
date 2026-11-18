# Data Dictionary

All bundled records are synthetic demonstration data.

## raw_sample.csv
- user_id: anonymized-style demo user identifier
- adgroup_id: demo advertisement identifier
- time_stamp: Unix timestamp
- pid: ad placement
- nonclk: 1 if not clicked, otherwise 0
- clk: 1 if clicked, otherwise 0

## ad_feature.csv
- adgroup_id: ad identifier
- cate_id: product category identifier
- campaign_id: campaign identifier
- customer_id: advertiser identifier
- brand_id: brand identifier
- price: demo product price

## user_profile.csv
- user_id: user identifier
- cms_segid, cms_group_id: demo segmentation fields
- final_gender_code, age_level, pvalue_level, shopping_level, occupation, new_user_class_level: demo profile fields

## behavior_log.csv
- user_id: user identifier
- time_stamp: Unix timestamp
- btag: pv, fav, cart or buy
- cate_id: product category identifier
- brand_id: brand identifier
