#!/usr/bin/env bash

INLOC_ROOT="$(python -c 'import config;print(config.INLOC_ROOT)')"
MATTERPORT_ROOT="$(python -c 'import config;print(config.MATTERPORT_ROOT)')"

DATASET_LOCATION=$INLOC_ROOT/DUC1

# Inputs
IMAGE_DIR=$DATASET_LOCATION/view_total
QUERY_IDS=$DATASET_LOCATION/query_db_split.json
DB_IDS=$DATASET_LOCATION/query_db_split.json
CACHE_DIR=$DATASET_LOCATION/features

# Distractor dataset
MERGED_DATASET=$MATTERPORT_ROOT/distractor_dataset
DATASET_LOCATION_DISTRACTOR=$MERGED_DATASET                 
CACHE_DIR_DISTRACTOR=$MERGED_DATASET

python flann_autotuned.py \
    --dataset $IMAGE_DIR \
    --temp_dir $CACHE_DIR \
    --query_file $QUERY_IDS \
    --db_file $DB_IDS \
    --view_poses_file view_poses.json \
    --dataset_distractor $DATASET_LOCATION_DISTRACTOR \
    --temp_dir_distractor $CACHE_DIR_DISTRACTOR \
    --pano_dist_threshold 10.0 \
    --center_norm_dataset \
    --feat_aggregation none
