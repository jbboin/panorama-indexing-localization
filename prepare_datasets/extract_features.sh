#!/usr/bin/env bash

DEEP_RETRIEVAL_ROOT="$(python -c 'import config;print(config.DEEP_RETRIEVAL_ROOT)')"
INLOC_ROOT="$(python -c 'import config;print(config.INLOC_ROOT)')"
MATTERPORT_ROOT="$(python -c 'import config;print(config.MATTERPORT_ROOT)')"

dataset_array=("DUC1" "DUC2")

for dataset in "${dataset_array[@]}"
do
    echo "Dataset: $dataset"

    DATASET_LOCATION=$INLOC_ROOT/$dataset
    IMAGE_DIR=$DATASET_LOCATION/view_total
    CACHE_DIR=$DATASET_LOCATION/features

    python extract_features.py \
        --dataset $IMAGE_DIR \
        --temp_dir $CACHE_DIR \
        --view_poses_file view_poses.json \
        --proto $DEEP_RETRIEVAL_ROOT/deploy_resnet101_normpython.prototxt \
        --weights $DEEP_RETRIEVAL_ROOT/model.caffemodel

done

dataset_array=("17DRP5sb8fy" "1LXtFkjw3qL" "1pXnuDYAj8r" "29hnd4uzFmX" "2azQ1b91cZZ")

for dataset in "${dataset_array[@]}"
do
    echo "Dataset: Matterport (distractor) - $dataset"

    DATASET_LOCATION=$MATTERPORT_ROOT/$dataset
    IMAGE_DIR=$DATASET_LOCATION/views
    CACHE_DIR=$DATASET_LOCATION/features

    python extract_features.py \
        --dataset $IMAGE_DIR \
        --temp_dir $CACHE_DIR \
        --view_poses_file view_poses.json \
        --proto $DEEP_RETRIEVAL_ROOT/deploy_resnet101_normpython.prototxt \
        --weights $DEEP_RETRIEVAL_ROOT/model.caffemodel

done
