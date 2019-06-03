#!/usr/bin/env bash

INLOC_ROOT="$(python -c 'import config;print(config.INLOC_ROOT)')"
MATTERPORT_ROOT="$(python -c 'import config;print(config.MATTERPORT_ROOT)')"

pooling_mode_array=("gmp" "mean")
dataset_array=("DUC1" "DUC2")
aggr_mode_array=("pano" "pano_1_3" "pano_2" "pano_2_3" "pano_4" "pano_4_3" "pano_8" "pano_8_3" "pano_16" "pano_16_3" "pano_24" "pano_24_3" "pano_48" "pano_48_3")

for pooling_mode in "${pooling_mode_array[@]}"
do

    echo "Pooling mode: ${pooling_mode}"

    for dataset in "${dataset_array[@]}"
    do
        echo "Dataset: $dataset"

        DATASET_LOCATION=$INLOC_ROOT/${dataset}

        # Inputs
        IMAGE_DIR=$DATASET_LOCATION/view_total
        DB_IDS=$DATASET_LOCATION/query_split_0.json
        QUERY_IDS=$DATASET_LOCATION/query_split_0.json
        CACHE_DIR=$DATASET_LOCATION/features

        # Distractor dataset
        MERGED_DATASET=$MATTERPORT_ROOT/distractor_dataset
        DATASET_LOCATION_DISTRACTOR=$MERGED_DATASET
        CACHE_DIR_DISTRACTOR=$MERGED_DATASET

        for aggr_mode in "${aggr_mode_array[@]}"
        do

            echo "Aggregation mode: ${aggr_mode}"

            python semantic_evaluation.py \
                --dataset $IMAGE_DIR \
                --temp_dir $CACHE_DIR \
                --query_file $QUERY_IDS \
                --db_file $DB_IDS \
                --view_poses_file view_poses.json \
                --dataset_distractor $DATASET_LOCATION_DISTRACTOR \
                --temp_dir_distractor $CACHE_DIR_DISTRACTOR \
                --pano_dist_threshold 10.0 \
                --center_norm_dataset \
                --normalized_desc \
                --feat_aggregation ${aggr_mode} \
                --evaluation_mode flann_single \
                --pooling ${pooling_mode}

        done
        echo

    done
    echo

done
