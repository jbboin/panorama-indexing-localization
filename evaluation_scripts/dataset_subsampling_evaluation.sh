#!/usr/bin/env bash

INLOC_ROOT="$(python -c 'import config;print(config.INLOC_ROOT)')"
MATTERPORT_ROOT="$(python -c 'import config;print(config.MATTERPORT_ROOT)')"

dataset_array=("DUC1" "DUC2")

for dataset in "${dataset_array[@]}"
do
    echo "Dataset: $dataset"

    DATASET_LOCATION=$INLOC_ROOT/${dataset}

    # Inputs
    IMAGE_DIR=$DATASET_LOCATION/view_total
    QUERY_IDS=$DATASET_LOCATION/query_db_split.json
    CACHE_DIR=$DATASET_LOCATION/features

    # Distractor dataset
    MERGED_DATASET=$MATTERPORT_ROOT/distractor_dataset
    DATASET_LOCATION_DISTRACTOR=$MERGED_DATASET
    CACHE_DIR_DISTRACTOR=$MERGED_DATASET

    horiz_samp_array=(4 8 12 16 24 48)
    vert_samp_array=(1 3)

    for horiz_samp in "${horiz_samp_array[@]}"
    do
        for vert_samp in "${vert_samp_array[@]}"
        do

            DB_IDS=$DATASET_LOCATION/query_db_split_samp_${horiz_samp}_${vert_samp}.json
            DB_IDS_DISTRACTOR=$MERGED_DATASET/query_db_split_samp_${horiz_samp}_${vert_samp}.json

            echo "Sampling: $horiz_samp, $vert_samp"
        
            python semantic_evaluation.py \
                --dataset $IMAGE_DIR \
                --temp_dir $CACHE_DIR \
                --query_file $QUERY_IDS \
                --db_file $DB_IDS \
                --view_poses_file view_poses.json \
                --dataset_distractor $DATASET_LOCATION_DISTRACTOR \
                --db_file_distractor $DB_IDS_DISTRACTOR \
                --temp_dir_distractor $CACHE_DIR_DISTRACTOR \
                --pano_dist_threshold 10.0 \
                --center_norm_dataset \
                --normalized_desc \
                --feat_aggregation none \
                --evaluation_mode flann_single

        done
    done
    
    echo

done
