#!/usr/bin/env bash

INLOC_ROOT="$(python -c 'import config;print(config.INLOC_ROOT)')"
MATTERPORT_ROOT="$(python -c 'import config;print(config.MATTERPORT_ROOT)')"

dataset_array=("DUC1" "DUC2")

hierarchy_array=(
    "room pano pano_2 pano_4 pano_8 pano_8_3"
    "room pano pano_4 pano_8_3"
    "room pano pano_8_3"
    "room pano pano_4 pano_4_3"
    "room pano pano_4 pano_48_3"
)

for dataset in "${dataset_array[@]}"
do

    echo "Dataset: $dataset"

    # Number of rooms chosen by script "get_number_rooms.py"
    if [ "$dataset" = "DUC1" ]; then
        num_rooms=25
    else
        num_rooms=37
    fi

    # Inputs
    DATASET_LOCATION=$INLOC_ROOT/${dataset}
    IMAGE_DIR=$DATASET_LOCATION/view_total
    QUERY_IDS=$DATASET_LOCATION/query_db_split.json
    DB_IDS=$DATASET_LOCATION/query_db_split.json
    CACHE_DIR=$DATASET_LOCATION/features

    # Distractor dataset
    MERGED_DATASET=$MATTERPORT_ROOT/distractor_dataset
    DATASET_LOCATION_DISTRACTOR=$MERGED_DATASET
    CACHE_DIR_DISTRACTOR=$MERGED_DATASET

    for hierarchy in "${hierarchy_array[@]}"
    do

        echo "Hierarchy: $hierarchy"

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
            --feat_aggregation $hierarchy \
            --pooling gmp \
            --num_rooms_dataset $num_rooms

    done
    echo

done
