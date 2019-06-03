#!/usr/bin/env bash

INLOC_ROOT="$(python -c 'import config;print(config.INLOC_ROOT)')"
MATTERPORT_ROOT="$(python -c 'import config;print(config.MATTERPORT_ROOT)')"
BUILD_PATH="view_rendering/build"

dataset_array=("DUC1" "DUC2")

for dataset in "${dataset_array[@]}"
do
    echo "Dataset: $dataset"
    
    $BUILD_PATH/render_views_from_pano_inloc \
        --in_pano_dir $INLOC_ROOT/$dataset/panoramas \
        --in_poses_dir $INLOC_ROOT/$dataset/pose \
        --out_rgb_view_dir $INLOC_ROOT/$dataset/views \
        --num_yaw_angles 48 \
        --render_views_from_panoramas

done
echo

dataset_array=("17DRP5sb8fy" "1LXtFkjw3qL" "1pXnuDYAj8r" "29hnd4uzFmX" "2azQ1b91cZZ")

for dataset in "${dataset_array[@]}"
do
    echo "Dataset: $dataset"
    
    $BUILD_PATH/render_views_from_pano_matterport \
        --in_pano_dir $MATTERPORT_ROOT/$dataset/panoramas \
        --in_poses_dir $MATTERPORT_ROOT/$dataset/matterport_camera_poses \
        --in_rooms_file $MATTERPORT_ROOT/$dataset/house_segmentations/panorama_to_region.txt \
        --out_rgb_view_dir $MATTERPORT_ROOT/$dataset/views \
        --num_yaw_angles 48 \
        --render_views_from_panoramas

done
echo

