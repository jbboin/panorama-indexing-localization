#!/usr/bin/env bash

MATTERPORT_ROOT="$(python -c 'import config;print(config.MATTERPORT_ROOT)')"

dataset_id_array=("17DRP5sb8fy" "1LXtFkjw3qL" "1pXnuDYAj8r" "29hnd4uzFmX" "2azQ1b91cZZ")

for dataset_id in "${dataset_id_array[@]}"
do
    echo "Downloading dataset: ${dataset_id}"
    python scripts/download_mp.py -o $MATTERPORT_ROOT --id ${dataset_id} --type matterport_skybox_images house_segmentations matterport_camera_poses
done

for dataset_id in "${dataset_id_array[@]}"
do
    echo "Unzipping dataset: ${dataset_id}"
    unzip -u $MATTERPORT_ROOT/v1/scans/${dataset_id}/matterport_skybox_images.zip -d $MATTERPORT_ROOT
    unzip -u $MATTERPORT_ROOT/v1/scans/${dataset_id}/house_segmentations.zip -d $MATTERPORT_ROOT
    unzip -u $MATTERPORT_ROOT/v1/scans/${dataset_id}/matterport_camera_poses.zip -d $MATTERPORT_ROOT
done
