import config
import evaluate_lib

import os
import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict

dataset_distractor_path = os.path.join(config.MATTERPORT_ROOT, 'distractor_dataset')
view_poses_file_distractor = 'view_poses.json'

dataset_names = ['DUC1', 'DUC2']

# Load distractor dataset
dataset_distractor = evaluate_lib.PanoRetrievalDataset(dataset_distractor_path, [], view_poses_file_distractor)
dataset_distractor.set_retrieval_info('', '')

# Get panorama locations in each room
pano_info_distractor = [p for p in dataset_distractor.get_pano_info() if len(p['db_ids']) > 0]
room_to_locations = defaultdict(list)
for p in pano_info_distractor:
    room_to_locations[p['room_id']].append(p['location'])

# Get mean distance to room centroid
dists = []
for room, locations in room_to_locations.items():
    res = locations - np.mean(locations, 0)
    for x in np.linalg.norm(res, axis=1):
        dists.append(x)

mean_dist_distractor = np.mean(dists)

for dataset_name in dataset_names:
    dataset_path = os.path.join(config.INLOC_ROOT, dataset_name, 'view_total')
    db_file = os.path.join(config.INLOC_ROOT, dataset_name, 'query_db_split.json')
    view_poses_file = 'view_poses.json'

    # Load main dataset
    dataset_obj = evaluate_lib.PanoRetrievalDataset(dataset_path, [], view_poses_file)
    dataset_obj.set_retrieval_info('', db_file)

    # Get all panorama locations
    pano_info = dataset_obj.get_pano_info()
    pano_locations = np.array([p['location'] for p in pano_info if len(p['db_ids']) > 0])

    # Cluster the locations into various numbers of rooms and save the mean distance to the room centroid
    num_rooms_values = range(1, len(pano_locations))
    mean_dists = []
    for num_rooms in num_rooms_values:
        # Cluster panorama locations using the desired number of clusters
        kmeans = KMeans(n_clusters=num_rooms, random_state=0).fit(pano_locations)
        res = pano_locations - kmeans.cluster_centers_[kmeans.labels_]
        mean_dist = np.mean(np.linalg.norm(res, axis=1))
        mean_dists.append(np.mean(np.linalg.norm(res, axis=1)))

    # Get the number of rooms that gives the value of mean distance most similar to the distractor dataset
    room_idx = np.argmin(np.abs(np.array(mean_dists) - mean_dist_distractor))
    recommended_num_rooms = num_rooms_values[room_idx]

    print('Dataset %s:' % dataset_name)
    print('    Recommended number of rooms: %d' % recommended_num_rooms)
    print('        Mean distance with room centroid: %f' % mean_dists[room_idx])
    print('        Mean distance with for distractor dataset: %f' % mean_dist_distractor)
