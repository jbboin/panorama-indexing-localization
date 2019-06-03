import numpy as np
import os
import json

import config

dataset_names = ['17DRP5sb8fy','1LXtFkjw3qL','1pXnuDYAj8r','29hnd4uzFmX','2azQ1b91cZZ']

output_dir = os.path.join(config.MATTERPORT_ROOT, 'distractor_dataset')

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Generate view_poses.json
full_data = {'images': []}
id_offset = 0
num_images = []
for dataset_name in dataset_names:
    dataset_images = json.load(open(os.path.join(config.MATTERPORT_ROOT, dataset_name, 'views/view_poses.json')))['images']
    for img in dataset_images:
        img['id'] += id_offset
        img['room'] = dataset_name + '_' + img['room']
    full_data['images'].extend(dataset_images)
    id_offset += len(dataset_images)
    num_images.append(len(dataset_images))

full_ids = [x['id'] for x in full_data['images']]
assert full_ids == range(len(full_ids))

with open(os.path.join(output_dir, 'view_poses.json'), 'w') as f:
    json.dump(full_data, f)

# Generate data_file
feature_data = np.empty((sum(num_images),2048), dtype=np.float32)
for i, dataset_name in enumerate(dataset_names):
    lo = sum(num_images[:i])
    hi = sum(num_images[:i+1])
    feature_data[lo:hi,:] = np.load(os.path.join(config.MATTERPORT_ROOT, dataset_name, 'features/descr_S800_L2.npy'))

np.save(os.path.join(output_dir, 'descr_S800_L2.npy'), feature_data)
