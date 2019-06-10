

import config

import json
import os
import csv

with open('data/room_labels.csv', mode='r') as f:
    reader = csv.reader(f)
    room_dict = dict(reader)

for building in ['DUC1', 'DUC2']:

    json_input_file = os.path.join(config.INLOC_ROOT, building, 'view_total', 'view_poses.json')
    img_data = json.load(open(json_input_file))

    for img in img_data['images']:
        source_stem = os.path.splitext(img['source_pano_filename'])[0]
        assert source_stem in room_dict, 'Source file (%s) does not have a corresponding room label specified' % source_stem
        img['room'] = room_dict[source_stem]

    # Note: we use the same file as output, which overwrites the original file
    json_output_file = json_input_file
    with open(json_output_file, 'w') as f:
        json.dump(img_data, f)
