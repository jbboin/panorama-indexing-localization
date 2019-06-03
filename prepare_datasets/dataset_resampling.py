import numpy as np
import os
import json
from itertools import chain

import config


def resample_dataset(input_dir, input_file_name):

    horiz_sampling_rates = [4,8,12,16,24,48]
    sampling_rates = list(chain.from_iterable(((r,1),(r,3)) for r in horiz_sampling_rates))

    input_file = os.path.join(input_dir, input_file_name)
    dataset_full = json.load(open(input_file))

    if 'db_views' in dataset_full:
        dataset_query = dataset_full['query_views']
        dataset_db = dataset_full['db_views']
    else:
        dataset_query = []
        dataset_db = dataset_full['images']

    num_panos = len(dataset_db) / 144

    for sampling_rate in sampling_rates:

        output_file = os.path.join(input_dir, 'query_db_split_samp_%d_%d.json' % sampling_rate)

        # horizontal sampling
        assert 48 % sampling_rate[0] == 0
        h_step = 48/sampling_rate[0]
        indices = range(48*num_panos)[::h_step]

        # vertical sampling
        if sampling_rate[1] == 1:
            indices = [3*i+1 for i in indices]
        elif sampling_rate[1] == 3:
            indices = list(chain.from_iterable((3*i, 3*i+1, 3*i+2) for i in indices))
        else:
            raise NotImplementedError('Unsupported vertical sampling value')

        assert len(indices) == num_panos * sampling_rate[0] * sampling_rate[1]

        dataset_resampled = {}
        dataset_resampled['query_views'] = dataset_query
        dataset_resampled['db_views'] = [{'id': dataset_db[i]['id'], 'filename': dataset_db[i]['filename']} for i in indices]

        with open(output_file, 'w') as f:
            json.dump(dataset_resampled, f)


resample_dataset(os.path.join(config.INLOC_ROOT, 'DUC1'), 'query_db_split.json')
resample_dataset(os.path.join(config.INLOC_ROOT, 'DUC2'), 'query_db_split.json')
resample_dataset(os.path.join(config.MATTERPORT_ROOT, 'distractor_dataset'), 'view_poses.json')
