from __future__ import print_function
import sys
import os
import json
from collections import Counter
from operator import itemgetter
import cv2
import numpy as np
os.environ['GLOG_minloglevel'] = '2'
from tqdm import tqdm


class ImageHelper(object):
    def __init__(self, S, L, means):
        self.S = S
        self.L = L
        self.means = means

    def prepare_image_and_grid_regions(self, fname, roi=None):
        # Extract image, resize at desired size, and extract roi region if
        # available. Then compute the rmac grid in the net format: ID X Y W H
        I, im_resized = self.load_and_prepare_image(fname, roi)
        if self.L == 0:
            # Encode query in mac format instead of rmac, so only one region
            # Regions are in ID X Y W H format
            R = np.zeros((1, 5), dtype=np.float32)
            R[0, 3] = im_resized.shape[1] - 1
            R[0, 4] = im_resized.shape[0] - 1
        else:
            # Get the region coordinates and feed them to the network.
            all_regions = []
            all_regions.append(self.get_rmac_region_coordinates(im_resized.shape[0], im_resized.shape[1], self.L))
            R = self.pack_regions_for_network(all_regions)
        return I, R

    @staticmethod
    def get_rmac_features(I, R, net):
        net.blobs['data'].reshape(I.shape[0], 3, int(I.shape[2]), int(I.shape[3]))
        net.blobs['data'].data[:] = I
        net.blobs['rois'].reshape(R.shape[0], R.shape[1])
        net.blobs['rois'].data[:] = R.astype(np.float32)
        net.forward(end='rmac/normalized')
        return np.squeeze(net.blobs['rmac/normalized'].data)

    def load_and_prepare_image(self, fname, roi=None):
        # Read image, get aspect ratio, and resize such as the largest side equals S
        im = cv2.imread(fname)
        im_size_hw = np.array(im.shape[0:2])
        ratio = float(self.S)/np.max(im_size_hw)
        new_size = tuple(np.round(im_size_hw * ratio).astype(np.int32))
        im_resized = cv2.resize(im, (new_size[1], new_size[0]))
        # If there is a roi, adapt the roi to the new size and crop. Do not rescale
        # the image once again
        if roi is not None:
            roi = np.round(roi * ratio).astype(np.int32)
            im_resized = im_resized[roi[1]:roi[3], roi[0]:roi[2], :]
        # Transpose for network and subtract mean
        I = im_resized.transpose(2, 0, 1) - self.means
        return I, im_resized

    @staticmethod
    def pack_regions_for_network(all_regions):
        n_regs = np.sum([len(e) for e in all_regions])
        R = np.zeros((n_regs, 5), dtype=np.float32)
        cnt = 0
        # There should be a check of overflow...
        for i, r in enumerate(all_regions):
            try:
                R[cnt:cnt + r.shape[0], 0] = i
                R[cnt:cnt + r.shape[0], 1:] = r
                cnt += r.shape[0]
            except:
                continue
        assert cnt == n_regs
        R = R[:n_regs]
        # regs where in xywh format. R is in xyxy format, where the last coordinate is included. Therefore...
        R[:n_regs, 3] = R[:n_regs, 1] + R[:n_regs, 3] - 1
        R[:n_regs, 4] = R[:n_regs, 2] + R[:n_regs, 4] - 1
        return R

    @staticmethod
    def get_rmac_region_coordinates(H, W, L):
        # Almost verbatim from Tolias et al Matlab implementation.
        # Could be heavily pythonized, but really not worth it...
        # Desired overlap of neighboring regions
        ovr = 0.4
        # Possible regions for the long dimension
        steps = np.array((2, 3, 4, 5, 6, 7), dtype=np.float32)
        w = np.minimum(H, W)

        b = (np.maximum(H, W) - w) / (steps - 1)
        # steps(idx) regions for long dimension. The +1 comes from Matlab
        # 1-indexing...
        idx = np.argmin(np.abs(((w**2 - w * b) / w**2) - ovr)) + 1

        # Region overplus per dimension
        Wd = 0
        Hd = 0
        if H < W:
            Wd = idx
        elif H > W:
            Hd = idx

        regions_xywh = []
        for l in range(1, L+1):
            wl = np.floor(2 * w / (l + 1))
            wl2 = np.floor(wl / 2 - 1)
            # Center coordinates
            if l + Wd - 1 > 0:
                b = (W - wl) / (l + Wd - 1)
            else:
                b = 0
            cen_w = np.floor(wl2 + b * np.arange(l - 1 + Wd + 1)) - wl2
            # Center coordinates
            if l + Hd - 1 > 0:
                b = (H - wl) / (l + Hd - 1)
            else:
                b = 0
            cen_h = np.floor(wl2 + b * np.arange(l - 1 + Hd + 1)) - wl2

            for i in cen_h:
                for j in cen_w:
                    regions_xywh.append([j, i, wl, wl])

        # Round the regions. Careful with the borders!
        for region in regions_xywh:
            for j in range(4):
                region[j] = int(round(region[j]))
            if region[0] + region[2] > W:
                region[0] -= ((region[0] + region[2]) - W)
            if region[1] + region[3] > H:
                region[1] -= ((region[1] + region[3]) - H)
        return np.array(regions_xywh).astype(np.float32)


class Dataset(object):
    def __init__(self, img_root, two_stage_num_clusters, filename_poses):
        self.img_root = img_root
        self.two_stage_num_clusters = two_stage_num_clusters
        self.image_data = self.load_image_data(filename_poses)
        self.size_dataset = len(self.image_data)
        self.db_indices = None
        self.all_query_info = None
        self.query_indices = None
        self.num_rooms = -1
        # Mapping from room string to room ID (used in custom aggregation mode and for ground truth in pano retrieval)
        self.room_string_to_id = None
        # Ground truth info
        self.query_id_to_positive_classes = None
        self.query_id_to_junk_classes = None
        # Mapping from image ID to room or panorama ID
        self.image_info = {}
        # Mapping from cluster ID to room ID / panorama ID / view IDs
        self.cluster_info = None
        # Random offset for intra-panorama aggregation
        self.pano_rand_offset = None

    def load_image_data(self, filename_poses):
        # Load image data from JSON file of full dataset and sort by increasing ID value
        #filename_poses = 'view_poses_filtered.json'
        assert os.path.exists(os.path.join(self.img_root, filename_poses)), \
            'The file %s was not found in %s' % (filename_poses, self.img_root)
        json_data = json.load(open(os.path.join(self.img_root, filename_poses)))
        image_data = sorted(json_data['images'], key=itemgetter('id'))
        assert [image['id'] for image in image_data] == range(len(image_data)), \
            'Non consecutive image IDs'
        return image_data

    def set_query_info(self, query_file):
        if query_file:
            json_data = json.load(open(query_file))
            for query in json_data['query_views']:
                assert query['filename'] == self.image_data[query['id']]['filename'], \
                    'ID mismatch between dataset file and query file'
            self.all_query_info = sorted(json_data['query_views'], key=itemgetter('id'))
        else:
            self.all_query_info = []

    def set_db_info(self, db_file):
        if db_file:
            json_data = json.load(open(db_file))
            for db_entry in json_data['db_views']:
                assert db_entry['filename'] == self.image_data[db_entry['id']]['filename'], \
                    'ID mismatch between dataset file and database file'
            self.db_indices = np.sort([db_entry['id'] for db_entry in json_data['db_views']])
        else:
            self.db_indices = np.arange(len(self.image_data))

    def set_retrieval_info(self, query_file, db_file):
        # Get query IDs from the query JSON file
        self.set_query_info(query_file)
        # Get database IDs from the database JSON file
        self.set_db_info(db_file)
        # Check that there is no overlap
        all_query_indices = [query['id'] for query in self.all_query_info]
        assert len(set(all_query_indices).intersection(set(self.db_indices))) == 0, \
            'Intersection between query and db images is not empty'

        # Get room IDs
        rooms = [image['room'] for image in self.image_data]
        room_id_to_string, self.image_info['room'] = np.unique(rooms, return_inverse=True)
        self.room_string_to_id = {}
        for room_id, room in enumerate(room_id_to_string):
            self.room_string_to_id[room] = room_id
        self.num_rooms = len(room_id_to_string)

        # Get panorama IDs
        panos = [image['source_pano_filename'] for image in self.image_data]
        _, self.image_info['pano'] = np.unique(panos, return_inverse=True)

    # This function takes as input the original view-level features and returns the aggregated features. For each
    # aggregated feature (cluster), it also populates the cluster information into the list self.cluster_info.
    def aggregate_features(self, features, feat_aggregation_stages, pooling, cluster_files, randomize=False):
        features_stages = []
        cluster_info_stages = []
        num_custom_stages = feat_aggregation_stages.count('custom')
        assert num_custom_stages == len(cluster_files) or num_custom_stages == 0, \
            'The number of cluster files should match the number of stages with custom aggregation'
        custom_idx = 0
        for feat_aggregation in feat_aggregation_stages:
            if feat_aggregation == 'custom':
                cluster_info = self.compute_clusters(feat_aggregation, cluster_files[custom_idx], randomize)
                custom_idx += 1
            else:
                cluster_info = self.compute_clusters(feat_aggregation, None, randomize)
            features_stage, cluster_info_nonempty = self.aggregate_by_cluster(features, cluster_info, pooling)
            # Remove empty clusters
            cluster_info_stages.append(cluster_info_nonempty)
            features_stages.append(features_stage)
        self.cluster_info = cluster_info_stages
        return features_stages

    def compute_clusters(self, feat_aggregation, cluster_file=None, randomize=False):
        assert feat_aggregation.startswith(('none', 'yaw', 'pano', 'room', 'custom')), \
            'Invalid feat_aggregation string'
        cluster_info = []
        if feat_aggregation == 'none':
            for i in self.db_indices:
                cluster_info.append({
                    'room_id': self.image_info['room'][i],
                    'pano_id': self.image_info['pano'][i],
                    'image_id': i,
                    'image_ids': [i]
                })
        elif feat_aggregation.startswith('yaw'):
            # Adaptive scheme means that the number of bins is automatically related to the number of panoramas in the
            # room
            if '_' in feat_aggregation:
                num_bins_specified = int(feat_aggregation.split('_')[1])
                adaptive_scheme = feat_aggregation.startswith('yawadap')
            else:
                # yaw is equivalent to yawadap_1
                num_bins_specified = 1
                adaptive_scheme = True
            for room_id in range(self.num_rooms):
                db_in_room = np.intersect1d(np.where(self.image_info['room'] == room_id)[0], self.db_indices)
                if adaptive_scheme:
                    # We bin the yaw angle so that the number of descriptors per room is (around)
                    # the same as the number of panoramas
                    panoramas = [self.image_data[db_idx]['source_pano_filename'] for db_idx in db_in_room]
                    num_panos_in_room = len(np.unique(panoramas))
                    num_bins = num_panos_in_room * num_bins_specified
                else:
                    num_bins = num_bins_specified
                if num_bins == 0:
                    continue
                yaw = np.array(
                    [self.get_yaw_from_quaternion(self.image_data[db_idx]['final_camera_rotation'])
                     for db_idx in db_in_room]
                )
                yaw_quant = np.digitize(yaw, np.arange(-np.pi, np.pi, 2*np.pi/num_bins))-1
                for yaw_id in range(num_bins):
                    ids = db_in_room[np.where(yaw_quant == yaw_id)[0]]
                    cluster_info.append({
                        'room_id': room_id,
                        'pano_id': None,
                        'image_id': None,
                        'image_ids': list(ids)
                    })
        elif feat_aggregation.startswith('pano'):
            num_bins = 1
            num_bins_v = 1
            if '_' in feat_aggregation:
                num_bins = int(feat_aggregation.split('_')[1])
                if feat_aggregation.count('_') == 2:
                    num_bins_v = int(feat_aggregation.split('_')[2])
                    assert num_bins_v == 1 or num_bins_v == 3, \
                        'Only 1 or 3 vertical bins are allowed'
            panoramas = []
            # For each panorama, draw a random offset value in [0, 360) so that the aggregation is randomized
            if self.pano_rand_offset is None:
                self.pano_rand_offset = {}
                for pano_filename in set([image['source_pano_filename'] for image in self.image_data]):
                    self.pano_rand_offset[pano_filename] = np.random.rand() * 360
            for image in self.image_data:
                if len(image['filename'].split('_')) < 3:
                    rel_yaw = 0
                else:
                    rel_yaw = float(image['filename'].split('_')[-3])
                if randomize:
                    rel_yaw = (rel_yaw + self.pano_rand_offset[image['source_pano_filename']]) % 360
                rel_yaw_bin = np.floor(rel_yaw * num_bins / 360)
                pitch_bin = 0
                if num_bins_v > 1 and len(image['filename'].split('_')) >= 3:
                    pitch = float(image['filename'].split('_')[-1][:-4])
                    if pitch < -10.0:
                        pitch_bin = 0
                    elif pitch > 10.0:
                        pitch_bin = 2
                    else:
                        pitch_bin = 1
                panoramas.append('%s_%d_%d' % (image['source_pano_filename'], rel_yaw_bin, pitch_bin))
            panoramas = np.array(panoramas)
            unique_panoramas = np.unique(panoramas)
            for pano in unique_panoramas:
                cluster_ids = np.where(panoramas == pano)[0]
                # Unchecked: we expect that all images associated to a panorama have the same pano ID
                room_id = self.image_info['room'][cluster_ids[0]]
                pano_id = self.image_info['pano'][cluster_ids[0]]
                ids = np.intersect1d(cluster_ids, self.db_indices)
                cluster_info.append({
                    'room_id': room_id,
                    'pano_id': pano_id,
                    'image_id': None,
                    'image_ids': list(ids)
                })
        elif feat_aggregation == 'room':
            for room_id in range(self.num_rooms):
                db_in_room = np.intersect1d(np.where(self.image_info['room'] == room_id)[0], self.db_indices)
                cluster_info.append({
                    'room_id': room_id,
                    'pano_id': None,
                    'image_id': None,
                    'image_ids': list(db_in_room)
                })
        elif feat_aggregation == 'custom':
            clusters = json.load(open(cluster_file))
            for room in clusters:
                if not isinstance(clusters[room], list):
                    continue
                if room in self.room_string_to_id:
                    room_id = self.room_string_to_id[room]
                else:
                    room_id = None
                for cluster in clusters[room]:
                    assert len(np.setdiff1d(cluster['ids'], self.db_indices)) == 0, \
                        'The clustering file should only include views included in the database'
                    cluster_info.append({
                        'room_id': room_id,
                        'pano_id': None,
                        'image_id': None,
                        'image_ids': cluster['ids']
                    })
        else:
            raise NotImplementedError('Unsupported aggregation mode')
        return cluster_info

    @staticmethod
    def get_yaw_from_quaternion(quat):
        # Returns a yaw value in [-pi, pi]
        return np.arctan2(2.0*(quat[1]*quat[2] + quat[3]*quat[0]),
                          quat[3]*quat[3] - quat[0]*quat[0] - quat[1]*quat[1] + quat[2]*quat[2])

    def aggregate_by_cluster(self, features, cluster_info, pooling):
        valid_cluster_indices = []
        aggregated_features = []
        for cluster_idx, cluster in enumerate(cluster_info):
            feat_indices = [np.where(self.db_indices == i)[0][0] for i in cluster['image_ids']]
            if not feat_indices:
                continue
            valid_cluster_indices.append(cluster_idx)
            x = features[feat_indices, :]
            if pooling == 'mean':
                aggregated_feature = np.mean(x, axis=0)
            elif pooling == 'gmp':
                aggregated_feature = np.mean(np.dot(np.linalg.inv(np.dot(x, x.T)+np.eye(x.shape[0])), x), axis=0)
            else:
                raise NotImplementedError('Unsupported pooling strategy')
            aggregated_features.append(aggregated_feature)
        # Only keep non-empty clusters
        cluster_info_nonempty = [cluster_info[i] for i in valid_cluster_indices]
        aggregated_features = np.array(aggregated_features)
        # Normalize features
        aggregated_features /= np.sqrt((aggregated_features * aggregated_features).sum(axis=1))[:, None]
        return aggregated_features, cluster_info_nonempty

    def get_filename(self, i):
        return os.path.normpath("{0}/{1}".format(self.img_root, self.image_data[i]['filename']))


class PanoRetrievalDataset(Dataset):
    def get_pano_info(self):
        # For each panorama, we look up the camera location and the room ID
        panos = [image['source_pano_filename'] for image in self.image_data]
        _, unique_panos_index = np.unique(panos, return_index=True)
        pano_info = []
        for i in unique_panos_index:
            pano_location = np.array(self.image_data[i]['camera_location'])
            room_id = self.image_info['room'][i]
            db_ids = set(np.where(self.image_info['pano'] == self.image_info['pano'][i])[0]). \
                intersection(set(self.db_indices))
            pano_info.append({'location': pano_location, 'room_id': room_id, 'db_ids': sorted(list(db_ids))})
        return pano_info

    def set_ground_truth_info(self, pano_dist_threshold, ignore_rooms=False):
        pano_info = self.get_pano_info()

        # Get ground truth information based on panorama location
        self.query_indices = []
        self.query_id_to_positive_classes = {}
        for query_info in self.all_query_info:
            query_idx = query_info['id']
            query_location = np.array(self.image_data[query_idx]['camera_location'])
            if 'room_labels' in query_info:
                query_pano_rooms = [self.room_string_to_id[room] for room in query_info['room_labels']]
            else:
                query_pano_rooms = [self.image_info['room'][query_idx]]
            assert self.image_info['room'][query_idx] in query_pano_rooms, \
                'The room containing the panorama is not listed in the list of ground truth rooms'
            # Find matching panoramas (only consider the ones corresponding to db images)
            pano_query_dist = np.full(len(pano_info), np.nan)
            for i, pano in enumerate(pano_info):
                if not ignore_rooms and pano['room_id'] not in query_pano_rooms:
                    continue
                if len(pano['db_ids']) == 0:
                    # Pano isn't a db pano
                    continue
                pano_query_dist[i] = np.linalg.norm(query_location - pano['location'])
            if np.all(np.isnan(pano_query_dist)):
                print('WARNING: No available panorama for ground truth.')
                continue
            # Note: we suppress warnings for this command because Numpy complains about nan comparisons
            with np.errstate(invalid='ignore'):
                sub_thresh_panos = np.where(pano_query_dist < pano_dist_threshold)[0]
            if len(sub_thresh_panos) > 0:
                gt_pano_ids = sub_thresh_panos
            else:
                gt_pano_ids = np.array([np.nanargmin(pano_query_dist)])
            self.query_indices.append(query_idx)
            self.query_id_to_positive_classes[query_idx] = gt_pano_ids
        self.query_indices = np.array(self.query_indices, dtype=int)


class FeatureExtractor(object):
    def __init__(self, temp_dir, multires, S, L):
        self.temp_dir = temp_dir
        self.multires = multires
        self.S = S
        self.L = L
        self.features_queries = None
        self.features_dataset = None

    def extract_features(self, dataset, image_helper, net):
        Ss = [self.S, ] if not self.multires else [self.S - 250, self.S, self.S + 250]
        for S in Ss:
            # Set the scale of the image helper
            image_helper.S = S
            out_descr_fname = "{0}/descr_S{1}_L{2}.npy".format(self.temp_dir, S, self.L)
            generate_features = False
            if os.path.exists(out_descr_fname):
                features = np.load(out_descr_fname)
                if features.shape[0] != dataset.size_dataset:
                    print('--> Pre-generated features have incorrect shape ; generating new features')
                    generate_features = True
            else:
                generate_features = True
            if generate_features:
                dim_features = net.blobs['rmac/normalized'].data.shape[1]
                features = np.empty((dataset.size_dataset, dim_features), dtype=np.float32)
                features.fill(np.nan)
            # Check which of the features we need to compute haven't been computed yet (= nan)
            compute_indices = []
            for i in np.concatenate((dataset.query_indices, dataset.db_indices)):
                if np.isnan(features[i, 0]):
                    compute_indices.append(i)
            for i in tqdm(compute_indices, file=sys.stdout, leave=False, dynamic_ncols=True):
                # Load image, process image, get image regions, feed into the network, get descriptor, and store
                I, R = image_helper.prepare_image_and_grid_regions(dataset.get_filename(i), roi=None)
                features[i] = image_helper.get_rmac_features(I, R, net)
            # Save new matrix if needed
            if compute_indices:
                np.save(out_descr_fname, features)
        features = np.dstack(
            [np.load("{0}/descr_S{1}_L{2}.npy".format(self.temp_dir, S, self.L)) for S in Ss]
        ).sum(axis=2)
        features /= np.sqrt((features * features).sum(axis=1))[:, None]
        # Restore the original scale
        image_helper.S = self.S

        # Extract queries and db features in 2 different matrices
        self.features_queries = features[dataset.query_indices, :]
        assert (np.isnan(self.features_queries).any() == False)
        self.features_dataset = [features[dataset.db_indices, :]]
        assert (np.isnan(self.features_dataset).any() == False)
