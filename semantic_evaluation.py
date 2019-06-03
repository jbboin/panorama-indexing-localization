import evaluate_lib
import tree_lib

import argparse
import numpy as np
import tempfile
from sklearn.cluster import KMeans
import pyflann


def parse_arguments():
    parser = argparse.ArgumentParser(description='Evaluate dataset')
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Path to the dataset directory'
    )
    parser.add_argument(
        '--dataset_distractor',
        type=str,
        required=False,
        default='',
        help='(Distractor dataset) Path to the dataset directory'
    )
    parser.add_argument(
        '--query_file',
        type=str,
        required=True,
        help='Path to the JSON file containing the IDs of files used as queries'
    )
    parser.add_argument(
        '--db_file',
        type=str,
        required=True,
        help='Path to the JSON file containing the IDs of files used as database'
    )
    parser.add_argument(
        '--db_file_distractor',
        type=str,
        required=False,
        default='',
        help='(Distractor dataset) Path to the JSON file containing the IDs of files used as database'
    )
    parser.add_argument(
        '--temp_dir',
        type=str,
        required=True,
        help='Path to a temporary directory to store features and scores'
    )
    parser.add_argument(
        '--temp_dir_distractor',
        type=str,
        required=False,
        default='',
        help='(Distractor dataset) Path to a temporary directory to store features and scores'
    )
    parser.add_argument(
        '--view_poses_file',
        type=str,
        required=False,
        default='view_poses.json',
        help='File with poses of views, their filenames, inds etc.'
    )
    parser.add_argument(
        '--view_poses_file_distractor',
        type=str,
        required=False,
        default='view_poses.json',
        help='(Distractor dataset) File with poses of views, their filenames, inds etc.'
    )
    parser.add_argument(
        '--feat_aggregation',
        type=str,
        required=False,
        nargs='*',
        default=['none'],
        help='Feature aggregation mode. Accepted values are: none (default), yaw, yaw_###, yawadap_###, '
             'pano, pano_###, room'
    )
    parser.add_argument(
        '--pooling',
        type=str,
        required=False,
        default='mean',
        help='Feature pooling strategy used for feature points when building the data structure. Only used if there is '
             'some aggregation. Accepted values are: mean (default), gmp'
    )
    parser.add_argument(
        '--center_norm_dataset',
        dest='center_norm_dataset',
        action='store_true',
        help='If set, the view descriptors will first be centered and normalized (happens before aggregation)'
    )
    parser.set_defaults(center_norm_dataset=False)
    parser.add_argument(
        '--normalized_desc',
        dest='normalized_desc',
        action='store_true',
        help='If set, the descriptors in the tree will be normalized.'
    )
    parser.set_defaults(normalized_desc=False)
    parser.add_argument(
        '--bf',
        type=int,
        required=False,
        default=-1,
        help='Sets value of branching factor used to build the k-means tree on top of the semantic tree (if positive).'
    )
    parser.add_argument(
        '--num_rooms_dataset',
        type=int,
        required=False,
        default=-1,
        help='If positive, enables position based room clustering with the given number of clusters. Otherwise the '
             'room segmentation is the one from the view_poses file.'
    )
    parser.add_argument(
        '--pano_dist_threshold',
        type=float,
        required=False,
        default=0.0,
        help='Distance threshold under which a panorama is considered a match'
    )
    parser.add_argument(
        '--evaluation_mode',
        type=str,
        required=False,
        default='flann',
        help='Accepted values are: flann (default), flann_single, multistage_rank, multistage_dist'
    )
    args = parser.parse_args()
    return args


def main(args):
    # Parameters left as default
    args.S = 800
    args.L = 2
    args.multires = False
    args.proto = '/scratch/PI/bgirod/localization/deep-retrieval-files/deep_retrieval/deploy_resnet101_normpython.prototxt'
    args.weights = '/scratch/PI/bgirod/localization/deep-retrieval-files/deep_retrieval/model.caffemodel'
    args.ignore_rooms = False

    print('--> Building dataset')

    dataset_obj = evaluate_lib.PanoRetrievalDataset(args.dataset, [], args.view_poses_file)
    # Set query / db info
    dataset_obj.set_retrieval_info(args.query_file, args.db_file)
    # Set ground truth info
    dataset_obj.set_ground_truth_info(args.pano_dist_threshold, args.ignore_rooms)

    # Note: We assume that all necessary features were already extracted and are saved in a file
    image_helper = evaluate_lib.ImageHelper(args.S, args.L, None)
    feature_extractor = evaluate_lib.FeatureExtractor(args.temp_dir, args.multires, args.S, args.L)
    feature_extractor.extract_features(dataset_obj, image_helper, None)

    if args.dataset_distractor:

        print('--> Building distractor dataset')

        dataset_distractor = evaluate_lib.PanoRetrievalDataset(args.dataset_distractor, [], args.view_poses_file_distractor)
        # Set db info
        dataset_distractor.set_retrieval_info('', args.db_file_distractor)
        dataset_distractor.query_indices = np.empty(0, dtype=int)

        # Note: We assume that all necessary features were already extracted and are saved in a file
        feature_extractor_distractor = evaluate_lib.FeatureExtractor(args.temp_dir_distractor, args.multires, args.S, args.L)
        feature_extractor_distractor.extract_features(dataset_distractor, evaluate_lib.ImageHelper(args.S, args.L, None), None)

    # Get mean of full dataset
    if args.dataset_distractor:
        full_dataset = np.concatenate((feature_extractor.features_dataset[0], feature_extractor_distractor.features_dataset[0]))
        dataset_mean = full_dataset.mean(axis=0)
    else:
        dataset_mean = feature_extractor.features_dataset[0].mean(axis=0)

    # Center + normalize dataset
    if args.center_norm_dataset:
        features = feature_extractor.features_dataset[0]
        features -= dataset_mean
        features /= np.sqrt((features * features).sum(axis=1))[:, None]
        if args.dataset_distractor:
            features_distractor = feature_extractor_distractor.features_dataset[0]
            features_distractor -= dataset_mean
            features_distractor /= np.sqrt((features_distractor * features_distractor).sum(axis=1))[:, None]

    # Note: we add an extra "no aggregation" stage in order to keep track of view statistics (will be removed later)
    print('--> Performing feature aggregation on dataset')
    feature_extractor.features_dataset = dataset_obj.aggregate_features(feature_extractor.features_dataset[0],
                                                                        args.feat_aggregation + ['none'],
                                                                        args.pooling,
                                                                        [])

    queries = feature_extractor.features_queries
    if args.center_norm_dataset:
        queries -= dataset_mean

    db_indices = dataset_obj.db_indices
    query_indices = dataset_obj.query_indices

    query_id_to_positive_classes_int = dataset_obj.query_id_to_positive_classes
    query_id_to_positive_classes = {}
    for x in query_id_to_positive_classes_int:
        query_id_to_positive_classes[str(x)] = list(query_id_to_positive_classes_int[x])

    cluster_info = dataset_obj.cluster_info
    view_info = dataset_obj.cluster_info[-1]
    dataset = feature_extractor.features_dataset[-1]
    dataset = dataset.astype(np.float32)

    if args.num_rooms_dataset > 0 and 'room' in args.feat_aggregation:
        assert args.feat_aggregation.count('room') == 1

        # Get panorama information (location, db ids)
        pano_info = dataset_obj.get_pano_info()
        # Remove non-db panoramas
        pano_info = [p for p in pano_info if len(p['db_ids']) > 0]

        # Cluster panorama locations using the desired number of clusters
        pano_locations = np.array([pano['location'] for pano in pano_info])
        kmeans = KMeans(n_clusters=args.num_rooms_dataset, random_state=0).fit(pano_locations)

        # Define clusters for descriptor aggregation
        stage_idx = args.feat_aggregation.index('room')
        cluster_info[stage_idx] = []
        for room_label in range(args.num_rooms_dataset):
            image_ids = []
            for pano_id in np.where(kmeans.labels_ == room_label)[0]:
                image_ids += pano_info[pano_id]['db_ids']
            cluster_info[stage_idx].append({'image_id': None, 'room_id': None, 'pano_id': None, 'image_ids': image_ids})

    if args.dataset_distractor:

        print('--> Performing feature aggregation on distractor dataset')

        index_offset = len(dataset_obj.image_data)
        pano_index_offset = len(set([image['source_pano_filename'] for image in dataset_obj.image_data]))

        # Note: we add an extra "no aggregation" stage in order to keep track of view statistics (will be removed later)
        feature_extractor_distractor.features_dataset = dataset_distractor.aggregate_features(feature_extractor_distractor.features_dataset[0],
                                                                            args.feat_aggregation + ['none'],
                                                                            args.pooling,
                                                                            [])

        # Merge datasets

        dataset_distractor.db_indices += index_offset
        db_indices = np.concatenate((db_indices, dataset_distractor.db_indices))
        assert len(db_indices) == len(set(db_indices)), 'DB indices should be unique'

        for cluster_stage_idx, cluster_stage in enumerate(dataset_distractor.cluster_info):
            for cluster in cluster_stage:
                cluster['image_ids'] = [x + index_offset for x in cluster['image_ids']]
                if cluster['pano_id'] is not None:
                    cluster['pano_id'] += pano_index_offset
            cluster_info[cluster_stage_idx].extend(cluster_stage)

        dataset = np.concatenate((dataset, feature_extractor_distractor.features_dataset[-1]))

    print('--> Building tree')
    # Note: we remove the last "no aggregation" stage
    root = tree_lib.build_tree(db_indices, cluster_info[:-1], view_info, dataset, gmp=args.pooling == 'gmp',
                      normalize=args.normalized_desc)

    if args.bf > 0:
        flann = pyflann.FLANN()
        aggr_dataset = np.array([c.descriptor for c in root.children])
        flann.build_index(aggr_dataset, algorithm="kmeans", branching=args.bf, iterations=15, cb_index=0.0, random_seed=1)
        # Save .flann index to temporary file then load that file to read it as a string
        index_flann_file = tempfile.NamedTemporaryFile(delete=True).name
        flann.save_index(index_flann_file)
        serialized_tree = tree_lib.serialize_flann_index(index_flann_file, aggr_dataset)
        root = tree_lib.deserialize_tree_head(serialized_tree, dataset, view_info, root.children, gmp=args.pooling == 'gmp',
                                              normalize=args.normalized_desc)

    tree_lib.simplify_tree(root)

    # Exploration parameters
    complete_list = True
    all_leaf_exploration_mode = True
    pull_node_at_every_step = False
    use_internal_nodes = False
    if args.normalized_desc:
        dist_fun = tree_lib.cosine_sim
    else:
        dist_fun = tree_lib.l2_sq_dist

    if args.evaluation_mode == 'flann':
        for k in np.exp(np.arange(0, 12, .1)):
            tree_lib.flann_type_evaluation(dist_fun, None, query_indices, query_id_to_positive_classes, queries, root, k,
                                           complete_list, all_leaf_exploration_mode, pull_node_at_every_step, use_internal_nodes)
    elif args.evaluation_mode == 'flann_single':
        tree_lib.flann_type_evaluation(dist_fun, None, query_indices, query_id_to_positive_classes, queries, root, 1,
                                       complete_list, all_leaf_exploration_mode, pull_node_at_every_step, use_internal_nodes)
    elif args.evaluation_mode == 'multistage_rank':
        for num_clusters in range(100):
            tree_lib.multi_stage_evaluation_extended_rank_based(None, query_indices, query_id_to_positive_classes, queries, root,
                                                                num_clusters)
    elif args.evaluation_mode == 'multistage_dist':
        for dist_threshold in np.arange(2.0, -1, -.05):
            tree_lib.multi_stage_evaluation_extended_dist_based(None, query_indices, query_id_to_positive_classes, queries, root,
                                                                dist_threshold)
    else:
        raise NotImplementedError('Unsupported evaluation mode')


if __name__ == '__main__':
    main(parse_arguments())
