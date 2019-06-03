import evaluate_lib
import tree_lib

import argparse
import numpy as np
import tempfile
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
             'some aggregation Accepted values are: mean (default), gmp'
    )
    parser.add_argument(
        '--tree_pooling',
        type=str,
        required=False,
        default='mean',
        help='Feature pooling strategy used for the internal nodes of the tree.'
             'Accepted values are: mean (default), gmp'
    )
    parser.add_argument(
        '--center_norm_dataset',
        dest='center_norm_dataset',
        action='store_true',
        help='If set, the view descriptors will first be centered and normalized (happens before aggregation)'
    )
    parser.set_defaults(center_norm_dataset=False)
    parser.add_argument(
        '--tree_normalized_desc',
        dest='tree_normalized_desc',
        action='store_true',
        help='If set, the descriptors in the tree will be normalized.'
    )
    parser.set_defaults(tree_normalized_desc=False)
    parser.add_argument(
        '--bf',
        type=int,
        required=False,
        default=8,
        help='Sets value of branching factor used to build the k-means tree.'
    )
    parser.add_argument(
        '--pano_dist_threshold',
        type=float,
        required=False,
        default=0.0,
        help='Distance threshold under which a panorama is considered a match'
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

    # Load the image helper
    image_helper = evaluate_lib.ImageHelper(args.S, args.L, None)

    # Note: We assume that all necessary features were already extracted and are saved in a file
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

    print('--> Performing feature aggregation on dataset')
    feature_extractor.features_dataset = dataset_obj.aggregate_features(feature_extractor.features_dataset[0],
                                                                        args.feat_aggregation,
                                                                        args.pooling,
                                                                        [])

    queries = feature_extractor.features_queries
    if args.center_norm_dataset:
        queries -= dataset_mean

    query_indices = dataset_obj.query_indices

    query_id_to_positive_classes_int = dataset_obj.query_id_to_positive_classes
    query_id_to_positive_classes = {}
    for x in query_id_to_positive_classes_int:
        query_id_to_positive_classes[str(x)] = list(query_id_to_positive_classes_int[x])

    view_info = dataset_obj.cluster_info[-1]
    dataset = feature_extractor.features_dataset[-1]
    dataset = dataset.astype(np.float32)

    if args.dataset_distractor:

        print('--> Performing feature aggregation on distractor dataset')

        index_offset = len(dataset_obj.image_data)
        pano_index_offset = len(set([image['source_pano_filename'] for image in dataset_obj.image_data]))

        feature_extractor_distractor.features_dataset = dataset_distractor.aggregate_features(feature_extractor_distractor.features_dataset[0],
                                                                            args.feat_aggregation,
                                                                            args.pooling,
                                                                            [])

        # Merge datasets

        dataset_distractor.db_indices += index_offset

        for view in dataset_distractor.cluster_info[-1]:
            view['image_ids'] = [x + index_offset for x in view['image_ids']]
            if view['pano_id'] is not None:
                view['pano_id'] += pano_index_offset
        view_info.extend(dataset_distractor.cluster_info[-1])

        dataset = np.concatenate((dataset, feature_extractor_distractor.features_dataset[-1]))

    print('--> Building FLANN tree')

    flann = pyflann.FLANN()
    flann.build_index(dataset, algorithm="kmeans", branching=args.bf, iterations=15, cb_index=0.0, random_seed=1)
    # Save .flann index to temporary file then load that file to read it as a string
    index_flann_file = tempfile.NamedTemporaryFile(delete=True).name
    flann.save_index(index_flann_file)

    serialized_tree = tree_lib.serialize_flann_index(index_flann_file, dataset)

    root = tree_lib.deserialize_tree(serialized_tree, dataset, view_info, gmp=args.tree_pooling == 'gmp', normalize=args.tree_normalized_desc)

    # Exploration parameters
    complete_list = True
    all_leaf_exploration_mode = True
    pull_node_at_every_step = False
    use_internal_nodes = False
    if args.tree_normalized_desc:
        dist_fun = tree_lib.cosine_sim
    else:
        dist_fun = tree_lib.l2_sq_dist

    for k in np.exp(np.arange(0, 12, .1)):
        tree_lib.flann_type_evaluation(dist_fun, None, query_indices, query_id_to_positive_classes, queries, root, k,
                                       complete_list, all_leaf_exploration_mode, pull_node_at_every_step, use_internal_nodes)


if __name__ == '__main__':
    main(parse_arguments())
