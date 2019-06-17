#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Extraction of features
"""

import evaluate_lib
import tree_lib

import argparse
import numpy as np
import os
import caffe


def parse_arguments():
    parser = argparse.ArgumentParser(description='Extract features for dataset')
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Path to the dataset directory (should contain image files)'
    )
    parser.add_argument(
        '--temp_dir',
        type=str,
        required=True,
        help='Path to a temporary directory to store features'
    )
    parser.add_argument(
        '--view_poses_file',
        type=str,
        required=False,
        default='view_poses.json',
        help='File with poses of views, their filenames, inds etc.'
    )
    parser.add_argument(
        '--proto',
        type=str,
        required=True,
        help='Path to the prototxt file containing the model'
    )
    parser.add_argument(
        '--weights',
        type=str,
        required=True,
        help='Path to the caffemodel file'
    )
    args = parser.parse_args()
    return args


def main(args):
    # Parameters left as default
    args.S = 800
    args.L = 2
    args.multires = False
    args.gpu = 0

    print('--> Building dataset')

    dataset_obj = evaluate_lib.PanoRetrievalDataset(args.dataset, [], args.view_poses_file)
    # Set query / db info
    dataset_obj.set_retrieval_info('', '')
    # Set ground truth info
    dataset_obj.set_ground_truth_info(0.0)

    # Load and reshape the means to subtract to the inputs
    args.means = np.array([103.93900299, 116.77899933, 123.68000031], dtype=np.float32)[None, :, None, None]

    # Configure caffe and load the network
    caffe.set_device(args.gpu)
    caffe.set_mode_gpu()
    net = caffe.Net(args.proto, args.weights, caffe.TEST)

    # Load the image helper
    image_helper = evaluate_lib.ImageHelper(args.S, args.L, args.means)

    # Extract features
    if not os.path.exists(args.temp_dir):
        os.makedirs(args.temp_dir)
    feature_extractor = evaluate_lib.FeatureExtractor(args.temp_dir, args.multires, args.S, args.L)
    feature_extractor.extract_features(dataset_obj, image_helper, net)


if __name__ == '__main__':
    main(parse_arguments())
