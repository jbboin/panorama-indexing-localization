import numpy as np
import os
import sys
import threading
from collections import Counter

try:
    from Queue import PriorityQueue  # Python 2
except ImportError:
    from queue import PriorityQueue    # Python 3

import pyflann
import faiss


class Node(object):
    def __init__(self, descriptor=None, pano_list=[], ids=[]):
        """

        :param descriptor:
        :param pano_list:
        :param ids:
        """
        self.descriptor = descriptor
        self.ids = ids
        self.pano_list = pano_list
        self.children = []
        self.leaf = False

    def add_child(self, obj):
        self.children.append(obj)


def compute_ap(matches, num_pos_matches= None):
    """

    :param matches: list
    :param num_pos_matches: int
    :return:
    """
    pos_indices = np.where(matches)[0]
    if num_pos_matches is None:
        num_pos_matches = len(pos_indices)
    else:
        assert num_pos_matches >= len(pos_indices)
    if num_pos_matches == 0:
        return 0
    old_recall = 0.0
    old_precision = 1.0
    a_p = 0.0
    intersect_size = 0
    j = 0
    for x in matches:
        if x:
            intersect_size += 1
        recall = float(intersect_size) / num_pos_matches
        precision = float(intersect_size) / (j + 1)
        a_p += (recall - old_recall) * ((old_precision + precision)/2.0)
        old_recall = recall
        old_precision = precision
        j += 1
    return a_p


def compute_patk(matches, k):
    """

    :param matches: list
    :param k: float
    :return: float
    """
    num_matches = 0
    for x in matches[:k]:
        if x:
            num_matches += 1
    return float(num_matches)/k


def compute_descriptor(x, gmp=False, normalize=True):
    if gmp:
        descriptor = np.mean(np.linalg.solve(np.dot(x, x.T) + np.eye(x.shape[0]), x), axis=0)
    else:
        descriptor = np.mean(x, axis=0)
    if normalize:
        descriptor /= np.sqrt((descriptor * descriptor).sum())
    return descriptor


def build_tree(db_indices, cluster_info, view_info, dataset, gmp=False, normalize=True):
    # Build inverted index
    db_inv = {x:i for i, x in enumerate(db_indices)}
    # Initialize dict that contains
    cluster_ids_stages = {}
    for x in db_indices:
        cluster_ids_stages[x] = []
    # Define all nodes at each stage
    nodes = []
    for cluster_info_stage_idx, cluster_info_stage in enumerate(cluster_info):
        nodes_stage = []
        leaf_val = (cluster_info_stage_idx == len(cluster_info)-2)
        for cluster_id, cluster in enumerate(cluster_info_stage):
            for x in cluster['image_ids']:
                cluster_ids_stages[x].append(cluster_id)
            image_ids = [db_inv[x] for x in cluster['image_ids']]
            descriptor = compute_descriptor(dataset[image_ids], gmp, normalize)
            pano_counter = Counter()
            for image_id in image_ids:
                pano_id = view_info[image_id]['pano_id']
                pano_counter[pano_id] += 1
            pano_list = [c for c, _ in pano_counter.most_common()]
            node = Node(descriptor, pano_list, image_ids)
            node.leaf = leaf_val
            nodes_stage.append(node)
        nodes.append(nodes_stage)
    # Link nodes from different stages together
    root = Node()
    for node in nodes[0]:
        root.add_child(node)
    if len(cluster_info) == 1:
        root.leaf = True
    for stage, cluster_info_stage in enumerate(cluster_info[:-1]):
        for cluster_id, cluster in enumerate(cluster_info_stage):
            node = nodes[stage][cluster_id]
            # Get ids of clusters at next stage
            cluster_ids = set([cluster_ids_stages[x][stage+1] for x in cluster['image_ids']])
            # Add children corresponding to these sub-clusters
            for c in cluster_ids:
                node.add_child(nodes[stage+1][c])
    return root


# Build tree from serialized string
def deserialize_tree(serialized_tree, dataset, view_info, gmp=False, normalize=True):
    branching = int(serialized_tree.split(',')[0])
    node_list = serialized_tree.split(',')[1].strip().split(' ')
    root, _ = deserialize_tree_helper(node_list, branching, dataset, view_info, gmp, normalize, True)
    assert(len(node_list) == 0)
    return root


def deserialize_tree_helper(node_list, branching, dataset, view_info, gmp, normalize, is_root):
    val = node_list.pop(0)
    node = Node()
    if val == '.':
        image_ids = []
        for i in range(branching):
            child_node, child_image_ids = deserialize_tree_helper(node_list, branching, dataset, view_info, gmp, normalize, False)
            node.add_child(child_node)
            image_ids = image_ids + child_image_ids
    else:
        image_ids = [int(id) for id in val.split('/')]
        node.leaf = True
        for id in val.split('/'):
            node.add_child(Node(descriptor=dataset[int(id)], pano_list=[view_info[int(id)]['pano_id']], ids=[int(id)]))
    if not is_root:
        # Compute descriptor
        node.descriptor = compute_descriptor(dataset[image_ids], gmp, normalize)
        # Get list of panoramas
        pano_counter = Counter()
        for image_id in image_ids:
            pano_id = view_info[image_id]['pano_id']
            pano_counter[pano_id] += 1
        pano_list = [c for c, _ in pano_counter.most_common()]
        node.pano_list = pano_list
        node.ids = image_ids
    return node, image_ids


# Is used to add more stages before the first stage of the tree.
# Takes as an input the serialized string corresponding to the clustering of this first stage, as well as the nodes
# that constitute this stage, and it returns the new tree with these added stages
def deserialize_tree_head(serialized_tree, dataset, view_info, tail, gmp=False, normalize=True):
    branching = int(serialized_tree.split(',')[0])
    node_list = serialized_tree.split(',')[1].strip().split(' ')
    root, _ = deserialize_tree_head_helper(node_list, branching, dataset, view_info, tail, gmp, normalize, True)
    assert(len(node_list) == 0)
    return root


def deserialize_tree_head_helper(node_list, branching, dataset, view_info, tail, gmp, normalize, is_root):
    val = node_list.pop(0)
    node = Node()
    if val == '.':
        image_ids = []
        for i in range(branching):
            child_node, child_image_ids = deserialize_tree_head_helper(node_list, branching, dataset, view_info, tail, gmp, normalize, False)
            node.add_child(child_node)
            image_ids += child_image_ids
    else:
        image_ids = []
        for id in val.split('/'):
            image_ids += tail[int(id)].ids
            node.add_child(tail[int(id)])
    if not is_root:
        # Compute descriptor
        node.descriptor = compute_descriptor(dataset[image_ids], gmp, normalize)
        # Get list of panoramas
        pano_counter = Counter()
        for image_id in image_ids:
            pano_id = view_info[image_id]['pano_id']
            pano_counter[pano_id] += 1
        pano_list = [c for c, _ in pano_counter.most_common()]
        node.pano_list = pano_list
        node.ids = image_ids
    return node, image_ids


# Utility function: simplify tree by removing internal nodes that have only one child
def simplify_tree(node):
    if len(node.children) == 0:
        return
    for child in node.children:
        simplify_tree(child)
    if len(node.children) == 1 and not node.leaf:
        node.leaf = node.children[0].leaf
        node.children = node.children[0].children


# Evaluate queries

def single_stage_evaluation(query_indices, query_id_to_positive_classes, queries, nodes):
    ap_values = []
    pat1_values = []
    for q, query_idx in enumerate(query_indices):
        pos_classes = query_id_to_positive_classes[str(query_idx)]
        query_desc = queries[q]
        dists = np.array([query_desc.dot(node.descriptor) for node in nodes])
        pano_retrieved_list = []
        for desc_idx in np.argsort(dists)[::-1]:
            node = nodes[desc_idx]
            for pano_ids in node.pano_list:
                if pano_ids not in pano_retrieved_list:
                    pano_retrieved_list.append(pano_ids)
        matches = np.isin(pano_retrieved_list, pos_classes)
        ap_values.append(compute_ap(matches))
        pat1_values.append(compute_patk(matches, 1))
    m_a_p = np.mean(ap_values) * 100
    m_pat1 = np.mean(pat1_values) * 100
    print('%.2f,%.2f' % (m_a_p, m_pat1))


def multi_stage_evaluation(stage_names,
                           query_indices,
                           query_id_to_positive_classes,
                           queries,
                           root,
                           num_clusters,
                           use_ratio=False):
    """

    :param stage_names:
    :param query_indices:
    :param query_id_to_positive_classes:
    :param queries:
    :param root:
    :param num_clusters:
    :param use_ratio:
    :return:
    """
    ap_values = []
    pat1_values = []
    num_desc = []
    for q, query_idx in enumerate(query_indices):
        pos_classes = query_id_to_positive_classes[str(query_idx)]
        query_desc = queries[q]
        nodes_stage = root.children
        ranked_nodes = []
        num_desc_q = 0
        for stage in range(len(num_clusters)+1):
            num_desc_q += len(nodes_stage)
            dists = np.array([query_desc.dot(node.descriptor) for node in nodes_stage])
            ranked_node_idx = np.argsort(dists)[::-1]
            # Add unused nodes to ranked_nodes
            if stage == len(num_clusters):
                ranked_nodes = [nodes_stage[n] for n in ranked_node_idx] + ranked_nodes
            else:
                if use_ratio:
                    k = int(round(len(ranked_node_idx) * num_clusters[stage]))
                else:
                    k = int(round(num_clusters[stage]))
                ranked_nodes = [nodes_stage[n] for n in ranked_node_idx[k:]] + ranked_nodes
                # Explore top nodes
                nodes_stage_next = []
                for n in ranked_node_idx[:k]:
                    nodes_stage_next.extend(nodes_stage[n].children)
                nodes_stage = nodes_stage_next
        pano_retrieved_list = []
        for node in ranked_nodes:
            for pano_ids in node.pano_list:
                if pano_ids not in pano_retrieved_list:
                    pano_retrieved_list.append(pano_ids)
        matches = np.isin(pano_retrieved_list, pos_classes)
        ap_values.append(compute_ap(matches))
        pat1_values.append(compute_patk(matches, 1))
        num_desc.append(num_desc_q)
    m_a_p = np.mean(ap_values) * 100
    m_pat1 = np.mean(pat1_values) * 100
    print('%s,db_48,%s,%.2f,%.2f,%.2f,0.00,0.00' % ('/'.join(stage_names), '/'.join([str(x) for x in num_clusters]), np.mean(num_desc), m_a_p, m_pat1))


def multi_stage_evaluation_extended_rank_based(stage_names,
                                               query_indices,
                                               query_id_to_positive_classes,
                                               queries,
                                               root,
                                               num_clusters):
    """
    Version of multi_stage_evaluation that only allows a fixed number of clusters per stage, but it supports all types
    of trees, even if leaf nodes aren't all at the same height. Gives the same results as multi_stage_evaluation if the
    number of clusters per stage is set to a fixed value in multi_stage_evaluation.
    :param stage_names:
    :param query_indices:
    :param query_id_to_positive_classes:
    :param queries:
    :param root:
    :param num_clusters:
    :return:
    """
    ap_values = []
    pat1_values = []
    num_desc = []
    for q, query_idx in enumerate(query_indices):
        pos_classes = query_id_to_positive_classes[str(query_idx)]
        query_desc = queries[q]
        nodes_stage = root.children
        ranked_nodes = []
        leaf_nodes = []
        num_desc_q = 0
        while len(nodes_stage) > 0:
            num_desc_q += len(nodes_stage)
            dists = np.array([query_desc.dot(node.descriptor) for node in nodes_stage])
            ranked_node_idx = np.argsort(dists)[::-1]
            nodes_stage_next = []
            # We explore num_clusters internal nodes and keep leaf nodes in separate list
            num_explored = 0
            ranked_nodes_stage = []
            for n in ranked_node_idx:
                if len(nodes_stage[n].children) == 0:
                    leaf_nodes.append((dists[n], nodes_stage[n]))
                elif num_explored < num_clusters:
                    nodes_stage_next.extend(nodes_stage[n].children)
                    num_explored += 1
                else:
                    ranked_nodes_stage.append((dists[n], nodes_stage[n]))
            ranked_nodes = ranked_nodes_stage + ranked_nodes
            nodes_stage = nodes_stage_next
        leaf_nodes.sort(reverse=True)
        ranked_nodes = [n for d,n in leaf_nodes] + [n for d,n in ranked_nodes]
        pano_retrieved_list = []
        for node in ranked_nodes:
            for pano_ids in node.pano_list:
                if pano_ids not in pano_retrieved_list:
                    pano_retrieved_list.append(pano_ids)
        matches = np.isin(pano_retrieved_list, pos_classes)
        ap_values.append(compute_ap(matches, len(pos_classes)))
        pat1_values.append(compute_patk(matches, 1))
        num_desc.append(num_desc_q)
    m_a_p = np.mean(ap_values) * 100
    m_pat1 = np.mean(pat1_values) * 100
    if stage_names is not None:
        print('%s,db_48,%s,%.2f,%.2f,%.2f,0.00,0.00' % ('/'.join(stage_names), str(num_clusters), np.mean(num_desc), m_a_p, m_pat1))
    else:
        print('%.2f,%.2f,%.2f' % (np.mean(num_desc), m_a_p, m_pat1))


def multi_stage_evaluation_extended_dist_based(stage_names,
                                               query_indices,
                                               query_id_to_positive_classes,
                                               queries,
                                               root,
                                               dist_threshold):
    """
    Version of multi_stage_evaluation_extended_rank_based that uses the distribution of distances at each stage, instead
    of the top k
    :param stage_names:
    :param query_indices:
    :param query_id_to_positive_classes:
    :param queries:
    :param root:
    :param dist_threshold:
    :return:
    """
    ap_values = []
    pat1_values = []
    num_desc = []
    for q, query_idx in enumerate(query_indices):
        pos_classes = query_id_to_positive_classes[str(query_idx)]
        query_desc = queries[q]
        nodes_stage = root.children
        ranked_nodes = []
        leaf_nodes = []
        num_desc_q = 0
        while len(nodes_stage) > 0:
            num_desc_q += len(nodes_stage)
            dists = np.array([query_desc.dot(node.descriptor) for node in nodes_stage])
            dists -= np.mean(dists)
            dists /= np.std(dists)
            ranked_node_idx = np.argsort(dists)[::-1]
            nodes_stage_next = []
            # We explore the best internal nodes and keep leaf nodes in separate list
            ranked_nodes_stage = []
            for n in ranked_node_idx:
                if len(nodes_stage[n].children) == 0:
                    leaf_nodes.append((dists[n], nodes_stage[n]))
                elif dists[n] > dist_threshold:
                   nodes_stage_next.extend(nodes_stage[n].children)
                else:
                    ranked_nodes_stage.append((dists[n], nodes_stage[n]))
            ranked_nodes = ranked_nodes_stage + ranked_nodes
            nodes_stage = nodes_stage_next
        leaf_nodes.sort(reverse=True)
        ranked_nodes = [n for d,n in leaf_nodes] + [n for d,n in ranked_nodes]
        pano_retrieved_list = []
        for node in ranked_nodes:
            for pano_ids in node.pano_list:
                if pano_ids not in pano_retrieved_list:
                    pano_retrieved_list.append(pano_ids)
        matches = np.isin(pano_retrieved_list, pos_classes)
        ap_values.append(compute_ap(matches, len(pos_classes)))
        pat1_values.append(compute_patk(matches, 1))
        num_desc.append(num_desc_q)
    m_a_p = np.mean(ap_values) * 100
    m_pat1 = np.mean(pat1_values) * 100
    if stage_names is not None:
        print('%s,db_48,%.2f,%.2f,%.2f,%.2f,0.00,0.00' % ('/'.join(stage_names), dist_threshold, np.mean(num_desc), m_a_p, m_pat1))
    else:
        print('%.2f,%.2f,%.2f' % (np.mean(num_desc), m_a_p, m_pat1))


def cosine_sim(x, y):
    return -x.dot(y)


def l2_sq_dist(x, y):
    return (x-y).dot(x-y)


def flann_pano_list_retrieval(root,
                              query_desc,
                              dist_fun,
                              num_panos,
                              leaf_nodes_max,
                              complete_list=True,
                              all_leaf_exploration_mode=False,
                              pull_node_at_every_step=False,
                              use_internal_nodes=False):
    """

    :param root:
    :param query_desc:
    :param dist_fun:
    :param num_panos:
    :param leaf_nodes_max:
    :param complete_list:
    :param all_leaf_exploration_mode:
    :param pull_node_at_every_step:
    :param use_internal_nodes:
    :return:
    """
    num_desc_q = 0

    branch_queue = PriorityQueue()
    branch_queue.put((0, root))
    leaf_nodes_explored = 0
    leaf_nodes = []
    while not branch_queue.empty() and leaf_nodes_explored < leaf_nodes_max:
        dist, node = branch_queue.get()
        while True:
            if use_internal_nodes and node.leaf and len(node.pano_list) == 1:
                leaf_nodes.append((dist, node))
            if not all_leaf_exploration_mode:
                if len(node.children) > 0:
                    # We don't re-count clusters with only one single child
                    if len(node.children) > 1:
                        num_desc_q += len(node.children)
                    dists = np.array([dist_fun(query_desc, child.descriptor) for child in node.children])
                    if pull_node_at_every_step:
                        for i, child in enumerate(node.children):
                            branch_queue.put((dists[i], child))
                        break
                    else:
                        best_idx = np.argmin(dists)
                        for i, child in enumerate(node.children):
                            if i != best_idx:
                                branch_queue.put((dists[i], child))
                        node = node.children[best_idx]
                        dist = dists[best_idx]
                else:
                    leaf_nodes_explored += 1
                    leaf_nodes.append((dist, node))
                    break
            else:
                if not node.leaf:
                    num_desc_q += len(node.children)
                    dists = np.array([dist_fun(query_desc, child.descriptor) for child in node.children])
                    if pull_node_at_every_step:
                        for i, child in enumerate(node.children):
                            branch_queue.put((dists[i], child))
                        break
                    else:
                        best_idx = np.argmin(dists)
                        for i, child in enumerate(node.children):
                            if i != best_idx:
                                branch_queue.put((dists[i], child))
                        node = node.children[best_idx]
                        dist = dists[best_idx]
                else:
                    # We don't re-count clusters with only one sample
                    if len(node.children) > 1:
                        num_desc_q += len(node.children)
                    leaf_nodes_explored += len(node.children)
                    for child in node.children:
                        leaf_nodes.append((dist_fun(query_desc, child.descriptor), child))
                    break

    ranked_nodes = sorted(leaf_nodes)
    pano_retrieved_list = []
    for _, node in ranked_nodes:
        for pano_id in node.pano_list:
            if pano_id not in pano_retrieved_list:
                pano_retrieved_list.append(pano_id)

    # Add non-retrieved panoramas if needed
    if complete_list:
        while len(pano_retrieved_list) < num_panos and not branch_queue.empty():
            _, node = branch_queue.get()
            pano_retrieved_list += [i for i in node.pano_list if i not in pano_retrieved_list]

    return pano_retrieved_list, num_desc_q


def evaluate_ranked_lists(pano_retrieved_lists, pos_classes_lists):
    ap_values = []
    pat1_values = []
    for pano_retrieved, pos_classes in zip(pano_retrieved_lists, pos_classes_lists):
        matches = np.isin(pano_retrieved, pos_classes)
        ap_values.append(compute_ap(matches, len(pos_classes)))
        pat1_values.append(compute_patk(matches, 1))
    m_a_p = np.mean(ap_values) * 100
    m_pat1 = np.mean(pat1_values) * 100
    return m_a_p, m_pat1


def flann_type_evaluation(dist_fun,
                          stage_names,
                          query_indices,
                          query_id_to_positive_classes,
                          queries,
                          root,
                          leaf_nodes_max,
                          complete_list=True,
                          all_leaf_exploration_mode=False,
                          pull_node_at_every_step=False,
                          use_internal_nodes=False):
    """

    :param dist_fun:
    :param stage_names:
    :param query_indices:
    :param query_id_to_positive_classes:
    :param queries:
    :param root:
    :param leaf_nodes_max:
    :param complete_list:
    :param all_leaf_exploration_mode:
    :param pull_node_at_every_step:
    :param use_internal_nodes:
    :return:
    """
    # Get number of panoramas (for early stopping when getting the retrieved lists)
    pano_ids = set()
    for node in root.children:
        pano_ids |= set(node.pano_list)
    num_panos = len(pano_ids)

    pos_classes = []
    for query_idx in query_indices:
        pos_classes.append(query_id_to_positive_classes[str(query_idx)])

    num_desc = []
    pano_retrieved_lists = []
    for query_desc in queries:
        pano_retrieved_list, num_desc_q = flann_pano_list_retrieval(root, query_desc, dist_fun, num_panos, leaf_nodes_max,
                                                                    complete_list, all_leaf_exploration_mode,
                                                                    pull_node_at_every_step, use_internal_nodes)
        num_desc.append(num_desc_q)
        pano_retrieved_lists.append(pano_retrieved_list)

    m_a_p, m_pat1 = evaluate_ranked_lists(pano_retrieved_lists, pos_classes)

    if stage_names is not None:
        print('%s,%d,%.2f,%.2f,%.2f' % ('/'.join(stage_names), leaf_nodes_max, np.mean(num_desc), m_a_p, m_pat1))
    else:
        print('%.2f,%.2f,%.2f' % (np.mean(num_desc), m_a_p, m_pat1))


def pq_evaluation(query_indices, query_id_to_positive_classes, view_info, queries, dataset, num_bytes):
    """
    Evalutes with respect to mAP and at position 1 the returned results from FAISS
    :param query_indices:
    :param query_id_to_positive_classes:
    :param view_info:
    :param queries:
    :param dataset:
    :param num_bytes:
    :return:
    """
    pos_classes = []
    for query_idx in query_indices:
        pos_classes.append(query_id_to_positive_classes[str(query_idx)])

    dataset = dataset.astype(np.float32)
    index = faiss.index_factory(dataset.shape[1], "PQ%d" % num_bytes)
    index.train(dataset)
    index.add(dataset)

    k = dataset.shape[0]
    _, query_nns = index.search(queries, k)

    pano_ids = np.array([x['pano_id'] for x in view_info])
    pano_retrieved_lists = []
    for nn_ids in query_nns:
        pano_retrieved_list_duplicates = pano_ids[nn_ids]
        _, idx = np.unique(pano_retrieved_list_duplicates, return_index=True)
        pano_retrieved_list = pano_retrieved_list_duplicates[np.sort(idx)]
        pano_retrieved_lists.append(pano_retrieved_list)

    m_a_p, m_pat1 = evaluate_ranked_lists(pano_retrieved_lists, pos_classes)

    print('%.2f,%.2f' % (m_a_p, m_pat1))


def serialize_flann_index(index_file_name, dataset):
    """

    :param index_file_name: str
    :param dataset: np.ndarray
    :return:
    """
    global captured_stdout
    # Create pipe and dup2() the write end of it on top of stdout, saving a copy
    # of the old stdout
    stdout_fileno = sys.stdout.fileno()
    stdout_save = os.dup(stdout_fileno)
    stdout_pipe = os.pipe()
    os.dup2(stdout_pipe[1], stdout_fileno)
    os.close(stdout_pipe[1])

    captured_stdout = ''

    def drain_pipe():
        global captured_stdout
        while True:
            data = os.read(stdout_pipe[0], 1024)
            if not data:
                break
            captured_stdout += data

    t = threading.Thread(target=drain_pipe)
    t.start()

    # Load FLANN index so as to print it on stdout
    flann = pyflann.FLANN()
    flann.load_index(index_file_name, dataset)

    # Close the write end of the pipe to unblock the reader thread and trigger it
    # to exit
    os.close(stdout_fileno)
    t.join()

    # Clean up the pipe and restore the original stdout
    os.close(stdout_pipe[0])
    os.dup2(stdout_save, stdout_fileno)
    os.close(stdout_save)

    return captured_stdout