"""Utils for Results Generation"""
from typing import Dict
from itertools import product
import gc
import os
import pickle
import pandas as pd
import time
from tqdm import tqdm
import ray
import ray.actor
from pprl.matching import Matcher
import numpy as np
from pprl.comparison_cleaning import AbstractMetablocking
from pprl.encoder import BloomFilterConfig, BloomEncodedData, BloomFilter
from pprl.clustering import AbstractClustering, UniqueMappingClustering, \
    ConnectedComponentsClustering, KiralyMSMApproximateClustering

def load_dataframe(path: str, files: tuple, sep: str):
    """Loads Dataframe"""

    file, file2 = files
    d1 = pd.read_csv(f"{path}/{file}.csv", sep=sep)
    d2 = pd.read_csv(f"{path}/{file2}.csv", sep=sep)

    gt = pd.read_csv(f'{path}/gtclean.csv' , sep=sep)
    # Add index columns for lookup
    df_a = d1.reset_index().rename(columns={"index": "index_A"})
    df_b = d2.reset_index().rename(columns={"index": "index_B"})

    df_a = df_a[["index_A", "id"]]
    df_b = df_b[["index_B", "id"]]

    gt_index = gt.merge(left_on='D1', right=df_a, right_on='id')

    gt_index = gt_index.drop(columns=['id', 'D1'])
    gt_index.columns = ['D2', 'D1']

    gt_index = gt_index.merge(left_on='D2', right=df_b, right_on='id')
    gt_index = gt_index.drop(columns=['id', 'D2'])
    gt_index.columns = ['D1', 'D2']


    d1 = d1.astype(str)
    d2 = d2.astype(str)


    return d1, d2, gt_index


def config_bloom_filter_with_configuration(configuration: dict,
                    bloom_name, loaded_dataframe):
    """Configure new bloom filter with configuration"""

    config = BloomFilterConfig(**configuration)
    bl = BloomFilter(config)
    bloom_filters : BloomEncodedData = bl.encode(loaded_dataframe[0])
    bloom_filters.to_file(f'{bloom_name}_1.pkl')

    bloom_filters : BloomEncodedData = bl.encode(loaded_dataframe[1])
    bloom_filters.to_file(f'{bloom_name}_2.pkl')
    new_bloom = BloomEncodedData.from_file(f'{bloom_name}_1.pkl', f'{bloom_name}_2.pkl')

    new_bloom.set_ground_truth(loaded_dataframe[2])
    return new_bloom

def config_bloom_filter(attributes,
        size, qgrams, bloom_name,
        loaded_dataframe, adjacent_config: dict):
    """BF Gerator Method"""

    for a, s, q in product(attributes, size, qgrams):
        for n in range(int(s/(8*q))):
            configuration = {
                'attributes': a,
                'size': s,
                'offset': 0,
                'num_hashes': n+1,
                'hashing_type': 'salted_qgrams',
                'salt': '',
                'qgrams': q
            }

            flag_all_found: bool = True
            for key in adjacent_config:
                if not configuration in adjacent_config[key]:
                    flag_all_found = False
                    break


            if flag_all_found and adjacent_config:
                yield None, None
                continue



            config = BloomFilterConfig(**configuration)
            bl = BloomFilter(config)
            bloom_filters : BloomEncodedData = bl.encode(loaded_dataframe[0])
            bloom_filters.to_file(f'{bloom_name}_1.pkl')

            bloom_filters : BloomEncodedData = bl.encode(loaded_dataframe[1])
            bloom_filters.to_file(f'{bloom_name}_2.pkl')
            new_bloom = BloomEncodedData.from_file(f'{bloom_name}_1.pkl', f'{bloom_name}_2.pkl')

            new_bloom.set_ground_truth(loaded_dataframe[2])
            yield configuration, new_bloom


@ray.remote
def comparison_cleaning_exp(cc: callable, new_bloom: BloomEncodedData,
            configuration: dict):
    """Experiments for comparison Cleaning"""
    wep : AbstractMetablocking = cc()

    metablocking_method = cc.__name__
    metablocking_results = []

    for i in [12]:

        new_candidates = wep.process(encoded_data=new_bloom,
                        adjacent_bits=i)
        ev : dict = wep.evaluate(new_candidates, verbose = False)

        cnt = 0
        for val in new_candidates.values():
            cnt += len(val)

        my_ev = {"name" : metablocking_method,
                'candidate_pairs': cnt,
                'adjacent_bits': i}

        for keys, items in ev.items():
            my_ev[keys] = round(items, 2) if isinstance(items, float) else items

        my_ev['config'] = configuration
        metablocking_results.append(my_ev)

    return metablocking_method, metablocking_results


@ray.remote
def evaluate_encoder_ray(new_bloom: BloomEncodedData,
            adjacent_bits: int, configuration: dict):
    """Experiments for comparison Cleaning"""

    ev: dict = new_bloom.evaluate(adjacent_bits = adjacent_bits,
                    verbose = False)

    my_ev = {'name': 'Encoder',
        'candidate_pairs': new_bloom.calculate_distinct_candidate_pairs(adjacent_bits),
    }

    for keys, items in ev.items():
        my_ev[keys] = round(items, 2) if isinstance(items, float) else items


    my_ev['config'] = configuration
    my_ev['Average Bits %'] = new_bloom.calculate_average_active_bits()


    return adjacent_bits, my_ev



@ray.remote
def candidate_pairs_and_active_bits_ray(new_bloom: BloomEncodedData,
            adjacent_bits: int, index: int):
    """Experiments for comparison Cleaning"""
    if adjacent_bits == 0:
        active_bits_100 = new_bloom.calculate_average_active_bits() * 100
        active_bits_100 = round(active_bits_100, 2)
        return adjacent_bits, active_bits_100, index


    distinct_candidate_pairs = new_bloom. \
                    calculate_distinct_candidate_pairs(adjacent_bits)



    return adjacent_bits, distinct_candidate_pairs, index


@ray.remote
def matching_ray(new_bloom: BloomEncodedData,
            thr: float, metric, configuration: dict):
    """Experiments for Matching"""

    matcher = Matcher(batch_size=100_000, threshold= thr,
            metric=metric)


    src, tgt, weights_list = matcher.predict(new_bloom)

    my_dict = {
        'src': src,
        'tgt': tgt,
        'w_list': weights_list,
    }

    with open(f"tmp/{metric}_{thr}.pkl", 'wb+') as f:
        pickle.dump(my_dict, f)

    return 0

@ray.remote(num_cpus=1)
def clustering_ray(new_bloom: BloomEncodedData,
                   cl: AbstractClustering,
                   graph: tuple,  metric: str, thr: float,
                   thr_cl: float,
                   configuration):
    """Clustering with ray"""
    evaluations = []
    clustering_method : AbstractClustering = cl()
    predicted = clustering_method.process(graph, new_bloom, thr_cl)
    ev_cl : dict = clustering_method.evaluate(predicted, verbose=False)
    cluster_ev = {
        'name': f'Clustering with {cl.__name__}',
        'total_clusters': len(predicted),
        'clustering_threshold': thr_cl,
        'matcher': metric,
        'matching_thr': thr,
    }

    for keys, items in ev_cl.items():
        cluster_ev[keys] = round(items, 2) \
        if isinstance(items, float) else items
    cluster_ev['config'] = configuration

    evaluations.append(cluster_ev)
    return cl.__name__, evaluations


    # clustering_name = cl.__name__

    # cluster = cl()
    # predicted = cluster.process(g, new_bloom, thr_cl)

    # ev_cl : dict = cluster.evaluate(predicted, verbose=False)
    # cluster_ev = {
    #     'name': f'Clustering with {cl.__name__}',
    #     'total_clusters': len(predicted),
    #     'clustering_threshold': thr_cl,
    #     'matcher': metric,
    #     'matching_thr': thr,
    # }
    # for keys, items in ev_cl.items():
    #     cluster_ev[keys] = round(items, 2) \
    #     if isinstance(items, float) else items

    # cluster_ev['config'] = configuration
    # evaluations.append(cluster_ev)

    # return clustering_name, metric, evaluations
    # cnt = graph.num_edges()

    # ev : dict = matcher.evaluate(graph, verbose = False)
    # matcher_ev = {
    #     'name': f'Matching with {metric}',
    #     'total_matches': cnt, 'similarity_threshold': thr}
    # for keys, items in ev.items():
    #     matcher_ev[keys] = round(items, 2) if isinstance(items, float) else items

    # matcher_ev['config'] = configuration

    # evaluations : Dict[str, list] = {}


    # for cl in [UniqueMappingClustering,
    #     ConnectedComponentsClustering, KiralyMSMApproximateClustering]:

    #     cluster = cl()
    #     evaluations[cl.__name__] = []
    #     for similarity_cl in range(10,100, 10):
    #         thr_cl = similarity_cl / 100
    #         predicted = cluster.process(graph, new_bloom, thr_cl)
    #         ev_cl : dict = cluster.evaluate(predicted, verbose=False)
    #         cluster_ev = {
    #             'name': f'Clustering with {cl.__name__}',
    #             'total_clusters': len(predicted),
    #             'clustering_threshold': thr_cl,
    #             'matcher': metric,
    #             'matching_thr': thr,
    #         }
    #         for keys, items in ev_cl.items():
    #             cluster_ev[keys] = round(items, 2) \
    #                 if isinstance(items, float) else items

    #         cluster_ev['config'] = configuration
    #         evaluations[cl.__name__].append(cluster_ev)

    # return matcher_ev, evaluations, metric


def update_graph(graph: tuple, thr: float):
    """Updates graph with new threshold"""
    edges, weights = graph

    del graph
    mask = thr < weights

    new_edges = edges[mask].astype(np.int32)
    del edges

    new_weights = weights[mask].astype(np.float32)
    del weights, mask

    gc.collect()
    return new_edges, new_weights



@ray.remote
def matching_clustering_ray(new_bloom: BloomEncodedData,
                        configuration: dict,
                        metric: str,
                        RESULTS: str, RESULTS_CLUSTERING: str):
    """Matching and Clustering with ray"""

    print(f"Running for {metric}")
    start = time.time()
    similarity = 5
    thr = similarity / 100

    matcher = Matcher(batch_size=100_000,
        threshold = thr,
        metric=metric)
    graph = matcher.predict(new_bloom)


    kiraly = KiralyMSMApproximateClustering()
    connected = ConnectedComponentsClustering()
    unique = UniqueMappingClustering()

    matchers_ev = []
    kiraly_ev = []
    connected_ev = []
    unique_ev = []
    for similarity in range(5, 100, 5):
        if similarity != 5:
            thr = similarity / 100
            graph = update_graph(graph, thr)

        edges, _ = graph

        ev = matcher.evaluate(graph, verbose=False)
        matcher_ev = {
            'name': f'Matching with {metric}',
            'total_matches': edges.shape[0],
            'similarity_threshold': thr}
        for keys, items in ev.items():
            matcher_ev[keys] = round(items, 2) if isinstance(items, float) else items

        matcher_ev['config'] = configuration

        matchers_ev.append(matcher_ev)

        for cl in [connected, kiraly, unique]:
            clustering_name = cl.__class__.__name__
            clusters = cl.process(graph, new_bloom, thr)
            ev_cl = cl.evaluate(clusters, verbose=False)
            cluster_ev = {
                'name': f'Clustering with {clustering_name}',
                'total_clusters': len(clusters),
                'threshold': thr,
                'matcher': metric,
            }
            for keys, items in ev_cl.items():
                cluster_ev[keys] = round(items, 2) if isinstance(items, float) else items
            cluster_ev['config'] = configuration

            if clustering_name == 'ConnectedComponentsClustering':
                connected_ev.append(cluster_ev)
            elif clustering_name == 'KiralyMSMApproximateClustering':
                kiraly_ev.append(cluster_ev)
            elif clustering_name == 'UniqueMappingClustering':
                unique_ev.append(cluster_ev)


    matcher_filename = f"{metric}.csv"
    loaded = os.path.exists(f"{RESULTS}/{matcher_filename}")
    df = pd.DataFrame(matchers_ev)
    df.to_csv(f'{RESULTS}/{matcher_filename}', mode='a+', index=False,
              header=not loaded)


    kiraly_filename = f"{kiraly.__class__.__name__}_{metric}.csv"
    loaded = os.path.exists(f"{RESULTS_CLUSTERING}/{kiraly_filename}")
    df = pd.DataFrame(kiraly_ev)
    df.to_csv(f'{RESULTS_CLUSTERING}/{kiraly_filename}', mode='a+', index=False,
              header=not loaded)

    connected_filename = f"{connected.__class__.__name__}_{metric}.csv"
    loaded = os.path.exists(f"{RESULTS_CLUSTERING}/{connected_filename}")
    df = pd.DataFrame(connected_ev)
    df.to_csv(f'{RESULTS_CLUSTERING}/{connected_filename}', mode='a+', index=False,
              header=not loaded)

    unique_filename = f"{unique.__class__.__name__}_{metric}.csv"
    loaded = os.path.exists(f"{RESULTS_CLUSTERING}/{unique_filename}")
    df = pd.DataFrame(unique_ev)
    df.to_csv(f'{RESULTS_CLUSTERING}/{unique_filename}', mode='a+', index=False,
              header=not loaded)

    return time.time() - start, metric
