"""Utils for Results Generation"""
from typing import Dict
from itertools import product
import pandas as pd
import ray
import ray.actor
import networkx
from pprl.matching import Matcher

from pprl.comparison_cleaning import AbstractMetablocking
from pprl.encoder import BloomFilterConfig, BloomEncodedData, BloomFilter
from pprl.clustering import UniqueMappingClustering, \
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

    ev: dict = new_bloom.serial_evaluate(adjacent_bits = adjacent_bits,
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

    matcher = Matcher(batch_size=1_000, threshold= thr,
            metric=metric)

    graph : networkx.Graph = matcher.predict(new_bloom)
    cnt = graph.number_of_edges()

    ev : dict = matcher.evaluate(graph, verbose = False)
    matcher_ev = {
        'name': f'Matching with {metric}',
        'total_matches': cnt, 'similarity_threshold': thr}
    for keys, items in ev.items():
        matcher_ev[keys] = round(items, 2) if isinstance(items, float) else items

    matcher_ev['config'] = configuration

    evaluations : Dict[str, list] = {}


    for cl in [UniqueMappingClustering,
        ConnectedComponentsClustering, KiralyMSMApproximateClustering]:

        cluster = cl()
        evaluations[cl.__name__] = []
        for similarity_cl in range(10,100, 10):
            thr_cl = similarity_cl / 100
            predicted = cluster.process(graph, new_bloom, thr_cl)
            ev_cl : dict = cluster.evaluate(predicted, verbose=False)
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
            evaluations[cl.__name__].append(cluster_ev)

    return matcher_ev, evaluations, metric
