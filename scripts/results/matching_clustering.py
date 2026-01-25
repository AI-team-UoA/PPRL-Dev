"""Matching & Clustering Results """
from __future__ import annotations

import os
import pandas as pd
from tqdm import tqdm
# pylint: disable=no-name-in-module
from rustworkx import PyGraph as Graph
import ray
import logging
from utils import load_dataframe, config_bloom_filter, clustering_ray
from pprl.matching import Matcher
from pprl.clustering import AbstractClustering, UniqueMappingClustering, \
    ConnectedComponentsClustering, KiralyMSMApproximateClustering


if __name__ == '__main__':


    DIR = "D9"
    PATH = f"data/ccer/{DIR}"
    FILE = 'dblpclean'
    FILE2 = 'scholarclean'
    attributes = [['title']]
    SEP = ">"


    # DIR = "D2"
    # PATH = f"data/ccer/{DIR}"
    # FILE = 'abtclean'
    # FILE2 = 'buyclean'
    # attributes = [['name']]
    # SEP = "|"

    RESULTS = f"results/matching/{DIR}"
    RESULTS_CLUSTERING = f"results/clustering/{DIR}"

    loaded_dataframe = load_dataframe(PATH, (FILE, FILE2), SEP)
    BLOOM_NAME = 'matching'
    print(f"Running MATCHING & CLUSTERING for {DIR}")


    size = [1024]
    qgrams = [4]
    TOTAL_LENGTH = 32
    futures = []
    MAX_CONCURRENCY = 24
    CLUSTER_MAX_CONCURRENCY = 3


    adjacent_config = {}
    matching_dict = {}
    matching_dict['dice'] = {
        'matching': {1 : [0.05, 0.15, 0.25, 0.35, 0.45]},
        'ConnectedComponentsClustering': {1 : [0.05, 0.15, 0.25, 0.35]},
        'UniqueMappingClustering': {1 : [0.05, 0.15, 0.25]},
        'KiralyMSMApproximateClustering': {1 : [0.05, 0.15, 0.25, 0.35]},
        'skip': {1 : [0.05, 0.15, 0.25, 0.35]}
    }

    if not ray.is_initialized():
        logging.getLogger("ray").setLevel(logging.ERROR)
        ray.init(log_to_driver=False, logging_level=logging.ERROR,
                local_mode=False)

    for configuration, new_bloom in tqdm(
        config_bloom_filter(attributes, size, qgrams,
                BLOOM_NAME, loaded_dataframe, adjacent_config),
        total=TOTAL_LENGTH):

        new_bloom_ref = ray.put(new_bloom)
        for metric in ['dice', 'jaccard', 'cosine']:
            print(f"Running for {metric}")
            graph = None
            for similarity in range(5,100,5):
                if metric in matching_dict and \
                    configuration['num_hashes'] in matching_dict[metric] \
                    ['skip'] and \
                    similarity/100 in matching_dict[metric] \
                    ['skip'][configuration['num_hashes']]:
                    continue
                thr = similarity / 100
                if graph is None:
                    matcher = Matcher(batch_size=10_000, threshold=thr, metric=metric)
                    graph : Graph = matcher.predict(new_bloom)
                else:
                    filter_edges = graph.filter_edges(lambda w, l_thr=thr: w < l_thr)
                    matcher.matcher_config.threshold = thr
                    for edge_index in filter_edges:
                        graph.remove_edge_from_index(edge_index)
                ev = matcher.evaluate(graph, verbose=False)
                matcher_ev = {
                    'name': f'Matching with {metric}',
                    'total_matches': graph.num_edges(),
                    'similarity_threshold': thr}
                for keys, items in ev.items():
                    matcher_ev[keys] = round(items, 2) if isinstance(items, float) else items

                matcher_ev['config'] = configuration
                filename = f"{metric}.csv"
                loaded = os.path.exists(f"{RESULTS}/{filename}")
                df = pd.DataFrame([matcher_ev])
                df.to_csv(f'{RESULTS}/{filename}',
                mode='a+', index=False, header=not loaded)

                graph_ref = ray.put(graph)
                futures = []
                for cl in [ConnectedComponentsClustering,
                        KiralyMSMApproximateClustering,
                        UniqueMappingClustering]:

                    for cl_similarity in tqdm(range(10, 100, 10),
                        desc=f"Clustering {cl.__name__}",
                        total=9):
                        thr_cl = cl_similarity / 100
                        future = clustering_ray.remote(
                            new_bloom_ref, cl,
                            graph_ref, metric,  thr, thr_cl,
                            configuration)
                        futures.append(future)
                        if len(futures) >= CLUSTER_MAX_CONCURRENCY:
                            finished, futures = ray.wait(
                                futures, num_returns=1)
                            for done in finished:
                                clustering_name, evaluations = ray.get(done)
                                filename = f"{clustering_name}_{metric}.csv"
                                loaded = os.path.exists(f"{RESULTS_CLUSTERING}/{filename}")
                                df = pd.DataFrame(evaluations)
                                df.to_csv(f'{RESULTS_CLUSTERING}/{filename}',
                                mode='a+', index=False, header=not loaded)

                while futures:
                    finished, futures = ray.wait(
                        futures, num_returns=1)
                    for done in finished:
                        clustering_name, evaluations = ray.get(done)
                        filename = f"{clustering_name}_{metric}.csv"
                        loaded = os.path.exists(f"{RESULTS_CLUSTERING}/{filename}")
                        df = pd.DataFrame(evaluations)
                        df.to_csv(f'{RESULTS_CLUSTERING}/{filename}',
                        mode='a+', index=False, header=not loaded)

    print(f"Finished {DIR}")
