"""Matching & Clustering Results """
from __future__ import annotations
import ast
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
    CLUSTER_MAX_CONCURRENCY = 5


    adjacent_config = {}

    matched_done = set()
    clustered_done = set()
    skip = set()
    for metric in ['dice']:
        matching_df = pd.read_csv(f"{RESULTS}/{metric}.csv")
        matching_df['config'] = matching_df['config'].apply(ast.literal_eval)
        matching_df['metric'] = metric
        matching_df['num_hashes'] = matching_df['config'].apply(lambda x: x['num_hashes'])
        list_done = set(zip(
            matching_df['metric'],
            matching_df['num_hashes'],
            matching_df['similarity_threshold']))
        matched_done.update(list_done)
        max_num_hashes = matching_df['num_hashes'].max()


        def clustering_done_list(clustering_method: str):
            filename = f"{clustering_method}_{metric}.csv"
            if not os.path.exists(f"{RESULTS_CLUSTERING}/{filename}"):
                return False
            clustering_df = pd.read_csv(f"{RESULTS_CLUSTERING}/"
                f"{clustering_method}_{metric}.csv")
            clustering_df['config'] = clustering_df['config'].apply(
                ast.literal_eval)
            clustering_df['num_hashes'] = clustering_df['config'].apply(
                lambda x: x['num_hashes'])
            clustering_df['method'] = clustering_method
            list_done_cl = set(zip(
                    clustering_df['method'],
                    clustering_df['matcher'],
                    clustering_df['num_hashes'],
                    clustering_df['matching_thr'],
                    clustering_df['clustering_threshold']))
            return list_done_cl


        for hashing in range(1, max_num_hashes + 1):
            flag = True
            for similarity in range(5, 100, 5):
                if flag == False:
                    break
                if (metric, hashing, similarity / 100) not in matched_done:
                    flag = False
                    break
                for cl_method in [
                    'ConnectedComponentsClustering',
                    'UniqueMappingClustering',
                    'KiralyMSMApproximateClustering']:
                    cl_done = clustering_done_list(cl_method)
                    clustered_done.update(cl_done if cl_done else set())
                    if flag == False:
                        continue
                    if cl_done:
                        for cl_similarity in range(10, 100, 10):
                            if (cl_method, metric, hashing,
                                similarity / 100,
                                cl_similarity / 100) not in cl_done:
                                flag = False
                                break
                    else:
                        flag = False
                        break
            if flag:
                skip.add((metric, hashing))



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
            if (metric, configuration['num_hashes']) in skip:
                continue
            print(f"Running for {metric}")
            graph = None
            for similarity in range(5,100,5):
                thr = similarity / 100
                if graph is None:
                    matcher = Matcher(batch_size=10_000, threshold=thr, metric=metric)
                    graph = matcher.predict(new_bloom)
                else:
                    edges, weights = graph
                    mask = thr <= weights
                    edges = edges[mask]
                    weights = weights[mask]
                    graph = edges, weights
                    matcher.matcher_config.threshold = thr

                if (metric, configuration['num_hashes'], similarity / 100) not in matched_done:

                    edges, _ = graph
                    ev = matcher.evaluate(graph, verbose=True)
                    matcher_ev = {
                        'name': f'Matching with {metric}',
                        'total_matches': edges.shape[0],
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
                        if (cl.__name__, metric,
                            configuration['num_hashes'],
                            thr,
                            cl_similarity / 100) in clustered_done:
                            continue
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
