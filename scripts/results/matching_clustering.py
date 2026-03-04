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
from utils import load_dataframe, config_bloom_filter_with_configuration,\
    config_bloom_filter, clustering_ray, matching_clustering_ray
from pprl.matching import Matcher
from pprl.clustering import AbstractClustering, UniqueMappingClustering, \
    ConnectedComponentsClustering, KiralyMSMApproximateClustering





if __name__ == '__main__':


    DIR = "D10"
    PATH = f"data/ccer/{DIR}"
    FILE = 'imdbclean'
    FILE2 = 'dbpediaclean'
    attributes = ['title', 'starring']
    SEP = "|"


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


    size = 1024 // 2
    qgrams = 4
    TOTAL_LENGTH = 32
    futures = []
    MAX_CONCURRENCY = 3
    # CLUSTER_MAX_CONCURRENCY = 3

    futures = []
    adjacent_config = {}

    matched_done = set()
    clustered_done = set()
    skip = set()

    if not ray.is_initialized():
        logging.getLogger("ray").setLevel(logging.ERROR)
        ray.init(log_to_driver=False, logging_level=logging.ERROR,
                local_mode=False)

    for num_hashes in tqdm(range(11, 33), total=33-11):
        configuration = {'attributes': attributes,
            'size': size,
            'offset': 0,
            'num_hashes': num_hashes,
            'hashing_type': 'salted_qgrams',
            'salt': '',
            'qgrams': qgrams
        }

        new_bloom = config_bloom_filter_with_configuration(configuration,
                    BLOOM_NAME, loaded_dataframe)

        new_bloom_ref = ray.put(new_bloom)


        for metric in ['dice', 'jaccard', 'cosine']:
            print(f"Scheduling MATCHING & CLUSTERING for {DIR} with {metric} and {num_hashes} hashes")
            future = matching_clustering_ray.remote(new_bloom_ref, configuration,
                        metric, RESULTS, RESULTS_CLUSTERING)
            futures.append(future)
            if len(futures) == 2:
                finished, futures = ray.wait(
                    futures, num_returns=1)
                for done in finished:
                    time_taken, metric = ray.get(done)
                    hours, remainder = divmod(time_taken, 3600)
                    minutes, seconds = divmod(remainder, 60)
                    print(f"Finished {metric} in {int(hours)}h {int(minutes)}m {int(seconds)}s")
                
        while futures:
            finished, futures = ray.wait(
                futures, num_returns=1)
            for done in finished:
                time_taken, metric = ray.get(done)
                hours, remainder = divmod(time_taken, 3600)
                minutes, seconds = divmod(remainder, 60)
                print(f"Finished {metric} in {int(hours)}h {int(minutes)}m {int(seconds)}s")

    print(f"Finished {DIR}")
