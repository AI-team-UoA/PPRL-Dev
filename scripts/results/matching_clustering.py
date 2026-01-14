"""FAISS Results """


import os
import logging
import pandas as pd
from tqdm import tqdm
from pprl.matching import Matcher
from utils import load_dataframe, config_bloom_filter, matching_ray
import ray


def process_done(processed, results, results_clustering):
    """
    Docstring for process_done

    :param done: Description
    """

    matcher_ev, cluster_evaluations, matching_metric = ray.get(processed)
    filename = f"matcher_{matching_metric}.csv"
    loaded = os.path.exists(f"{results}/{filename}")
    df = pd.DataFrame([matcher_ev])
    df.to_csv(f'{results}/{filename}',
    mode='a+', index=False, header=not loaded)


    for key, cl_ev in cluster_evaluations.items():
        filename = f"clustering_{key}_{matching_metric}.csv"
        loaded = os.path.exists(f"{results_clustering}/{filename}")
        df = pd.DataFrame(cl_ev)
        df.to_csv(f'{results_clustering}/{filename}',
        mode='a+', index=False, header=not loaded)



if __name__ == '__main__':


    DIR = "D2"
    PATH = f"data/ccer/{DIR}"
    FILE = 'abtclean'
    FILE2 = 'buyclean'
    attributes = [['name']]
    SEP = "|"

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


    adjacent_config = {}

    if not ray.is_initialized():
        logging.getLogger("ray").setLevel(logging.ERROR)
        ray.init(log_to_driver=False, logging_level=logging.ERROR)

    for configuration, new_bloom in tqdm(
        config_bloom_filter(attributes, size, qgrams,
                BLOOM_NAME, loaded_dataframe, adjacent_config),
        total=TOTAL_LENGTH):


        for metric in ['dice', 'scm']:
            print(f"Running for {metric}")
            for similarity in range(10, 100, 10):
                thr = similarity / 100

                future = matching_ray.remote(new_bloom, thr, metric, configuration)
                futures.append(future)
                if len(futures) >= MAX_CONCURRENCY:
                    finished, futures = ray.wait(futures, num_returns = 1)
                    for done in finished:
                        process_done(done, RESULTS, RESULTS_CLUSTERING)

    while futures:
        finished, futures = ray.wait(futures, num_returns = 1)
        for done in finished:
            process_done(done, RESULTS, RESULTS_CLUSTERING)


    print(f"Finished {DIR}")
