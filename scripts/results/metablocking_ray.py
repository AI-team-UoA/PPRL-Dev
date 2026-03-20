"""FAISS Results """

import logging
import os
import pandas as pd
from tqdm import tqdm
import ray
from utils import load_dataframe, config_bloom_filter, comparison_cleaning_exp

from privjedai.comparison_cleaning import (
    WeightedEdgePruning,
    WeightedNodePruning,
    CardinalityEdgePruning,
    CardinalityNodePruning,
)


if __name__ == '__main__':

    DIR = "D10"
    PATH = f"data/ccer/{DIR}"
    FILE = 'imdbclean'
    FILE2 = 'dbpediaclean'
    SEP = '|'
    attributes = [['title', 'starring']]


    RESULTS = f"results/metablocking/{DIR}"
    loaded_dataframe = load_dataframe(PATH, (FILE, FILE2), SEP)
    BLOOM_NAME = 'metablocking'
    print(f"Running METABLOCKING for {DIR}")


    size = [1024]
    qgrams = [4]
    TOTAL_LENGTH = 32

    adjacent_config = {}

    if not ray.is_initialized():
        logging.getLogger("ray").setLevel(logging.ERROR)
        ray.init(log_to_driver=False, logging_level=logging.ERROR)

    futures = []
    MAX_CONCURRENCY = 24


    for configuration, new_bloom in tqdm(
        config_bloom_filter(attributes, size, qgrams,
                BLOOM_NAME, loaded_dataframe, adjacent_config),
        total=TOTAL_LENGTH):

        for CC in [ WeightedEdgePruning,
            WeightedNodePruning,
            CardinalityEdgePruning,
            CardinalityNodePruning]:

            future = comparison_cleaning_exp.remote(CC, new_bloom,
                                                configuration)
            futures.append(future)

            if len(futures) >= MAX_CONCURRENCY:
                finished, futures = ray.wait(futures, num_returns = 1)
                for done in finished:
                    metablocking_method, metablocking_results = ray.get(done)
                    filename = f"{metablocking_method}_12.csv"
                    loaded = os.path.exists(f"{RESULTS}/{filename}")
                    df = pd.DataFrame(metablocking_results)
                    df.to_csv(f'{RESULTS}/{filename}',
                    mode='a+', index=False, header=not loaded)

    while futures:
        finished, futures = ray.wait(futures, num_returns = 1)
        for done in finished:
            metablocking_method, metablocking_results = ray.get(done)
            filename = f"{metablocking_method}_12.csv"
            loaded = os.path.exists(f"{RESULTS}/{filename}")
            df = pd.DataFrame(metablocking_results)
            df.to_csv(f'{RESULTS}/{filename}',
            mode='a+', index=False, header=not loaded)

    print(f"FINISHED {DIR}")

        # for metablocking_method, metablocking_results in ray.get(futures):
        #     LOADED = os.path.exists(f'{RESULTS}/{metablocking_method}')
        #     df = pd.DataFrame(metablocking_results)
        #     df.to_csv(f'{RESULTS}/{metablocking_method}',
        #             mode='a+', index=False, header=not LOADED)
