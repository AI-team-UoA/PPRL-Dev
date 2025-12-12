"""FAISS Results """


import os
import pandas as pd
from tqdm import tqdm
import ray
import logging
from utils import load_dataframe, config_bloom_filter, comparison_cleaning_exp

from pprl.comparison_cleaning import (
    WeightedEdgePruning,
    WeightedNodePruning,
    CardinalityEdgePruning,
    CardinalityNodePruning,
    BLAST,
)


if __name__ == '__main__':

    DIR = "D2"
    PATH = f"data/ccer/{DIR}"
    FILE = 'abtclean'
    FILE2 = 'buyclean'
    SEP = '|'
    RESULTS = f"results/metablocking/{DIR}"
    attributes = [['name']]
    size = [64, 128, 256, 512, 1024]
    qgrams = [2,3,4]
    TOTAL_LENGTH = len(attributes) * len(size) * len(qgrams)



    loaded_dataframe = load_dataframe(PATH, (FILE, FILE2), SEP)

    if not ray.is_initialized():
        logging.getLogger("ray").setLevel(logging.ERROR)
        ray.init(log_to_driver=False, logging_level=logging.ERROR)


    for configuration, new_bloom in tqdm(
        config_bloom_filter(attributes, size, qgrams, 'metablocking', loaded_dataframe),
        total=TOTAL_LENGTH):
        new_bloom_ref = ray.put(new_bloom)
        configuration_ref = ray.put(configuration)

        futures = []
        for CC in [ WeightedEdgePruning,
            WeightedNodePruning,
            CardinalityEdgePruning,
            CardinalityNodePruning,
            BLAST]:

            future = comparison_cleaning_exp.remote(CC, new_bloom_ref,
                                                configuration_ref)
            futures.append(future)


        for metablocking_method, metablocking_results in ray.get(futures):
            LOADED = os.path.exists(f'{RESULTS}/{metablocking_method}')
            df = pd.DataFrame(metablocking_results)
            df.to_csv(f'{RESULTS}/{metablocking_method}',
                    mode='a+', index=False, header=not LOADED)
