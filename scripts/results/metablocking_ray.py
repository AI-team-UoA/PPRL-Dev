"""FAISS Results """


import ast
import os

import pandas as pd
from tqdm import tqdm
import ray

from pprl.comparison_cleaning import (
    WeightedEdgePruning,
    WeightedNodePruning,
    CardinalityEdgePruning,
    CardinalityNodePruning,
    BLAST,
)
from utils import load_dataframe, config_bloom_filter


if __name__ == '__main__':
    DIR = "D2"
    PATH = f"data/ccer/{DIR}"
    FILE = 'abtclean'
    FILE2 = 'buyclean'
    SEP = '|'



    DIR = f"results/metablocking/{DIR}"


    loaded_dataframe = load_dataframe(PATH, (FILE, FILE2), SEP)



    attributes = [['name']]
    size = [64, 128, 256, 512, 1024]
    qgrams = [2,3,4]

    TOTAL_LENGTH = len(attributes) * len(size) * len(qgrams)
    for configuration, new_bloom in tqdm(
        config_bloom_filter(attributes, size, qgrams, 'metablocking', load_dataframe),
        total=TOTAL_LENGTH):
        # for CC in []
        wep = WeightedEdgePruning()
        for i in [1,2,4,8,10,12]:

            new_candidates = wep.process(encoded_data=new_bloom,
                            adjacent_bits=i)
            ev : dict = wep.evaluate(new_candidates, verbose = False)

            cnt = 0
            for val in new_candidates.values():
                cnt += len(val)

            my_ev = {"name" : CC.__name__ ,
                    'candidate_pairs': cnt,
                    'adjacent_bits': i+2}

            for keys, items in ev.items():
                my_ev[keys] = round(items, 2) if isinstance(items, float) else items

                my_ev['config'] = configuration
                df = pd.DataFrame([my_ev])
                df.to_csv(f'{DIR}/{FILENAME}', mode='a+', index=False, header=not loaded)
                loaded = os.path.exists(f"{DIR}/{FILENAME}")
