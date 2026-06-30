"""Random Blocking Experimental Analysis Script"""
import math
import os
import random
import time
from typing import Dict
from tqdm import tqdm
import pandas as pd

from utils import load_dataframe, config_bloom_filter_with_configuration
from privjedai.blocking import  BitBlocker


if __name__ == '__main__':

    DIR = "D10"
    PATH = f"data/ccer/{DIR}"
    FILE = 'imdbclean'
    FILE2 = 'dbpediaclean'
    attributes = ['title', 'starring']
    SEP = "|"

    RESULTS = f"results/random_configurations/lsh_blocker/{DIR}"

    loaded_dataframe = load_dataframe(PATH, (FILE, FILE2), SEP)
    BLOOM_NAME = 'lsh'

    print(f"Running LSH for {DIR}")

    size = [64 // 2, 128 // 2, 256 // 2, 512 // 2, 1024 // 2]

    q_grams = [2, 3, 4]

    TOTAL_LENGTH = 100

    random.seed(DIR)

    for _ in tqdm(range(TOTAL_LENGTH),
              desc="Running lsh_blocking with random configurations",
              total=TOTAL_LENGTH):

        s = random.choice(size)
        q = random.choice(q_grams)

        end_range = s // (8 * q)

        if end_range == 1:
            n_hashes = 1
        else:
            n_hashes = random.randrange(1,end_range, 1)

        configuration = {
            'attributes': attributes,
            'size': s,
            'offset': 0,
            'num_hashes': n_hashes,
            'hashing_type': 'salted_qgrams',
            'salt': '',
            'qgrams': q
        }

        new_bloom = config_bloom_filter_with_configuration(configuration=configuration,
                                                        bloom_name=BLOOM_NAME,
                                                        loaded_dataframe=loaded_dataframe)

        start_time = time.time()

        threshold  = random.randrange(10, 55, 10)
        thr = threshold / 100
        distance_thr = int(thr * s * len(attributes))
        bit_blocker_config = BitBlocker.auto_psi_lambda(new_bloom, attributes, thr)
        psi = bit_blocker_config['psi']
        lambda_ = bit_blocker_config['lambda_']

        lsh_blocker = BitBlocker(psi=psi, lambda_=lambda_, seed=42)
        _ = lsh_blocker.build_blocks(new_bloom)

        end_time = time.time()
        ev = lsh_blocker.evaluate_blocks(verbose=True)

        blocking_config = {
            "psi": psi,
            "lambda": lambda_,
            "distance_threshold": distance_thr
        }
        blocking_evaluation_: Dict[str, float | str | int | dict] = {
            "name": "LSH-Blocking",
            "runtime": round(end_time - start_time, 2),
            "candidate_pairs": int(ev['False Positives']) + int(ev['True Positives']),
            "psi": psi,
            "lambda": lambda_,
            "distance_threshold": distance_thr,
        }
        for keys, items in ev.items():
            blocking_evaluation_[keys] = round(items, 2) if isinstance(items, float) else items

        blocking_evaluation_['config'] = configuration

        df = pd.DataFrame([blocking_evaluation_])
        loaded = os.path.exists(f"{RESULTS}/lsh_blocker.csv")
        df.to_csv(f'{RESULTS}/lsh_blocker.csv',
                  mode='a+',
                  index=False,
                  header=not loaded)
