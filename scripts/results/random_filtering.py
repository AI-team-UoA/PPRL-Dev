"""Random Blocking Experimental Analysis Script"""
import os
import random
import time
from typing import Dict
from tqdm import tqdm
import pandas as pd

from utils import load_dataframe, config_bloom_filter_with_configuration, dataframe_dict
from privjedai.blocking import  BitBlocker
from privjedai.block_filtering import BlockFiltering


if __name__ == '__main__':


    for dir_ in ['D2', 'D3', 'D4', 'D8', 'D9', 'D10']:
        DIR = dir_
        PATH = f"data/ccer/{DIR}"

        FILE, FILE2, attributes, SEP = dataframe_dict[DIR]
        RESULTS = f"results/random_configurations/filtering/{DIR}"

        loaded_dataframe = load_dataframe(PATH, (FILE, FILE2), SEP)
        BLOOM_NAME = 'filtering'

        print(f"Running purging for {DIR}")

        size = [64, 128, 256, 512, 1024]

        q_grams = [2, 3, 4]

        TOTAL_LENGTH = 100

        random.seed(DIR)


        ratios = []
        for _ in tqdm(range(TOTAL_LENGTH),
                  desc="Running purging with random configurations",
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

            # new_bloom = config_bloom_filter_with_configuration(configuration=configuration,
            #                                                 bloom_name=BLOOM_NAME,
            #                                                 loaded_dataframe=loaded_dataframe)


            threshold  = random.randrange(10, 55, 10)
            # thr = threshold / 100
            # distance_thr = int(thr * s * len(attributes))
            # bit_blocker_config = BitBlocker.auto_psi_lambda(new_bloom, attributes, thr)
            # psi = bit_blocker_config['psi']
            # lambda_ = bit_blocker_config['lambda_']
            #
            # lsh_blocker = BitBlocker(psi=psi, lambda_=lambda_, seed=42)
            # _ = lsh_blocker.build_blocks(new_bloom)


            ratio = random.randrange(20, 90, 10)

            ratio = ratio / 100
            ratios.append(ratio)
            continue

        df = pd.read_csv(f"{RESULTS}/block_filtering.csv")
        df.insert(1, 'ratio', ratios)
        df.to_csv(f'{RESULTS}/block_filtering.csv',
                  index=False,
                  mode='w+')

            # start_time = time.time()
            #
            #
            # filtering = BlockPurging(smoothing_factor=0.9)
            # _ = filtering.process(new_bloom)
            #
            # ev = filtering.evaluate_blocks(verbose=False)
            # end_time = time.time()
            #
            # blocking_config = {
            #     "psi": psi,
            #     "lambda": lambda_,
            #     "distance_threshold": distance_thr
            # }
            # filtering_evaluation: Dict[str, float | str | int | dict] = {
            #     "name": "Block Purging",
            #     "smoothing_factor": 0.9,
            #     "runtime": round(end_time - start_time, 2),
            #     "candidate_pairs": int(ev['False Positives']) + int(ev['True Positives']),
            #     "lsh_config": blocking_config,
            # }
            #
            # for keys, items in ev.items():
            #     filtering_evaluation[keys] = round(items, 2) if isinstance(items, float) else items
            #
            # filtering_evaluation['config'] = configuration
            #
            # df = pd.DataFrame([filtering_evaluation])
            # loaded = os.path.exists(f"{RESULTS}/block_purging.csv")
            # df.to_csv(f'{RESULTS}/block_purging.csv',
            #           mode='a+',
            #           index=False,
            #           header=not loaded)
