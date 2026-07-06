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
from privjedai.encoded_data import BloomEncodedData
from privjedai.comparison_cleaning import ( BLAST,
                                    WeightedEdgePruning,
                                    WeightedNodePruning,
                                    CardinalityEdgePruning,
                                    CardinalityNodePruning,
                                )


if __name__ == '__main__':


    for dir_ in ['D10']:
        DIR = dir_
        PATH = f"data/ccer/{DIR}"

        FILE, FILE2, attributes, SEP = dataframe_dict[DIR]
        RESULTS_FILTERING = f"results/random_configurations/filtering/{DIR}"
        RESULTS_METABLOCKING = f"results/random_configurations/metablocking/{DIR}"

        loaded_dataframe = load_dataframe(PATH, (FILE, FILE2), SEP)
        BLOOM_NAME = 'purging'

        print(f"Running purging for {DIR}")

        size = [64, 128, 256, 512, 1024]
        if dir_ == 'D10':
            size = [s // 2 for s in size]

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

            new_bloom : BloomEncodedData = config_bloom_filter_with_configuration(configuration=configuration,
                                                            bloom_name=BLOOM_NAME,
                                                            loaded_dataframe=loaded_dataframe)


            threshold  = random.randrange(10, 55, 10)
            thr = threshold / 100
            distance_thr = int(thr * s * len(attributes))
            bit_blocker_config = BitBlocker.auto_psi_lambda(new_bloom, attributes, thr)
            psi = bit_blocker_config['psi']
            lambda_ = bit_blocker_config['lambda_']

            lsh_blocker = BitBlocker(psi=psi, lambda_=lambda_, seed=42)
            _ = lsh_blocker.build_blocks(new_bloom)


            ratio = random.randrange(20, 90, 10)

            ratio = ratio / 100
            ratios.append(ratio)

            start_time  = time.time()
            filtering = BlockFiltering(ratio=ratio)
            _ = filtering.process(new_bloom)
            end_time = time.time()
            filtering_ev = filtering.evaluate_blocks( verbose=False)

            blocking_config = {
                "psi": psi,
                "lambda": lambda_,
                "distance_threshold": distance_thr
            }
            filtering_config = {
                "method": "Block Filtering",
                "ratio": ratio
            }

            filtering_evaluation: Dict[str, float | str | int | dict] = {
                "name": "Block Filtering",
                "ratio": ratio,
                "runtime": round(end_time - start_time, 2),
                "candidate_pairs": int(filtering_ev['False Positives']) + int(filtering_ev['True Positives']),
                "lsh_config": blocking_config,
            }

            for keys, items in filtering_ev.items():
                filtering_evaluation[keys] = round(items, 2) if isinstance(items, float) else items

            filtering_evaluation['config'] = configuration

            df = pd.DataFrame([filtering_evaluation])
            loaded = os.path.exists(f"{RESULTS_FILTERING}/block_filtering.csv")
            df.to_csv(f'{RESULTS_FILTERING}/block_filtering.csv',
                    mode='a+',
                    index=False,
                    header=not loaded)

            prev_keys = new_bloom.blocks_with_keys.copy()
            for cc_class in [ WeightedEdgePruning, WeightedNodePruning, CardinalityEdgePruning, CardinalityNodePruning, BLAST]:
                new_bloom.blocks_with_keys = prev_keys.copy()
                start_time = time.time()
                cc = cc_class(weighting_scheme='CN-CBS')
                blocks = cc.process(new_bloom)
                ev = cc.evaluate(blocks, verbose=False)
                end_time = time.time()


                comparison_evaluation: Dict[str, float | str | int | dict] = {
                    "name": f"{cc_class.__name__}",
                    "weighting_scheme": 'CN-CBS',
                    "runtime": round(end_time - start_time, 2),
                    "candidate_pairs": int(ev['False Positives']) + int(ev['True Positives']),
                    "lsh_config": blocking_config,
                    "filtering_config": filtering_config,
                    }

                for keys, items in ev.items():
                    comparison_evaluation[keys] = round(items, 2) if isinstance(items, float) else items

                comparison_evaluation['config'] = configuration

                df = pd.DataFrame([comparison_evaluation])
                loaded = os.path.exists(f"{RESULTS_METABLOCKING}/{cc_class.__name__}.csv")
                df.to_csv(f'{RESULTS_METABLOCKING}/{cc_class.__name__}.csv',
                        mode='a+',
                        index=False,
                        header=not loaded)
