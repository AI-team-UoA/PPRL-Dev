"""Random Blocking Experimental Analysis Script"""
import os
import random
from collections import defaultdict
import ast
import time
from typing import Dict

from tqdm import tqdm
import pandas as pd

from utils import load_dataframe, config_bloom_filter_with_configuration, dataframe_dict
from privjedai.blocking import  BitBlocker
from privjedai.block_filtering import BlockPurging, BlockFiltering
from privjedai.encoded_data import BloomEncodedData


if __name__ == '__main__':

    dir_list = ['D2', 'D3', 'D4', 'D8', 'D9', 'D10']

    for dir_ in dir_list:
        DIR = dir_
        PATH = f"data/ccer/{DIR}"

        FILE, FILE2, attributes, SEP = dataframe_dict[DIR]
        # RESULTS_BLOCKING = f"results/random_configurations/unique_blocking/lsh_blocker/{DIR}"
        RESULTS_FILTERING = f"results/random_configurations/unique_blocking/filtering/{DIR}"
        # RESULTS_METABLOCKING = f"results/random_configurations/unique_blocking/metablocking/{DIR}"

        loaded_dataframe = load_dataframe(PATH, (FILE, FILE2), SEP)
        BLOOM_NAME = 'filtering'

        print(f"Running no filtering for {DIR}")

        size = [64, 128, 256, 512, 1024]
        if dir_ == 'D10':
            size = [s // 2 for s in size]

        q_grams = [2, 3, 4]

        TOTAL_LENGTH = 100

        random.seed(DIR)


        ratios = []
        for _ in tqdm(range(TOTAL_LENGTH),
                  desc="Running no filtering with random configurations",
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
            start_time = time.time()
            bit_blocker_config = BitBlocker.auto_psi_lambda(new_bloom, attributes, thr)
            psi = bit_blocker_config['psi']
            lambda_ = bit_blocker_config['lambda_']

            lsh_blocker = BitBlocker(psi=psi, lambda_=lambda_, seed=42)
            lsh_blocks = lsh_blocker.build_blocks(new_bloom)
            end_time = time.time()
            # blocking_ev = lsh_blocker.evaluate(lsh_blocks, verbose=False)


            blocking_config = {
                "psi": psi,
                "lambda": lambda_,
                "distance_threshold": distance_thr
            }

            # blocking_evaluation : Dict[str, float | str | int | dict] = {
                # "name": "LSH-Blocking",
                # "runtime": round(end_time - start_time, 2),
                # "candidate_pairs": int(blocking_ev['False Positives']) + int(blocking_ev['True Positives']),
                # "psi": psi,
                # "lambda": lambda_,
                # "distance_threshold": distance_thr
            # }

            # for keys, items in blocking_ev.items():
                # blocking_evaluation[keys] = round(items, 2) if isinstance(items, float) else items

            # blocking_evaluation['config'] = configuration
            # df = pd.DataFrame([blocking_evaluation])
            # loaded = os.path.exists(f"{RESULTS_BLOCKING}/lsh_blocker.csv")
            # df.to_csv(f'{RESULTS_BLOCKING}/lsh_blocker.csv',
                #   mode='a+',
                #   index=False,
                #   header=not loaded)


            blocking_runtime = round(end_time - start_time, 2)
            # ratio = random.randrange(20, 90, 10)

            # ratio = ratio / 100
            # ratios.append(ratio)

            start_time  = time.time()
            smoothing_factor = 0.9
            purging = BlockPurging(smoothing_factor=smoothing_factor)

            blocks_with_keys = purging.process(new_bloom)


            # filtering = BlockPurging(ratio=ratio)
            # blocks_with_keys = filtering.process(new_bloom)


            end_time = time.time()

            filter_blocks = defaultdict(set)
            d1 = defaultdict(list)
            d2 = defaultdict(list)
            for key, id_ in blocks_with_keys:
                if id_ < new_bloom.bounds[0]:
                    d1[key].append(id_)
                else:
                    d2[key].append(id_)


            for key, d1_entities in d1.items():
                if key in d2:
                    d2_entities = d2[key]
                    for id1 in d1_entities:
                        filter_blocks[id1].update(d2_entities)


            filtering_ev = purging.evaluate(filter_blocks, verbose=False)

            filtering_config = {
                "method": "Block Purging",
                "smoothing_factor": smoothing_factor
            }

            filtering_evaluation: Dict[str, float | str | int | dict] = {
                "name": "Block Purging",
                "smoothing_factor": smoothing_factor,
                "runtime": round(end_time - start_time, 2),
                'total_runtime': round(blocking_runtime + (end_time - start_time), 2),
                "candidate_pairs": int(filtering_ev['False Positives']) + int(filtering_ev['True Positives']),
                "lsh_config": blocking_config,
            }

            for keys, items in filtering_ev.items():
                filtering_evaluation[keys] = round(items, 2) if isinstance(items, float) else items

            filtering_evaluation['config'] = configuration

            df = pd.DataFrame([filtering_evaluation])
            loaded = os.path.exists(f"{RESULTS_FILTERING}/block_purging.csv")
            df.to_csv(f'{RESULTS_FILTERING}/block_purging.csv',
                    mode='a+',
                    index=False,
                    header=not loaded)


            # weighting_schemes = ['CN-CBS', 'CBS', 'SN-CBS',
            #     'SNC', 'CNC', 'SND', 'CND', 'CNJ', 'SNJ',
            #     'COSINE', 'DICE', 'ECBS', 'JS', 'EJS', 'X2']

            # for cc_class in [WeightedEdgePruning, WeightedNodePruning, CardinalityEdgePruning, CardinalityNodePruning, BLAST]:
            #     for weighting_scheme in tqdm(weighting_schemes, desc=f"Processing {cc_class.__name__}"):
            #         start_time = time.time()
            #         cc = cc_class(weighting_scheme=weighting_scheme)
            #         blocks = cc.process(new_bloom)
            #         ev = cc.evaluate(blocks, verbose=False)
            #         end_time = time.time()



            #         comparison_evaluation: Dict[str, float | str | int | dict] = {
            #             "name": f"{cc_class.__name__}",
            #             "weighting_scheme": weighting_scheme,
            #             "runtime": round(end_time - start_time, 2),
            #             "candidate_pairs": int(ev['False Positives']) + int(ev['True Positives']),
            #             "lsh_config": blocking_config,
            #             "filtering_config": filtering_config,
            #             }

            #         for keys, items in ev.items():
            #             comparison_evaluation[keys] = round(items, 2) if isinstance(items, float) else items

            #         comparison_evaluation['config'] = configuration

            #         df = pd.DataFrame([comparison_evaluation])
            #         loaded = os.path.exists(f"{RESULTS_METABLOCKING}/{cc_class.__name__}.csv")
            #         df.to_csv(f'{RESULTS_METABLOCKING}/{cc_class.__name__}.csv',
            #                 mode='a+',
            #                 index=False,
            #                 header=not loaded)


