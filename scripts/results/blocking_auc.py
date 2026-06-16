import os
import sys
import math

from typing import Dict
from tqdm import tqdm
import pandas as pd

from utils import load_dataframe, config_bloom_filter
from privjedai.encoded_data import  BloomEncodedData
from privjedai.blocking import  BitBlocker
from privjedai.block_filtering import BlockFiltering, BlockPurging
from privjedai.comparison_cleaning import (
    AbstractComparisonCleaning,
    WeightedEdgePruning,
    WeightedNodePruning,
    CardinalityEdgePruning,
    CardinalityNodePruning,
    BLAST
)


def psi_lambda_calculator(probabilities_: list,
                          size_of_bloom: int,
                          d_t: float = 0.1):

    for threshold in range(10, 55, 5):
        thr = threshold / 100
        distance_threshold = math.ceil(thr * size_of_bloom)
        p = 1 - distance_threshold / size_of_bloom
        for psi_ in range(10, 36, 2):
            prob_match = p ** psi_

            if prob_match <= 0.0000001:
                continue

            lambda__ = math.ceil(math.log(d_t) / math.log(1 - p ** psi_))
            if lambda__ > 500:
                break
            # print(f"ps: {psi_} , lambda : {lambda__}, distance_threshold: {distance_threshold}")
            yield psi_, lambda__, distance_threshold


def one_comparison_cleaning_loop(encoded_data: BloomEncodedData,
                                 blocking_config_: dict,
                                 filtering : str,
                                 weighting_scheme: str,
                                 encoding_config: dict,
                                 meta_blocking_results_: str,
                                 cc: WeightedEdgePruning):
    cleaned_blocks = cc.process(encoded_data)
    cc_ev = cc.evaluate(cleaned_blocks, verbose=False)

    cc_evaluation: Dict[str, float | str | int | dict] = {
        "name": f"{cc.__class__.__name__}",
        "candidate_pairs": int(ev['False Positives']) + int(ev['True Positives']),
        "weighting_scheme": weighting_scheme,
        "blocking_config" : blocking_config_,
        "filtering_technique" : filtering,
    }

    for keys, items in cc_ev.items():
        cc_evaluation[keys] = round(items, 2) if isinstance(items, float) else items


    cc_evaluation['config'] = encoding_config


    filename_ = f"{cc.__class__.__name__}.csv"
    df_ = pd.DataFrame([cc_evaluation])
    loaded_ = os.path.exists(f"{meta_blocking_results_}/{filename_}")
    df_.to_csv(f'{meta_blocking_results_}/{filename_}', mode='a+', index=False, header=not loaded_)


def comparison_cleaning_loop(encoded_data: BloomEncodedData,
                             blocking_config_: dict,
                             filtering: str,
                             encoding_config: dict,
                             meta_blocking_results_: str):

    for method in [WeightedNodePruning,
                   WeightedEdgePruning,
                   CardinalityNodePruning,
                   CardinalityEdgePruning,
                   BLAST]:

        cc = method(weighting_scheme = "CN-CBS")
        one_comparison_cleaning_loop(encoded_data,
                         blocking_config_,
                         filtering,
                         "CN-CBS",
                         encoding_config,
                         meta_blocking_results_,
                         cc)


if __name__ == '__main__':
    DIR = "D8"
    PATH = f"data/ccer/{DIR}"
    FILE = 'walmartclean'
    FILE2 = 'amazonclean'
    attributes = [['title']]
    SEP = "|"

    # log_filename = f"{DIR}.log"
    #
    # log_file = open(log_filename, 'w+')
    #
    # sys.stdout.flush()
    # sys.stderr.flush()
    #
    # sys.stdout = log_file
    # sys.stderr = log_file
    #
    # print(f"Script started. Logging everything to {log_filename}")

    RESULTS = f"results/lsh_blocker/{DIR}"
    RESULTS_FILTERING = f"results/filtering/{DIR}"
    RESULTS_META_BLOCKING = f"results/metablocking/{DIR}"

    loaded_dataframe = load_dataframe(PATH, (FILE, FILE2), SEP)
    BLOOM_NAME = 'lsh'
    print(f"Running LSH for {DIR}")

    size = [64, 128, 256, 512, 1024]
    qgrams = [2, 3, 4]

    TOTAL_LENGTH = 266
    probabilities = [0.75, 0.85, 0.95]

    evaluations = []
    for configuration, new_bloom in tqdm(
            config_bloom_filter(
                attributes,
                size,
                qgrams,
                BLOOM_NAME,
                loaded_dataframe,
                {}),
            desc="Blooms Generator",
            position=0,
            leave=True,
            total=TOTAL_LENGTH):
        bloom_size = new_bloom.metadata.length
        attr_len = len(attributes) if attributes else len(new_bloom.metadata.attributes)

        for threshold in range(10, 55, 5):
            thr = threshold / 100
            distance_thr = int(thr * bloom_size * attr_len)
            bit_blocker_config = BitBlocker.auto_psi_lambda(new_bloom, attributes, thr)
            # max_lambda_=1_000)

            psi = bit_blocker_config['psi']
            lambda_ = bit_blocker_config['lambda_']
            blocker = BitBlocker(psi=psi, lambda_=lambda_, seed=42)
            blocks = blocker.build_blocks(new_bloom)
            ev = blocker.evaluate_blocks(verbose=False)

            blocking_config = {
                "psi" : psi,
                "lambda" : lambda_,
                "distance_threshold": distance_thr
            }

            blocking_evaluation : Dict[str, float | str | int] = {
                "name": "LSH-Blocking",
                "candidate_pairs": int(ev['False Positives']) + int(ev['True Positives']),
                "psi": psi,
                "lambda": lambda_,
                "distance_threshold": distance_thr,
            }
            for keys, items in ev.items():
                blocking_evaluation[keys] = round(items, 2) if isinstance(items, float) else items

            blocking_evaluation['config'] = configuration
            filename = f"lsh_blocker.csv"
            df = pd.DataFrame([blocking_evaluation])
            loaded = os.path.exists(f"{RESULTS}/{filename}")
            df.to_csv(f'{RESULTS}/{filename}', mode='a+', index=False, header=not loaded)

            bf = BlockFiltering()
            filtered_blocks = bf.process(new_bloom)
            ev_filtered = bf.evaluate_blocks(verbose=False)

            filename = "block_filtering.csv"

            filtering_evaluation: Dict[str, float | str | int | dict] = {
                "name": "Block Filtering",
                "candidate_pairs": int(ev_filtered['False Positives']) + int(ev_filtered['True Positives']),
                "ratio" : 0.8,
                "blocking_config": blocking_config,
            }
            for keys, items in ev_filtered.items():
                filtering_evaluation[keys] = round(items, 2) if isinstance(items, float) else items
            filtering_evaluation['config'] = configuration

            df = pd.DataFrame([filtering_evaluation])
            loaded = os.path.exists(f"{RESULTS_FILTERING}/{filename}")
            df.to_csv(f'{RESULTS_FILTERING}/{filename}', mode='a+', index=False, header=not loaded)

            comparison_cleaning_loop(
                new_bloom,
                blocking_config,
                "Block Filtering",
                configuration,
                RESULTS_META_BLOCKING
            )

            new_bloom.set_blocks_with_keys(blocker.blocks_with_keys)

            bp = BlockPurging(smoothing_factor=0.9)
            purged_blocks = bp.process(new_bloom)
            ev_purged = bp.evaluate_blocks(verbose=False)

            filename = "block_purging.csv"

            purging_evaluation: Dict[str, float | str | int | dict] = {
                "name": "Block Purging",
                "candidate_pairs": int(ev_purged['False Positives']) + int(ev_purged['True Positives']),
                "ratio": 0.9,
                "blocking_config": blocking_config,
            }
            for keys, items in ev_purged.items():
                purging_evaluation[keys] = round(items, 2) if isinstance(items, float) else items
            purging_evaluation['config'] = configuration

            df = pd.DataFrame([purging_evaluation])
            loaded = os.path.exists(f"{RESULTS_FILTERING}/{filename}")
            df.to_csv(f'{RESULTS_FILTERING}/{filename}', mode='a+', index=False, header=not loaded)

            comparison_cleaning_loop(
                new_bloom,
                blocking_config,
                "Block Purging",
                configuration,
                RESULTS_META_BLOCKING
            )


