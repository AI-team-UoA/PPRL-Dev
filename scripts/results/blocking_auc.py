""""Blocking Experimental Analysis Script"""

import os
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict
from tqdm import tqdm
import pandas as pd

from utils import load_dataframe, config_bloom_filter
from privjedai.encoded_data import  BloomEncodedData
from privjedai.blocking import  BitBlocker
from privjedai.block_filtering import BlockFiltering, BlockPurging
from privjedai.comparison_cleaning import (
    WeightedEdgePruning,
    WeightedNodePruning,
    CardinalityEdgePruning,
    CardinalityNodePruning,
    BLAST
)


# def psi_lambda_calculator(size_of_bloom: int,
#                           d_t: float = 0.1):

#     for threshold in range(10, 55, 5):
#         thr = threshold / 100
#         distance_threshold = math.ceil(thr * size_of_bloom)
#         p = 1 - distance_threshold / size_of_bloom
#         for psi_ in range(10, 36, 8):
#             prob_match = p ** psi_

#             if prob_match <= 0.0000001:
#                 continue

#             lambda__ = math.ceil(math.log(d_t) / math.log(1 - p ** psi_))
#             if lambda__ > 500:
#                 break
#             # print(f"ps: {psi_} , lambda : {lambda__}, distance_threshold: {distance_threshold}")
#             yield psi_, lambda__, distance_threshold


def one_comparison_cleaning_loop(encoded_data: BloomEncodedData,
                                 blocking_config_: dict,
                                 filtering : str,
                                 weighting_scheme: str,
                                 encoding_config: dict,
                                 cc: WeightedEdgePruning):

    """Loop for one comparison cleaning method, returns the evaluation results as a dictionary
    and the filename to save the results"""

    cleaned_blocks = cc.process(encoded_data)
    cc_ev = cc.evaluate(cleaned_blocks, verbose=False)



    cc_evaluation_: Dict[str, float | str | int | dict] = {
        "name": f"{cc.__class__.__name__}",
        "candidate_pairs": int(cc_ev['False Positives']) + int(cc_ev['True Positives']),
        "weighting_scheme": weighting_scheme,
        "blocking_config" : blocking_config_,
        "filtering_technique" : filtering,
    }

    for keys, items in cc_ev.items():
        cc_evaluation_[keys] = round(items, 2) if isinstance(items, float) else items


    cc_evaluation_['config'] = encoding_config
    filename_ = f"{cc.__class__.__name__}.csv"

    return filename_, cc_evaluation_



def comparison_cleaning_loop(encoded_data: BloomEncodedData,
                             blocking_config_: dict,
                             filtering: str,
                             encoding_config: dict):

    """Comparison cleaning loop for all methods, returns a dictionary with the results for each method"""
    comparisons_results = {}
    for method in [WeightedNodePruning,
                   WeightedEdgePruning,
                   CardinalityNodePruning,
                   CardinalityEdgePruning,
                   BLAST]:

        cc = method(weighting_scheme = "CN-CBS")
        filename_, cc_evaluation_ = one_comparison_cleaning_loop(encoded_data,
                         blocking_config_,
                         filtering,
                         "CN-CBS",
                         encoding_config,
                         cc)
        comparisons_results[filename_] = cc_evaluation_
    return comparisons_results


def worker_blocking(
    threshold_: int,
    original_bloom: BloomEncodedData,
    bloom_size_: int,
    attr_len_: int,
    attributes_: list,
    configuration_: dict,
):
    """Worker function for blocking, returns the evaluation results for blocking, filtering and comparison cleaning"""

    new_bloom_ = copy.deepcopy(original_bloom)
    thr = threshold_ / 100
    distance_thr = int(thr * bloom_size_ * attr_len_)
    bit_blocker_config = BitBlocker.auto_psi_lambda(new_bloom_, attributes_, thr)
    psi = bit_blocker_config['psi']
    lambda_ = bit_blocker_config['lambda_']
    blocker = BitBlocker(psi=psi, lambda_=lambda_, seed=42)
    _ = blocker.build_blocks(new_bloom_)
    ev = blocker.evaluate_blocks(verbose=True)
    blocking_config = {
        "psi": psi,
        "lambda": lambda_,
        "distance_threshold": distance_thr
    }

    blocking_evaluation_: Dict[str, float | str | int] = {
        "name": "LSH-Blocking",
        "candidate_pairs": int(ev['False Positives']) + int(ev['True Positives']),
        "psi": psi,
        "lambda": lambda_,
        "distance_threshold": distance_thr,
    }
    for keys, items in ev.items():
        blocking_evaluation_[keys] = round(items, 2) if isinstance(items, float) else items

    blocking_evaluation_['config'] = configuration_


    bf = BlockFiltering()
    _ = bf.process(new_bloom_)
    ev_filtered = bf.evaluate_blocks(verbose=False)
    filtering_evaluation_: Dict[str, float | str | int | dict] = {
        "name": "Block Filtering",
        "candidate_pairs": int(ev_filtered['False Positives']) + int(ev_filtered['True Positives']),
        "ratio": 0.8,
        "blocking_config": blocking_config,
    }
    for keys, items in ev_filtered.items():
        filtering_evaluation_[keys] = round(items, 2) if isinstance(items, float) else items

    filtering_evaluation_['config'] = configuration_
    filtering_comparisons_results_ = comparison_cleaning_loop(
        new_bloom_,
        blocking_config,
        "Block Filtering",
        configuration_,
    )

    new_bloom_.set_blocks_with_keys(blocker.blocks_with_keys)

    bp = BlockPurging(smoothing_factor=0.9)
    _ = bp.process(new_bloom_)
    ev_purged = bp.evaluate_blocks(verbose=False)
    purging_evaluation_: Dict[str, float | str | int | dict] = {
        "name": "Block Purging",
        "candidate_pairs": int(ev_purged['False Positives']) + int(ev_purged['True Positives']),
        "ratio": 0.9,
        "blocking_config": blocking_config,
    }
    for keys, items in ev_purged.items():
        purging_evaluation_[keys] = round(items, 2) if isinstance(items, float) else items
    purging_evaluation_['config'] = configuration_
    purging_comparisons_results_ = comparison_cleaning_loop(
        new_bloom_,
        blocking_config,
        "Block Purging",
        configuration_,
    )

    return ( blocking_evaluation_, filtering_evaluation_,
        filtering_comparisons_results_, purging_evaluation_, purging_comparisons_results_
    )



if __name__ == '__main__':
    DIR = "D9"
    PATH = f"data/ccer/{DIR}"
    FILE = 'dblpclean'
    FILE2 = 'scholarclean'
    attributes = [['title']]
    SEP = ">"

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

    evaluations = []
    MAX_THREADS = 10


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


        with ProcessPoolExecutor(max_workers=MAX_THREADS) as executor:
            futures = []
            for threshold in range(10, 55, 5):
                future = executor.submit(
                    worker_blocking,
                    threshold_=threshold,
                    original_bloom=new_bloom,
                    bloom_size_=bloom_size,
                    attr_len_=attr_len,
                    attributes_=attributes,
                    configuration_=configuration
                )
                futures.append(future)

            for future in tqdm(as_completed(futures), desc="Blocking and Filtering", position=1, leave=True, total=len(futures)):
                try:
                    blocking_evaluation, filtering_evaluation, filtering_comparisons_results, \
                        purging_evaluation, purging_comparisons_results = future.result()
                    df = pd.DataFrame([blocking_evaluation])
                    loaded = os.path.exists(f"{RESULTS}/lsh_blocker.csv")
                    df.to_csv(f'{RESULTS}/lsh_blocker.csv', mode='a+', index=False,
                               header=not loaded)

                    df_filtering = pd.DataFrame([filtering_evaluation])
                    loaded_filtering = os.path.exists(f"{RESULTS_FILTERING}/block_filtering.csv")
                    df_filtering.to_csv(f'{RESULTS_FILTERING}/block_filtering.csv', mode='a+',
                                        index=False, header=not loaded_filtering)

                    for filename, cc_evaluation in filtering_comparisons_results.items():
                        df_cc = pd.DataFrame([cc_evaluation])
                        loaded_cc = os.path.exists(f"{RESULTS_META_BLOCKING}/{filename}")
                        df_cc.to_csv(f'{RESULTS_META_BLOCKING}/{filename}', mode='a+', index=False,
                                      header=not loaded_cc)

                    df_purging = pd.DataFrame([purging_evaluation])
                    loaded_purging = os.path.exists(f"{RESULTS_FILTERING}/block_purging.csv")
                    df_purging.to_csv(f'{RESULTS_FILTERING}/block_purging.csv', mode='a+',
                                       index=False, header=not loaded_purging)

                    for filename, cc_evaluation in purging_comparisons_results.items():
                        df_cc = pd.DataFrame([cc_evaluation])
                        loaded_cc = os.path.exists(f"{RESULTS_META_BLOCKING}/{filename}")
                        df_cc.to_csv(f'{RESULTS_META_BLOCKING}/{filename}', mode='a+', index=False,
                                      header=not loaded_cc)






                except Exception as e:
                    print(f"An error occurred: {e}")

