"""Random Blocking Experimental Analysis Script"""
import gc
import concurrent.futures
import os
import random
from collections import defaultdict
import ast
import time
from typing import Dict

from tqdm import tqdm
import pandas as pd

from multiprocessing import Process

# from scripts.data_manipulation.metablocking_total_runtime import filtering_runtime
from utils import load_dataframe, config_bloom_filter_with_configuration, dataframe_dict
from privjedai.blocking import  BitBlocker
from privjedai.block_filtering import BlockPurging, BlockFiltering
from privjedai.comparison_cleaning import (
    WeightedEdgePruning,
    WeightedNodePruning,
    BLAST,
    CardinalityEdgePruning,
    CardinalityNodePruning
)
from privjedai.encoded_data import BloomEncodedData


WEIGHTING_SCHEMES = ['CN-CBS', 'CBS', 'SN-CBS', 'SNC', 'CNC', 'SND', 'CND', 'CNJ', 'SNJ', 'COSINE', 'DICE', 'ECBS', 'JS', 'EJS', 'X2']


def evaluate_metablocking(cc_class, new_bloom, lsh_runtime, filtering_runtime, blocking_config, filtering_config, config):


    comparison_evaluations = []
    for ws in tqdm(WEIGHTING_SCHEMES, total=len(WEIGHTING_SCHEMES), desc=f"Processing {cc_class.__name__}", leave=False, position=1):
        start_time = time.time()
        cc = cc_class(weighting_scheme=ws)
        blocks = cc.process(new_bloom)
        end_time = time.time()
        ev = cc.evaluate(blocks, verbose=False)

        comparison_evaluation: Dict[str, float | str | int | dict] = {
            "name": f"{cc_class.__name__}",
            "weighting_scheme": ws,
            "runtime": round(end_time - start_time, 2),
            "total_runtime": round(lsh_runtime + filtering_runtime + (end_time - start_time), 2),
            "candidate_pairs": int(ev['False Positives']) + int(ev['True Positives']),
            "lsh_config": blocking_config,
            "filtering_config": filtering_config,
            }

        for keys, items in ev.items():
            comparison_evaluation[keys] = round(items, 2) if isinstance(items, float) else items

        comparison_evaluation['config'] = config


        comparison_evaluations.append(comparison_evaluation)
        del cc
        del blocks
    gc.collect()

    return cc_class, comparison_evaluations




def block_filtering_with_metablocking(psi, lambda_, config, distance_threshold, ratio, loaded_dataframe, bloom_name, original_filtering_,
                                      results_metablocking_):


    pbar = tqdm(total=7 + 2, position=2, leave=False, desc='Steps Remaining')

    config = ast.literal_eval(config)
    new_bloom : BloomEncodedData = config_bloom_filter_with_configuration(configuration=config,
                                                               bloom_name=bloom_name,
                                                               loaded_dataframe=loaded_dataframe)


    start_time = time.time()
    lsh_blocker = BitBlocker(psi=psi, lambda_=lambda_, seed=42)
    _ = lsh_blocker.build_blocks(new_bloom)
    end_time = time.time()
    pbar.update(1)
    lsh_runtime = end_time - start_time

    blocking_config = {
        "psi": int(psi),
        "lambda": int(lambda_),
        "distance_threshold": int(distance_threshold)
    }


    start_time = time.time()
    block_filter = BlockFiltering(ratio=ratio)
    _ = block_filter.process(new_bloom)
    pbar.update(1)
    end_time = time.time()


    filtering_ev_duplicates = block_filter.evaluate_blocks(verbose=False)
    pbar.update(1)

    filtering_ev = block_filter.evaluate(verbose=False)
    pbar.update(1)

    filtering_runtime = end_time - start_time


    filtering_config = {
        "method" : "Block Filtering",
        "ratio": ratio
    }


    filtering_evaluation: Dict[str, float | str | int | dict] = {
        "name": "Block Filtering",
        "ratio": ratio,
        "runtime": round(end_time - start_time, 2),
        "candidate_pairs": int(filtering_ev_duplicates['False Positives']) + int(filtering_ev_duplicates['True Positives']),
        'total_runtime': round(lsh_runtime + (end_time - start_time), 2),
        "unique_candidate_pairs": int(filtering_ev['False Positives']) + int(filtering_ev['True Positives']),
        "lsh_config": blocking_config,
        "Precision %": filtering_ev_duplicates["Precision %"],
        'Recall %' : filtering_ev_duplicates['Recall %'],
        'F1 %': filtering_ev_duplicates['F1 %'],
        "Unique Precision %": filtering_ev["Precision %"],
        "Unique F1 %": filtering_ev["F1 %"],
        "True Positives": filtering_ev_duplicates['True Positives'],
        "False Positives": filtering_ev_duplicates['False Positives'],
        "True Negatives": filtering_ev_duplicates['True Negatives'],
        "False Negatives": filtering_ev_duplicates['False Negatives'],
        "config" : config
    }

    df = pd.DataFrame([filtering_evaluation])
    loaded = os.path.exists(f"{original_filtering_}/block_filtering.csv")
    df.to_csv(f'{original_filtering_}/block_filtering.csv',
            mode='a+',
            index=False,
            header=not loaded, float_format="%.2f")


    for cc_class in [WeightedEdgePruning, WeightedNodePruning, CardinalityEdgePruning, CardinalityNodePruning, BLAST]:
        _, comparison_evaluations =  evaluate_metablocking(cc_class, new_bloom, lsh_runtime, filtering_runtime, blocking_config,
                                    filtering_config, config)
        df = pd.DataFrame(comparison_evaluations)
        loaded = os.path.exists(f"{results_metablocking_}/{cc_class.__name__}.csv")
        df.to_csv(f'{results_metablocking_}/{cc_class.__name__}.csv',
                mode='a+',
                index=False,
                header=not loaded, float_format="%.2f")
        pbar.update(1)
    pbar.close()

if __name__ == '__main__':

    # dir_list = ['D2', 'D3', 'D4', 'D8', 'D9', 'D10']
    dir_list = ['D10']
    for dir_ in dir_list:
        DIR = dir_
        PATH = f"data/ccer/{DIR}"

        FILE, FILE2, attributes, SEP = dataframe_dict[DIR]
        RESULTS_FILTERING = f"results/random_configurations/unique_blocking/filtering/{DIR}"
        ORIGINAL_FILTERING = f"results/random_configurations/filtering/{DIR}"
        # RESULTS_FILTERING = f"results/random_configurations/unique_blocking/filtering/{DIR}"
        RESULTS_METABLOCKING = f"results/random_configurations/metablocking/{DIR}"



        filtering_df = pd.read_csv(f'{ORIGINAL_FILTERING}/old_block_filtering.csv')
        done_df = pd.read_csv(f'{ORIGINAL_FILTERING}/block_filtering.csv')
        filtering_df['lsh_config'] = filtering_df['lsh_config'].apply(ast.literal_eval)


        psi_list = filtering_df['lsh_config'].apply(lambda  x: x.get('psi')).values
        lambda_list = filtering_df['lsh_config'].apply(lambda  x: x.get('lambda')).values
        config_list = filtering_df['config'].values
        distance_threshold_list = filtering_df['lsh_config'].apply(lambda x: x.get('distance_threshold')).values

        ratios = filtering_df['ratio'].values





        loaded_dataframe = load_dataframe(PATH, (FILE, FILE2), SEP)
        BLOOM_NAME = 'filtering'


        ratio = 0.8
        for i, (psi, lambda_, config, distance_threshold) in tqdm(enumerate(zip(psi_list,
                    lambda_list, config_list, distance_threshold_list)),
                    total=100, leave=False, desc=f"Running unique configurations for {DIR}"):

            if i < done_df.shape[0]:
                time.sleep(0.5)
                continue

            p = Process(target=block_filtering_with_metablocking,
                        kwargs={
                            "psi": psi,
                            "lambda_": lambda_,
                            "config": config,
                            "distance_threshold": distance_threshold,
                            "ratio": ratio,
                            "loaded_dataframe": loaded_dataframe,
                            "bloom_name": BLOOM_NAME,
                            "original_filtering_": ORIGINAL_FILTERING,
                            "results_metablocking_": RESULTS_METABLOCKING
                        })
            p.start()
            p.join()

            # blocking_ev = lsh_blocker.evaluate(lsh_blocks, verbose=False)
            # unique_candidate_pairs.append(int(blocking_ev['False Positives']) + int(blocking_ev['True Positives']))
            # unique_precision.append(round(blocking_ev['Precision %'], 2))
            # unique_f1.append(round(blocking_ev['F1 %'], 2))




            # weighting_schemes = ['CN-CBS', 'CBS', 'SN-CBS',
            #     'SNC', 'CNC', 'SND', 'CND', 'CNJ', 'SNJ',
            #     'COSINE', 'DICE', 'ECBS', 'JS', 'EJS', 'X2']

            # with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            #     futures = []
            #     for cc_class in [WeightedEdgePruning, WeightedNodePruning, CardinalityEdgePruning, CardinalityNodePruning, BLAST]:
            #         future = executor.submit(
            #             evaluate_metablocking,
            #             cc_class,
            #             new_bloom,
            #             lsh_runtime,
            #             filtering_runtime,
            #             blocking_config,
            #             filtering_config,
            #             config
            #         )

            #         futures.append(future)
            #     for future in concurrent.futures.as_completed(futures):
            #         cc_class, comparison_evaluations = future.result()
            #         df = pd.DataFrame(comparison_evaluations)
            #         loaded = os.path.exists(f"{RESULTS_METABLOCKING}/{cc_class.__name__}.csv")
            #         df.to_csv(f'{RESULTS_METABLOCKING}/{cc_class.__name__}.csv',
            #                 mode='a+',
            #                 index=False,
            #                 header=not loaded, float_format="%.2f")
            #         del df
            #         pbar.update(1)
            # gc.collect()
            # pbar.close()
            # continue

                # for weighting_scheme in tqdm(weighting_schemes, desc=f"Processing {cc_class.__name__}", leave=False, position=1):
                #     start_time = time.time()
                #     cc = cc_class(weighting_scheme=weighting_scheme)
                #     blocks = cc.process(new_bloom)
                #     end_time = time.time()
                #     ev = cc.evaluate(blocks, verbose=False)



                #     comparison_evaluation: Dict[str, float | str | int | dict] = {
                #         "name": f"{cc_class.__name__}",
                #         "weighting_scheme": weighting_scheme,
                #         "runtime": round(end_time - start_time, 2),
                #         "total_runtime": round(lsh_runtime + filtering_runtime + (end_time - start_time), 2),
                #         "candidate_pairs": int(ev['False Positives']) + int(ev['True Positives']),
                #         "lsh_config": blocking_config,
                #         "filtering_config": filtering_config,
                #         }

                #     for keys, items in ev.items():
                #         comparison_evaluation[keys] = round(items, 2) if isinstance(items, float) else items

                #     comparison_evaluation['config'] = config

                #     df = pd.DataFrame([comparison_evaluation])
                #     loaded = os.path.exists(f"{RESULTS_METABLOCKING}/{cc_class.__name__}.csv")
                #     df.to_csv(f'{RESULTS_METABLOCKING}/{cc_class.__name__}.csv',
                #             mode='a+',
                #             index=False,
                #             header=not loaded, float_format="%.2f")
                #     del cc
                #     del blocks
                #     del ev
                #     gc.collect()
            #     pbar.update(1)
            # pbar.close()



            # blocking_evaluation : Dict[str, float | str | int | dict] = {
            #     "name": "LSH-Blocking",
            #     "runtime": round(runtime, 2),
            #     "candidate_pairs": int(blocking_ev['False Positives']) + int(blocking_ev['True Positives']),
            #     "psi": psi,
            #     "lambda": lambda_,
            #     "distance_threshold": distance_threshold
            # }
            #
            # for keys, items in blocking_ev.items():
            #     blocking_evaluation[keys] = round(items, 2) if isinstance(items, float) else items
            #
            # blocking_evaluation['config'] = config
            # df = pd.DataFrame([blocking_evaluation])
            # loaded = os.path.exists(f"{RESULTS_BLOCKING}/lsh_blocker.csv")
            # df.to_csv(f'{RESULTS_BLOCKING}/lsh_blocker.csv',
            #       mode='a+',
            #       index=False,
            #       header=not loaded, float_format='%.2f')
        # filtering_df['total_runtime'] = total_runtime
        # filtering_df['unique_candidate_pairs'] = unique_candidate_pairs
        # filtering_df['Unique Precision %'] = unique_precision
        # filtering_df['Unique F1 %'] = unique_f1
        #
        # cols = list(filtering_df.columns)
        # cols.remove('total_runtime')
        # cols.remove('unique_candidate_pairs')
        #
        # index_col = cols.index('candidate_pairs')
        # cols.insert(index_col,'total_runtime')
        # cols.insert(index_col + 1, 'unique_candidate_pairs')
        #
        # cols.remove('Unique Precision %')
        # cols.remove("Unique F1 %")
        #
        # index_col = cols.index('F1 %')
        # cols.insert(index_col, 'Unique Precision %')
        # cols.insert(index_col + 1, 'Unique F1 %')
        #
        # filtering_df = filtering_df[cols]
        # filtering_df.to_csv(f"{ORIGINAL_FILTERING}/block_filtering.csv",
        #               index=False, float_format='%.2f')


        # del new_bloom
        # del lsh_blocker
        # del block_filter
        # del lsh_blocks
        # del filtered_blocks
        # gc.collect()
        # continue









        # print(f"Running no filtering for {DIR}")

        # size = [64, 128, 256, 512, 1024]
        # if dir_ == 'D10':
        #     size = [s // 2 for s in size]

        # q_grams = [2, 3, 4]

        # TOTAL_LENGTH = 100

        # random.seed(DIR)


        # ratios = []
        # for _ in tqdm(range(TOTAL_LENGTH),
        #           desc="Running no filtering with random configurations",
        #           total=TOTAL_LENGTH):

        #     s = random.choice(size)
        #     q = random.choice(q_grams)

        #     end_range = s // (8 * q)

        #     if end_range == 1:
        #         n_hashes = 1
        #     else:
        #         n_hashes = random.randrange(1,end_range, 1)

        #     configuration = {
        #         'attributes': attributes,
        #         'size': s,
        #         'offset': 0,
        #         'num_hashes': n_hashes,
        #         'hashing_type': 'salted_qgrams',
        #         'salt': '',
        #         'qgrams': q
        #     }

        #     new_bloom : BloomEncodedData = config_bloom_filter_with_configuration(configuration=configuration,
        #                                                     bloom_name=BLOOM_NAME,
        #                                                     loaded_dataframe=loaded_dataframe)


        #     threshold  = random.randrange(10, 55, 10)
        #     thr = threshold / 100
        #     distance_thr = int(thr * s * len(attributes))
        #     start_time = time.time()
        #     bit_blocker_config = BitBlocker.auto_psi_lambda(new_bloom, attributes, thr)
        #     psi = bit_blocker_config['psi']
        #     lambda_ = bit_blocker_config['lambda_']

        #     lsh_blocker = BitBlocker(psi=psi, lambda_=lambda_, seed=42)
        #     lsh_blocks = lsh_blocker.build_blocks(new_bloom)
        #     end_time = time.time()
        #     blocking_ev = lsh_blocker.evaluate(lsh_blocks, verbose=False)


        #     blocking_config = {
        #         "psi": psi,
        #         "lambda": lambda_,
        #         "distance_threshold": distance_thr
        #     }

        #     blocking_evaluation : Dict[str, float | str | int | dict] = {
        #         "name": "LSH-Blocking",
        #         "runtime": round(end_time - start_time, 2),
        #         "candidate_pairs": int(blocking_ev['False Positives']) + int(blocking_ev['True Positives']),
        #         "psi": psi,
        #         "lambda": lambda_,
        #         "distance_threshold": distance_thr
        #     }

        #     for keys, items in blocking_ev.items():
        #         blocking_evaluation[keys] = round(items, 2) if isinstance(items, float) else items

        #     blocking_evaluation['config'] = configuration
        #     df = pd.DataFrame([blocking_evaluation])
        #     loaded = os.path.exists(f"{RESULTS_BLOCKING}/lsh_blocker.csv")
        #     df.to_csv(f'{RESULTS_BLOCKING}/lsh_blocker.csv',
        #           mode='a+',
        #           index=False,
        #           header=not loaded)

        #     continue


            # blocking_runtime = round(end_time - start_time, 2)
            # start_time  = time.time()

            # ratio = random.randrange(20, 90, 10)

            # ratio = ratio / 100
            # ratios.append(ratio)

            # smoothing_factor = 0.9
            # purging = BlockPurging(smoothing_factor=smoothing_factor)

            # blocks_with_keys = purging.process(new_bloom)


            # filtering = BlockFiltering(ratio=ratio)
            # blocks_with_keys = filtering.process(new_bloom)


            # end_time = time.time()

            # filter_blocks = defaultdict(set)
            # d1 = defaultdict(list)
            # d2 = defaultdict(list)
            # for key, id_ in blocks_with_keys:
            #     if id_ < new_bloom.bounds[0]:
            #         d1[key].append(id_)
            #     else:
            #         d2[key].append(id_)


            # for key, d1_entities in d1.items():
            #     if key in d2:
            #         d2_entities = d2[key]
            #         for id1 in d1_entities:
            #             filter_blocks[id1].update(d2_entities)


            # filtering_ev = purging.evaluate(filter_blocks, verbose=False)

            # filtering_config = {
            #     "method": "Block Filtering + Block Purging",
            #     "ratio": ratio,
            #     "smoothing_factor": smoothing_factor,
            # }

            # filtering_evaluation: Dict[str, float | str | int | dict] = {
            #     "name": "Block Filtering + Block Purging",
            #     "ratio": ratio,
            #     "smoothing_factor": smoothing_factor,
            #     "runtime": round(end_time - start_time, 2),
            #     'total_runtime': round(blocking_runtime + (end_time - start_time), 2),
            #     "candidate_pairs": int(filtering_ev['False Positives']) + int(filtering_ev['True Positives']),
            #     "lsh_config": blocking_config,
            # }

            # for keys, items in filtering_ev.items():
            #     filtering_evaluation[keys] = round(items, 2) if isinstance(items, float) else items

            # filtering_evaluation['config'] = configuration

            # df = pd.DataFrame([filtering_evaluation])
            # loaded = os.path.exists(f"{RESULTS_FILTERING}/combination.csv")
            # df.to_csv(f'{RESULTS_FILTERING}/combination.csv',
            #         mode='a+',
            #         index=False,
            #         header=not loaded)



