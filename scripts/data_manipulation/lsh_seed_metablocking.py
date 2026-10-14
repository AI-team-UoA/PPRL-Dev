import os

import pandas as pd
import ast
import random
import time
import numpy as np
from tqdm import tqdm
from multiprocessing import Process

from collections import defaultdict

from utils import load_dataframe, config_bloom_filter_with_configuration, dataframe_dict
from privjedai.blocking import  BitBlocker
from privjedai.block_filtering import BlockPurging, BlockFiltering
from privjedai.encoded_data import BloomEncodedData
from privjedai.comparison_cleaning import WeightedEdgePruning, WeightedNodePruning, CardinalityNodePruning, CardinalityEdgePruning, BLAST


metablocking_dict = {
    "WEP": WeightedEdgePruning,
    "WNP": WeightedNodePruning,
    "CNP": CardinalityNodePruning,
    "CEP": CardinalityEdgePruning,
    "BLAST": BLAST
}


def bitblocker_function(method_, new_bloom_, results_row_, seed_):

    start_time = time.time()

    lsh_config = ast.literal_eval(results_row_['lsh_config'])

    psi = lsh_config['psi']
    lambda_ = lsh_config['lambda']
    blocker = BitBlocker(psi=psi, lambda_=lambda_, seed=seed_)
    blocks = blocker.build_blocks(new_bloom_)
    end_time = time.time()

    runtime = round(end_time - start_time, 2)


    start_time = time.time()

    block_filtering = BlockFiltering(ratio=0.8)
    blocks_filtered = block_filtering.process(new_bloom_)
    end_time = time.time()
    runtime += round(end_time - start_time, 2)

    start_time = time.time()
    wep = metablocking_dict[method_](results_row_['weighting_scheme'])
    blocks_wep = wep.process(new_bloom_)
    end_time = time.time()
    runtime += round(end_time - start_time, 2)
    ev = wep.evaluate(blocks_wep, verbose=False)


    return ev, runtime



def process_seed_run(method_, new_bloom_, results_row_, dataset_, configuration_, seed_):

    ev, runtime = bitblocker_function(method_, new_bloom_, results_row_, 42)

    df_method = pd.DataFrame({
        'Datase`t': [dataset_],
        'Method': [method_],
        'Precision %': [ev['Precision %']],
        'Recall %': [ev['Recall %']],
        'F1 %': [ev['F1 %']],
        'Candidate pairs': [ev['False Positives'] + ev['True Positives']],
        'Runtime': [runtime],
        'Weighting Scheme': [results_row_['weighting_scheme']],
        'Seed': [42],
        'Configuration': [configuration_],
    })


    df_method.to_csv(f"scripts/data_manipulation/seed_runs/{dataset_}_{method_}.csv",
            index=False, float_format='%.2f',
            header=not os.path.exists(f"scripts/data_manipulation/seed_runs/{dataset_}_{method_}.csv"),
            mode='a+')




def main_process(results_row_, method_, new_bloom_, dataset_, configuration_):



    ev, runtime = bitblocker_function(method_, new_bloom_, results_row_, 42)

    df_method = pd.DataFrame({
        'Dataset': [dataset_],
        'Method': [method_],
        'Precision %': [ev['Precision %']],
        'Recall %': [ev['Recall %']],
        'F1 %': [ev['F1 %']],
        'Candidate pairs': [ev['False Positives'] + ev['True Positives']],
        'Runtime': [runtime],
        'Weighting Scheme': [results_row_['weighting_scheme']],
        'Seed': [42],
        'Configuration': [configuration_],
    })

    df_method.to_csv(f"scripts/data_manipulation/seed_runs/{dataset_}_{method_}.csv",
            index=False, float_format='%.2f',
            header=not os.path.exists(f"scripts/data_manipulation/seed_runs/{dataset_}_{method_}.csv"),
            mode='a+')

    for _ in tqdm(range(9), position=2, total=9, desc=f"Processing seeds for {dataset_} - {method_}", leave=False):
        seed = random.randint(0, 10000)
        p_sub = Process(target=process_seed_run,
            kwargs={
                'method_': method_,
                'new_bloom_': new_bloom_,
                'results_row_': results_row_,
                'dataset_': dataset_,
                'configuration_': configuration_,
                'seed_': seed
            }
        )

        p_sub.start()
        p_sub.join()








if __name__ == "__main__":
    df = pd.read_csv("scripts/data_manipulation/metablocking_selected_configurations.csv")

    original_index = df['Original Row Index'].values


    methods = df['Method'].values
    datasets = df['Dataset'].values

    metablocking_filepath  = 'results/random_configurations/metablocking'

    filenames = {
        'WEP': 'WeightedEdgePruning.csv',
        'WNP': 'WeightedNodePruning.csv',
        'CNP': 'CardinalityNodePruning.csv',
        'CEP': 'CardinalityEdgePruning.csv',
        'BLAST': 'BLAST.csv',
    }


    random.seed(42)

    cols_avg = [
        'Dataset', 'Method', 'Precision %', 'Recall %', 'F1 %', 'Unique Precision %',
        'Unique F1 %',
        'Candidate pairs', 'Unique Candidate pairs', 'Runtime', 'Ratio'
    ]

    for i, orig_index in tqdm(enumerate(original_index), total=len(original_index), desc="Processing configurations"):


        results_df = pd.read_csv(f"{metablocking_filepath}/{datasets[i]}/{filenames[methods[i]]}")
        results_row = results_df.iloc[orig_index]


        DATASET = datasets[i]

        if DATASET == "D10":
            time.sleep(0.5)
            continue

        PATH = f"data/ccer/{DATASET}"

        FILE, FILE2, attributes, SEP = dataframe_dict[DATASET]
        loaded_dataframe = load_dataframe(PATH, (FILE, FILE2), SEP)
        BLOOM_NAME = "seeding"


        configuration = ast.literal_eval(results_row['config'])
        new_bloom = config_bloom_filter_with_configuration( configuration, BLOOM_NAME, loaded_dataframe)


        pr = Process(target=main_process,
            kwargs={
                'results_row_': results_row,
                'method_': methods[i],
                'new_bloom_': new_bloom,
                'dataset_': DATASET,
                'configuration_': configuration
            })

        pr.start()
        pr.join()

    print("Done processing all configurations. Results saved in 'scripts/data_manipulation/seed_runs/' and 'scripts/data_manipulation/avg_selected_configurations.csv'.")









