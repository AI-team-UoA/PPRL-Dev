import os

import pandas as pd
import ast
import random
import time
import numpy as np
from tqdm import tqdm


from collections import defaultdict

from utils import load_dataframe, config_bloom_filter_with_configuration, dataframe_dict
from privjedai.blocking import  BitBlocker
from privjedai.block_filtering import BlockPurging, BlockFiltering
from privjedai.encoded_data import BloomEncodedData


if __name__ == "__main__":
    df = pd.read_csv("scripts/data_manipulation/selected_configurations.csv")
    df_avg = pd.read_csv("scripts/data_manipulation/avg_selected_configurations.csv")

    original_index = df['Original Row Index'].values

    methods = df['Method'].values
    datasets = df['Dataset'].values
    ratios = df['Ratio'].values


    filepaths = {
        'LSH': 'results/random_configurations/lsh_blocker',
        'LSH+BP': 'results/random_configurations/filtering',
        'LSH+BF': 'results/random_configurations/filtering',
        'LSH+BP+BF': 'results/random_configurations/filtering',
    }

    filenames = {
        'LSH': 'lsh_blocker.csv',
        'LSH+BP' : 'block_purging.csv',
        'LSH+BF' : 'block_filtering.csv',
        'LSH+BP+BF' : 'combination.csv',
    }

    random.seed(42)

    cols_avg = [
        'Dataset', 'Method', 'Precision %', 'Recall %', 'F1 %', 'Unique Precision %',
        'Unique F1 %',
        'Candidate pairs', 'Unique Candidate pairs', 'Runtime', 'Ratio'
    ]

    for i, orig_index in tqdm(enumerate(original_index), total=len(original_index), desc="Processing configurations"):

        if i < df_avg.shape[0]:
            continue

        precision, recall, f1, unique_precision, unique_f1, candidate_pairs, unique_candidate_pairs, runtimes = [], [], [], [], [], [], [], []
        results_df = pd.read_csv(f"{filepaths[methods[i]]}/{datasets[i]}/{filenames[methods[i]]}")
        results_row = results_df.iloc[orig_index]


        DATASET = datasets[i]
        PATH = f"data/ccer/{DATASET}"

        FILE, FILE2, attributes, SEP = dataframe_dict[DATASET]
        loaded_dataframe = load_dataframe(PATH, (FILE, FILE2), SEP)
        BLOOM_NAME = "seeding"


        configuration = ast.literal_eval(results_row['config'])
        new_bloom = config_bloom_filter_with_configuration( configuration, BLOOM_NAME, loaded_dataframe)


        psi, lambda_ = None, None

        if methods[i] == 'LSH':
            psi = results_row['psi']
            lambda_ = results_row['lambda']
        else:
            lsh_config = ast.literal_eval(results_row['lsh_config'])
            psi = lsh_config['psi']
            lambda_ = lsh_config['lambda']


        start_time = time.time()
        blocker = BitBlocker(psi=psi, lambda_=lambda_, seed=42)
        blocks = blocker.build_blocks(new_bloom)
        end_time = time.time()

        runtime = round(end_time - start_time, 2)

        ev = blocker.evaluate(blocks, verbose=False)
        ev_with_keys = blocker.evaluate_blocks(verbose=False)

        if methods[i] == 'LSH+BP':
            start_time = time.time()
            block_purging = BlockPurging(smoothing_factor=0.9)
            blocks_purged = block_purging.process(new_bloom)
            end_time = time.time()
            runtime += round(end_time - start_time, 2)
            ev = block_purging.evaluate(verbose=False)
            ev_with_keys = block_purging.evaluate_blocks(verbose=False)
        elif methods[i] == 'LSH+BF':
            start_time = time.time()
            block_filtering = BlockFiltering(ratio=ratios[i])
            blocks_filtered = block_filtering.process(new_bloom)
            end_time = time.time()
            runtime += round(end_time - start_time, 2)
            ev = block_filtering.evaluate(verbose=False)
            ev_with_keys = block_filtering.evaluate_blocks(verbose=False)
        elif methods[i] == 'LSH+BP+BF':
            start_time = time.time()
            block_purging = BlockPurging(smoothing_factor=0.9)
            blocks_purged = block_purging.process(new_bloom)
            block_filtering = BlockFiltering(ratio=ratios[i])
            blocks_filtered = block_filtering.process(new_bloom)
            end_time = time.time()
            runtime += round(end_time - start_time, 2)
            ev = block_filtering.evaluate(verbose=False)
            ev_with_keys = block_filtering.evaluate_blocks(verbose=False)


        precision.append(ev_with_keys['Precision %'])
        recall.append(ev_with_keys['Recall %'])
        f1.append(ev_with_keys['F1 %'])
        unique_precision.append(ev['Precision %'])
        unique_f1.append(ev['F1 %'])
        candidate_pairs.append(int(ev_with_keys['False Positives']) + int(ev_with_keys['True Positives']))
        unique_candidate_pairs.append(int(ev['False Positives']) + int(ev['True Positives']))
        runtimes.append(runtime)

        df_method = pd.DataFrame({
            'Dataset': [DATASET],
            'Method': [methods[i]],
            'Precision %': [precision[0]],
            'Recall %': [recall[0]],
            'F1 %': [f1[0]],
            'Unique Precision %': [unique_precision[0]],
            'Unique F1 %': [unique_f1[0]],
            'Candidate pairs': [candidate_pairs[0]],
            'Unique Candidate pairs': [unique_candidate_pairs[0]],
            'Runtime': [runtimes[0]],
            'Ratio': [ratios[i]],
            'Seed': [42],
            'Configuration': [configuration],
        })

        df_method.to_csv(f"scripts/data_manipulation/seed_runs/{DATASET}_{methods[i]}.csv",
                index=False, float_format='%.2f',
                header=not os.path.exists(f"scripts/data_manipulation/seed_runs/{DATASET}_{methods[i]}.csv"),
                mode='a+')


        for j in range(9):
            seed = random.randint(0, 10000)
            start_time = time.time()
            blocker = BitBlocker(psi=psi, lambda_=lambda_, seed=seed)
            blocks = blocker.build_blocks(new_bloom)
            end_time = time.time()

            runtime = round(end_time - start_time, 2)

            ev = blocker.evaluate(blocks, verbose=False)
            ev_with_keys = blocker.evaluate_blocks(verbose=False)

            if methods[i] == 'LSH+BP':
                start_time = time.time()
                block_purging = BlockPurging(smoothing_factor=0.9)
                blocks_purged = block_purging.process(new_bloom)
                end_time = time.time()
                runtime += round(end_time - start_time, 2)
                ev = block_purging.evaluate(verbose=False)
                ev_with_keys = block_purging.evaluate_blocks(verbose=False)
            elif methods[i] == 'LSH+BF':
                start_time = time.time()
                block_filtering = BlockFiltering(ratio=ratios[i])
                blocks_filtered = block_filtering.process(new_bloom)
                end_time = time.time()
                runtime += round(end_time - start_time, 2)
                ev = block_filtering.evaluate(verbose=False)
                ev_with_keys = block_filtering.evaluate_blocks(verbose=False)
            elif methods[i] == 'LSH+BP+BF':
                start_time = time.time()
                block_purging = BlockPurging(smoothing_factor=0.9)
                blocks_purged = block_purging.process(new_bloom)
                block_filtering = BlockFiltering(ratio=ratios[i])
                blocks_filtered = block_filtering.process(new_bloom)
                end_time = time.time()
                runtime += round(end_time - start_time, 2)
                ev = block_filtering.evaluate(verbose=False)
                ev_with_keys = block_filtering.evaluate_blocks(verbose=False)


            precision.append(ev_with_keys['Precision %'])
            recall.append(ev_with_keys['Recall %'])
            f1.append(ev_with_keys['F1 %'])
            unique_precision.append(ev['Precision %'])
            unique_f1.append(ev['F1 %'])
            candidate_pairs.append(int(ev_with_keys['False Positives']) + int(ev_with_keys['True Positives']))
            unique_candidate_pairs.append(int(ev['False Positives']) + int(ev['True Positives']))
            runtimes.append(runtime)

            df_method = pd.DataFrame({
                'Dataset': [DATASET],
                'Method': [methods[i]],
                'Precision %': [precision[j + 1]],
                'Recall %': [recall[j + 1]],
                'F1 %': [f1[j + 1]],
                'Unique Precision %': [unique_precision[j+1]],
                'Unique F1 %': [unique_f1[j+1]],
                'Candidate pairs': [candidate_pairs[j + 1]],
                'Unique Candidate pairs': [unique_candidate_pairs[j + 1]],
                'Runtime': [runtimes[j + 1]],
                'Ratio': [ratios[i]],
                'Seed': [seed],
                'Configuration': [configuration],
            })

            df_method.to_csv(f"scripts/data_manipulation/seed_runs/{DATASET}_{methods[i]}.csv",
                index=False, float_format='%.2f',
                header=not os.path.exists(f"scripts/data_manipulation/seed_runs/{DATASET}_{methods[i]}.csv"),
                mode='a+')

        df_method = pd.DataFrame({
            'Dataset': [DATASET],
            'Method': [methods[i]],
            'Precision %': [np.mean(precision)],
            'Recall %': [np.mean(recall)],
            'F1 %': [np.mean(f1)],
            'Unique Precision %': [np.mean(unique_precision)],
            'Unique F1 %': [np.mean(unique_f1)],
            'Candidate pairs': [np.mean(candidate_pairs)],
            'Unique Candidate pairs': [np.mean(unique_candidate_pairs)],
            'Runtime': [np.mean(runtimes)],
            'Ratio': [ratios[i]],
            'Configuration': [configuration],
        })


        df_method.to_csv(f"scripts/data_manipulation/avg_selected_configurations.csv",
                index=False, float_format='%.2f',
                header=not os.path.exists(f"scripts/data_manipulation/avg_selected_configurations.csv"),
                mode='a+')

    print("Done processing all configurations. Results saved in 'scripts/data_manipulation/seed_runs/' and 'scripts/data_manipulation/avg_selected_configurations.csv'.")








