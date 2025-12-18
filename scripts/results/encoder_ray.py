"""Encoder Ray Evaluation """


import os
import logging

import pandas as pd
from tqdm import tqdm
import ray

from utils import load_dataframe, config_bloom_filter, evaluate_encoder_ray


# def get_results():
#     i_list = [1, 2, 4, 8, 10, 12]
#     df_list = []
#     for i in i_list:
#         filename = f'results/encoder_{i}.csv'
#         df = pd.read_csv(filename)
#         df['adjacent_bits'] = i
#         df_list.append(df)

#     df = pd.concat(df_list, ignore_index=False)

#     df = df.sort_values(by=['candidate_pairs', 'Recall %'], ascending=[True, False])

#     df = df.head(100)
#     df.to_csv('results/encoder_top_100.csv', index=False)




if __name__ == '__main__':

    DIR = "D2"
    PATH = f"data/ccer/{DIR}"
    FILE = 'abtclean'
    FILE2 = 'buyclean'
    SEP = '|'
    RESULTS = f"results/blocking/{DIR}"
    attributes = [['name']]
    size = [64, 128, 256, 512, 1024]
    qgrams = [2,3,4]
    TOTAL_LENGTH = 266
    MAX_CONCURRENT = 30

    loaded_dataframe = load_dataframe(PATH, (FILE, FILE2), SEP)

    if not ray.is_initialized():
        logging.getLogger("ray").setLevel(logging.ERROR)
        ray.init(log_to_driver=False, logging_level=logging.ERROR)

    futures = []

    for configuration, new_bloom in tqdm(
        config_bloom_filter(attributes, size, qgrams, 'standard_blokcing', loaded_dataframe),
        total=TOTAL_LENGTH):



        for adjacent_bits in [1,2,4,8,10,12]:
            future = evaluate_encoder_ray.remote(new_bloom,
                                adjacent_bits, configuration)
            futures.append(future)
            if len(futures) == MAX_CONCURRENT:
                done, futures = ray.wait(futures, num_returns=1)
                adj_bit, bloom_evaluation = ray.get(done[0])
                FILENAME = f"blocking_encoder_{adj_bit}.csv"
                LOADED = os.path.exists(f"{RESULTS}/{FILENAME}")
                df = pd.DataFrame([bloom_evaluation])
                df.to_csv(f'{RESULTS}/{FILENAME}',
                mode='a+', index=False, header=not LOADED)


    for adjacent_bits, bloom_evaluation in ray.get(futures):
        FILENAME = f"blocking_encoder_{adjacent_bits}.csv"
        LOADED = os.path.exists(f"{RESULTS}/{FILENAME}")
        df = pd.DataFrame([bloom_evaluation])
        df.to_csv(f'{RESULTS}/{FILENAME}',
                mode='a+', index=False, header=not LOADED)
