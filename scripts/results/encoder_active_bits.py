import ast
from tqdm.auto import tqdm
from typing import Dict, List, Union
import pandas as pd
import ray
import logging

from pprl.encoder import BloomFilterConfig, BloomEncodedData, BloomFilter

from utils import load_dataframe, candidate_pairs_and_active_bits_ray, config_bloom_filter, evaluate_encoder_ray

if __name__ == '__main__':

    # DIR = "D10"
    # PATH = f"data/ccer/{DIR}"
    # FILE = 'imdbclean'
    # FILE2 = 'dbpediaclean'
    # attributes = [['title', "starring"]]

    DIR = "D9"
    PATH = f"data/ccer/{DIR}"
    FILE = 'dblpclean'
    FILE2 = 'scholarclean'
    attributes = [['title']]

    SEP = ">"



    RESULTS = f"results/blocking/{DIR}"
    loaded_dataframe = load_dataframe(PATH, (FILE, FILE2), SEP)
    BLOOM_NAME = 'encoder'
    print(f"Adding Candidate Pairs and Active Bits for {DIR}")
    FILENAME = "blocking_encoder_1.csv"
    df  = pd.read_csv(f'{RESULTS}/{FILENAME}')

    # df.drop(labels=['Average Bits %'], inplace=True)
    df['config'] = df['config'].apply(ast.literal_eval)
    config_list = df['config'].to_list()

    df["size"] = df["config"].str["size"].astype(int)
    df["num_hashes"] = df["config"].str["num_hashes"].astype(int)
    df["qgrams"] = df["config"].str["qgrams"].astype(int)


    df = df.sort_values(
        by=["size", "qgrams", "num_hashes"],
        ascending=[True, True, True]
    )

    config_list = df['config'].to_list()

    if not ray.is_initialized():
        logging.getLogger("ray").setLevel(logging.ERROR)
        ray.init(log_to_driver=False, logging_level=logging.ERROR)


    # CANDIDATES_PAIRS : Union[None, Dict[int, List]] = None if DIR not in ['D8', 'D9', 'D10'] else {}
    CANDIDATES_PAIRS = None
    active_bits = []



    MAX_CONCURRENCY = 24

    futures = []

    for index, configuration in tqdm(
        enumerate(config_list), total=len(config_list)):

        config = BloomFilterConfig(**configuration)
        bl = BloomFilter(config)
        bloom_filters : BloomEncodedData = bl.encode(loaded_dataframe[0])
        bloom_filters.to_file(f'{BLOOM_NAME}_1.pkl')

        bloom_filters : BloomEncodedData = bl.encode(loaded_dataframe[1])
        bloom_filters.to_file(f'{BLOOM_NAME}_2.pkl')
        new_bloom = BloomEncodedData.from_file(f'{BLOOM_NAME}_1.pkl', f'{BLOOM_NAME}_2.pkl')

        new_bloom.set_ground_truth(loaded_dataframe[2])
        if CANDIDATES_PAIRS is not None:
            for adjacent_bits in [1,2,4,8,10,12]:
                future = candidate_pairs_and_active_bits_ray.remote(new_bloom,
                            adjacent_bits, index)
                futures.append(future)

                if len(futures) >= MAX_CONCURRENCY:
                    finished, futures = ray.wait(futures, num_returns = 1)
                    for done in finished:
                        adj_bit, distinct_candidate_pairs, pair_index = ray.get(done)
                        if adj_bit not in CANDIDATES_PAIRS:
                            CANDIDATES_PAIRS[adj_bit] = []
                        CANDIDATES_PAIRS[adj_bit].append((pair_index, distinct_candidate_pairs))
        else:
            if len(futures) >= MAX_CONCURRENCY:
                finished, futures = ray.wait(futures, num_returns = 1)
                for done in finished:
                    adj_bit, distinct_candidate_pairs, pair_index = ray.get(done)
                    active_bits.append((pair_index, distinct_candidate_pairs))

        future = candidate_pairs_and_active_bits_ray.remote(new_bloom,
                            0, index)
        futures.append(future)

    while futures:
        if CANDIDATES_PAIRS is not None:
            finished, futures = ray.wait(futures, num_returns = 1)
            for done in finished:
                adj_bit, distinct_candidate_pairs, pair_index = ray.get(done)
                if adj_bit not in CANDIDATES_PAIRS:
                    CANDIDATES_PAIRS[adj_bit] = []
                CANDIDATES_PAIRS[adj_bit].append((pair_index, distinct_candidate_pairs))
        else:
            finished, futures = ray.wait(futures, num_returns = 1)
            for done in finished:
                adj_bit, distinct_candidate_pairs, pair_index = ray.get(done)
                active_bits.append((pair_index, distinct_candidate_pairs))


    new_dict = {}
    if CANDIDATES_PAIRS is None:
        CANDIDATES_PAIRS = {}
        CANDIDATES_PAIRS[0] = active_bits


    for adj_bit, candidate_pair_list in CANDIDATES_PAIRS.items():
        sorted_list = [x[1] for x in sorted(candidate_pair_list,
                                key=lambda x: x[0])]
        new_dict[adj_bit] = sorted_list

    CANDIDATES_PAIRS = new_dict
    active_bits = CANDIDATES_PAIRS[0]

    for adjacent_bits in [1, 2, 4, 8, 10, 12]:
        FILENAME = f"blocking_encoder_{adjacent_bits}.csv"
        df  = pd.read_csv(f'{RESULTS}/{FILENAME}')

        df['config'] = df['config'].apply(ast.literal_eval)

        df["size"] = df["config"].str["size"].astype(int)
        df["num_hashes"] = df["config"].str["num_hashes"].astype(int)
        df["qgrams"] = df["config"].str["qgrams"].astype(int)

        df = df.sort_values(
            by=["size", "qgrams", "num_hashes"],
            ascending=[True, True, True]
        )

        df['Average Bits %'] = active_bits
        if adjacent_bits in CANDIDATES_PAIRS:
            df['candidate_pairs'] = CANDIDATES_PAIRS[adjacent_bits]

        df.drop(columns=['size', 'num_hashes', 'qgrams'], inplace=True)

        df.to_csv(f'{RESULTS}/{FILENAME}', index=False)
    print(f"Adding Candidate Pairs and Active Bits for {DIR}")
