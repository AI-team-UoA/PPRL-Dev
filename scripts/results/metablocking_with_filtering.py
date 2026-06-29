"""Metablocking Results """


import os

import pandas as pd
from tqdm import tqdm
from utils import load_dataframe, config_bloom_filter
from privjedai.blocking import BitBlocker
from privjedai.block_filtering import  BlockFiltering
from privjedai.comparison_cleaning import (WeightedEdgePruning,
    WeightedNodePruning,
    CardinalityEdgePruning,
    CardinalityNodePruning,
)


if __name__ == '__main__':


    DIR = "D2"
    PATH = f"data/ccer/{DIR}"
    FILE = 'abtclean'
    FILE2 = 'buyclean'
    attributes = [['name']]
    SEP = "|"


    results_type = ['blocking', 'filtering', 'metablocking']
    RESULTS = f"results"
    loaded_dataframe = load_dataframe(PATH, (FILE, FILE2), SEP)
    BLOOM_NAME = 'filtering'
    print(f"Running blocking for {DIR}")



    size = [64, 128, 256, 512, 1024]
    qgrams = [2, 3, 4]
    TOTAL_LENGTH = 266


    adjacent_config = {}

    for configuration, new_bloom in tqdm(
        config_bloom_filter(attributes, size, qgrams,
                BLOOM_NAME, loaded_dataframe, adjacent_config),
        total=TOTAL_LENGTH):

        bit_blocker_config = BitBlocker.auto_psi_lambda(
            encoded_data=new_bloom,
            threshold=0.3,
            delta=0.1,
            max_lambda_=200
        )

        blocking = BitBlocker(**bit_blocker_config, seed=42)
        blocks = blocking.build_blocks(new_bloom)
        ev = blocking.evaluate_blocks()

        # blocking_ev: = {
        #     'name': f'Bit Blocker',
        #     'candidate_pairs': ,
        #     'M': f_m_value,
        #     'top_k': f_k, }
        #
        # for faiss_mode in ['flat', 'hnsw', 'lsh']:
        #     faiss_blocking = FAISSBlocking(faiss_mode)
        #     for k in range(10):
        #         blocks = faiss_blocking.build_blocks(new_bloom, top_k=k+1)
        #         CNT = 0
        #         for val in blocks.values():
        #             CNT += len(val)
        #
        #         ev : dict = faiss_blocking.evaluate(blocks, verbose = False)
        #         faiss_evaluation = {'name': f'FAISS {faiss_mode.upper()} Blocking',
        #                 'candidate_pairs': CNT, 'top_k': k+1}
        #         for keys, items in ev.items():
        #             faiss_evaluation[keys] = round(items, 2) if isinstance(items, float) else items
        #         faiss_evaluation['config'] = configuration
        #
        #         filename = f"test_faiss_{faiss_mode}_bigrams.csv"
        #         loaded = os.path.exists(f"{RESULTS}/{filename}")
        #         df = pd.DataFrame([faiss_evaluation])
        #         df.to_csv(f'{RESULTS}/{filename}',
        #         mode='a+', index=False, header=not loaded)
