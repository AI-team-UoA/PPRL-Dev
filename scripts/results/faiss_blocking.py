"""FAISS Results """


import os

import pandas as pd
from typing import Dict, Literal
from tqdm import tqdm

from utils import load_dataframe, config_bloom_filter
from privjedai.blocking import FAISSBlocking

if __name__ == '__main__':


    DIR = "D9"
    PATH = f"data/ccer/{DIR}"
    FILE = 'dblpclean'
    FILE2 = 'scholarclean'
    attributes = [['title']]
    SEP = ">"

    RESULTS = f"results/faiss/{DIR}"
    loaded_dataframe = load_dataframe(PATH, (FILE, FILE2), SEP)
    BLOOM_NAME = 'faiss'
    print(f"Running FAISS for {DIR}")


    # # d1 = loaded_dataframe[0]
    # # d2 = loaded_dataframe[1]


    # # d1['title'] = d1['title'] + ' ' + d1['starring']
    # # d2['title'] = d2['title'] + ' ' + d2['starring']
    # # attributes = [['title']]

    # loaded_dataframe = d1,d2,loaded_dataframe[2]

    size = [64, 128, 256, 512, 1024]
    qgrams = [2, 3, 4]

    TOTAL_LENGTH = 266


    adjacent_config = {}

    FaissModeType = Literal['flat', 'hnsw', 'multihash']

    faiss_modes : list[FaissModeType] = ['flat', 'hnsw', 'multihash']
    for configuration, new_bloom in tqdm(
        config_bloom_filter(attributes, size, qgrams,
                BLOOM_NAME, loaded_dataframe, adjacent_config),
        total=TOTAL_LENGTH):

        for faiss_mode in faiss_modes:
            faiss_blocking = FAISSBlocking(faiss_mode)
            for k in range(10):
                blocks = faiss_blocking.build_blocks(new_bloom, top_k=k+1)


                ev : dict = faiss_blocking.evaluate(blocks, verbose = False)
                faiss_evaluation : Dict[str, dict | int | str | float] = {
                    'name': f'FAISS {faiss_mode.upper()} Blocking',
                    'candidate_pairs': ev['True Positives'] + ev['False Positives'],
                    'top_k': k+1,
                    'index_runtime': round(faiss_blocking._index_time, 2),
                    'blocking_runtime': round(faiss_blocking._blocking_time, 2),
                    'runtime': round(faiss_blocking.execution_time, 2),
                }

                for keys, items in ev.items():
                    faiss_evaluation[keys] = round(items, 2) if isinstance(items, float) else items
                faiss_evaluation['config'] = configuration

                filename = f"faiss_{faiss_mode}.csv"
                loaded = os.path.exists(f"{RESULTS}/{filename}")
                df = pd.DataFrame([faiss_evaluation])
                df.to_csv(f'{RESULTS}/{filename}',
                mode='a+', index=False, header=not loaded)
