"""FAISS Results """


import os

import pandas as pd
from tqdm import tqdm
from utils import load_dataframe, config_bloom_filter
from privjedai.blocking import FAISSBlocking

if __name__ == '__main__':


    DIR = "D10"
    PATH = f"data/ccer/{DIR}"
    FILE = 'imdbclean'
    FILE2 = 'dbpediaclean'
    attributes = [['title', 'starring']]
    SEP = "|"

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


    size = [1024//2]
    qgrams = [2]
    TOTAL_LENGTH = 32

    adjacent_config = {}

    for configuration, new_bloom in tqdm(
        config_bloom_filter(attributes, size, qgrams,
                BLOOM_NAME, loaded_dataframe, adjacent_config),
        total=TOTAL_LENGTH):

        for faiss_mode in ['flat', 'hnsw', 'lsh']:
            faiss_blocking = FAISSBlocking(faiss_mode)
            for k in range(10):
                blocks = faiss_blocking.build_blocks(new_bloom, top_k=k+1)
                CNT = 0
                for val in blocks.values():
                    CNT += len(val)

                ev : dict = faiss_blocking.evaluate(blocks, verbose = False)
                faiss_evaluation = {'name': f'FAISS {faiss_mode.upper()} Blocking',
                        'candidate_pairs': CNT, 'top_k': k+1}
                for keys, items in ev.items():
                    faiss_evaluation[keys] = round(items, 2) if isinstance(items, float) else items
                faiss_evaluation['config'] = configuration

                filename = f"test_faiss_{faiss_mode}_bigrams.csv"
                loaded = os.path.exists(f"{RESULTS}/{filename}")
                df = pd.DataFrame([faiss_evaluation])
                df.to_csv(f'{RESULTS}/{filename}',
                mode='a+', index=False, header=not loaded)
