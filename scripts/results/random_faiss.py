"""Random Blocking Experimental Analysis Script"""
import os
import random
from typing import Literal
from tqdm import tqdm
import pandas as pd

from utils import load_dataframe, config_bloom_filter_with_configuration, dataframe_dict
from privjedai.blocking import  FAISSBlocking



if __name__ == '__main__':

    faiss_mode_type = Literal['flat', 'hnsw', 'multihash']
    FAISS_MODES : list[faiss_mode_type] = ['flat', 'hnsw', 'multihash']

    for dir_ in ['D10']:

        DIR = dir_
        PATH = f"data/ccer/{DIR}"

        FILE, FILE2, attributes, SEP = dataframe_dict[DIR]
        RESULTS = f"results/random_configurations/faiss/{DIR}"

        loaded_dataframe = load_dataframe(PATH, (FILE, FILE2), SEP)
        BLOOM_NAME = 'faiss'

        print(f"Running faiss for {DIR}")

        size = [64, 128, 256, 512, 1024]
        if dir_ == 'D10':
            size = [s // 2 for s in size]

        q_grams = [2, 3, 4]

        TOTAL_LENGTH = 100

        random.seed(DIR)

        for _ in tqdm(range(TOTAL_LENGTH),
                  desc=f"Running faiss for {DIR} with random configurations",
                  total=TOTAL_LENGTH):

            s = random.choice(size)
            q = random.choice(q_grams)

            end_range = s // (8 * q)

            if end_range == 1:
                n_hashes = 1
            else:
                n_hashes = random.randrange(1,end_range, 1)

            configuration = {
                'attributes': attributes,
                'size': s,
                'offset': 0,
                'num_hashes': n_hashes,
                'hashing_type': 'salted_qgrams',
                'salt': '',
                'qgrams': q
            }

            new_bloom = config_bloom_filter_with_configuration(configuration=configuration,
                                                            bloom_name=BLOOM_NAME,
                                                            loaded_dataframe=loaded_dataframe)
            for faiss_mode in FAISS_MODES:
                faiss_blocking = FAISSBlocking(faiss_mode)
                m = None
                if faiss_mode == 'hnsw':
                    m = random.choice([32, 64])
                    faiss_blocking.configure_hsnw(m)

                for k in range(10):
                    blocks = faiss_blocking.build_blocks(new_bloom, top_k=k+1)
                    ev = faiss_blocking.evaluate(blocks, verbose=False)

                    faiss_evaluation : dict = {
                        'name': f'FAISS {faiss_mode.upper()} Blocking',
                        'candidate_pairs':  int(ev['False Positives']) + int(ev['True Positives']),
                    }

                    if m is not None:
                        faiss_evaluation['M'] = m
                    faiss_evaluation['top_k'] = k+1
                    faiss_evaluation['index_runtime'] = round(faiss_blocking._index_time, 2)
                    faiss_evaluation['blocking_runtime'] = round(faiss_blocking._blocking_time, 2)
                    faiss_evaluation['runtime'] = round(faiss_blocking.execution_time, 2)
                    for keys, items in ev.items():
                        faiss_evaluation[keys] = round(items, 2) if isinstance(items, float) else items
                    faiss_evaluation['config'] = configuration
                    filename = f"faiss_{faiss_mode}.csv"
                    loaded = os.path.exists(f"{RESULTS}/{filename}")
                    df = pd.DataFrame([faiss_evaluation])
                    df.to_csv(f'{RESULTS}/{filename}',
                              mode='a+', index=False, header=not loaded)
