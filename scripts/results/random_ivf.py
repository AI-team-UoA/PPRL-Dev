import ast
import math
import time
import pandas as pd
import random
import os
from privjedai.blocking import FAISSBlocking
from tqdm import tqdm
from multiprocessing import Process
from utils import load_dataframe, config_bloom_filter_with_configuration, dataframe_dict




def evaluate_ivf(config_: str, bloom_name: str, loaded_dataframe_, results_: str):

    config_dict = ast.literal_eval(config_)
    new_bloom = config_bloom_filter_with_configuration(configuration=config_dict,
                                                       bloom_name=bloom_name,
                                                       loaded_dataframe=loaded_dataframe_)
    faiss = FAISSBlocking(index_type='ivf')
    absolut_max_nlist = max(1, int(new_bloom.bounds[0] / 39))
    sqrt_n = math.sqrt(new_bloom.bounds[0])
    min_nlist = max(16, int(2 * sqrt_n))
    max_nlist = min(absolut_max_nlist, int(10 * sqrt_n))

    if min_nlist > max_nlist:
        min_nlist = max(1, max_nlist // 2)

    nlist = random.randint(min_nlist, max_nlist)
    max_allowed_nprobe = max(2, int(nlist * 0.2))

    valid_nprobes = []
    power = 1
    while (2 ** power) <= max_allowed_nprobe:
        valid_nprobes.append(2 ** power)
        power += 1

    if not valid_nprobes:
        valid_nprobes = [1,2]

    nprobe = random.choice(valid_nprobes)

    faiss.configure_ivf(nlist, nprobe)
    for k in range(10):
        start_time = time.time()
        blocks = faiss.build_blocks(encoded_data=new_bloom, top_k=k+1)
        end_time = time.time()
        ev = faiss.evaluate(blocks, verbose=False)
        faiss_evaluation : dict = {
            'name': f'FAISS IVF Blocking',
            'candidate_pairs': int(ev['False Positives']) + int(ev['True Positives']),
            'nlist' : nlist,
            'nprobe' : nprobe,
            'top_k': k,
            'index_runtime': faiss._index_time,
            'blocking_runtime': faiss._blocking_time,
            'runtime': end_time - start_time,
        }
        for keys, items in ev.items():
            faiss_evaluation[keys] = items
        faiss_evaluation['config'] = config_dict
        filename = f"faiss_ivf.csv"
        loaded = os.path.exists(f"{results_}/{filename}")
        df = pd.DataFrame([faiss_evaluation])
        df.to_csv(f'{results_}/{filename}',
                  mode='a+', index=False, header=not loaded,
                  float_format="%.2f")





if __name__ == '__main__':

    dir_list = ['D2', 'D3', 'D4', 'D8', 'D9', 'D10']
    for dir_ in dir_list:
        DIR = dir_

        random.seed(dir_)
        PATH = f"data/ccer/{DIR}"
        FILE, FILE2, attributes, SEP = dataframe_dict[DIR]
        RESULTS_FAISS = f"results/random_configurations/faiss/{DIR}"

        hnsw_df = pd.read_csv(f"{RESULTS_FAISS}/faiss_hnsw.csv")
        config_list = hnsw_df['config'].values

        loaded_dataframe = load_dataframe(PATH, (FILE, FILE2), SEP)
        BLOOM_NAME = 'faiss'

        config_index  = list(range(0, 10*100, 10))

        config_list = config_list[config_index]

        for config in tqdm(config_list, total=100, leave=False, desc=f'Running FAISS IVF for {DIR}'):
            p = Process(target=evaluate_ivf,
                        kwargs={
                            "config_": config,
                            "bloom_name": BLOOM_NAME,
                            "loaded_dataframe_": loaded_dataframe,
                            "results_": RESULTS_FAISS
                        })
            p.start()
            p.join()

