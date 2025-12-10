"""FAISS Results """


import ast
import os
from itertools import product

import pandas as pd
from tqdm import tqdm

from pprl.blocking import FAISSBlocking
from pprl.encoder import BloomFilterConfig, BloomEncodedData, BloomFilter


if __name__ == '__main__':
    PATH = "data/ccer/D2"
    FILE = 'abtclean'
    FILE2 = 'buyclean'
    SEP = '|'


    d1 = pd.read_csv(f"{PATH}/{FILE}.csv", sep=SEP)
    d2 = pd.read_csv(f"{PATH}/{FILE2}.csv", sep=SEP)

    gt = pd.read_csv(f'{PATH}/gtclean.csv' , sep=SEP)
    # Add index columns for lookup
    df_A = d1.reset_index().rename(columns={"index": "index_A"})
    df_B = d2.reset_index().rename(columns={"index": "index_B"})

    df_A = df_A[["index_A", "id"]]
    df_B = df_B[["index_B", "id"]]

    gt_index = gt.merge(left_on='D1', right=df_A, right_on='id')

    gt_index = gt_index.drop(columns=['id', 'D1'])
    gt_index.columns = ['D2', 'D1']

    gt_index = gt_index.merge(left_on='D2', right=df_B, right_on='id')
    gt_index = gt_index.drop(columns=['id', 'D2'])

    gt_index.columns = ['D1', 'D2']
    total_pairs = len(df_A) * len(df_B)
    metrics = ["Precision %", "Recall %", "F1 %"]

    DIR = "results"
    FILENAME = "faiss_hnsw.csv"

    loaded = os.path.exists(f"{DIR}/{FILENAME}")
    if loaded:
        actual_df = pd.read_csv(f"{DIR}/{FILENAME}")
        actual_df['config'] = actual_df['config'].apply(ast.literal_eval)


    conf_list = []


    attributes = [['name']]
    size = [64, 128, 256, 512, 1024]
    qgrams = [2,3,4]

    for a, s, q in tqdm(product(attributes, size, qgrams),
                desc='Configuration building', position=1):
        for n in tqdm(range(int(s/(8*q))), desc='Configuration building 2', position=2):
            configuration = {
                'attributes': a,
                'size': s,
                'offset': 0,
                'num_hashes': n+1,
                'hashing_type': 'salted_qgrams',
                'salt': '',
                'qgrams': q
            }


            BLOOM_NAME = 'faiss_hnsw'
            config = BloomFilterConfig(**configuration)
            bl = BloomFilter(config)
            bloom_filters : BloomEncodedData = bl.encode(d1)
            bloom_filters.to_file(f'{BLOOM_NAME}_1.pkl')

            bloom_filters : BloomEncodedData = bl.encode(d2)
            bloom_filters.to_file(f'{BLOOM_NAME}_2.pkl')
            new_bloom = BloomEncodedData.from_file(f'{BLOOM_NAME}_1.pkl', f'{BLOOM_NAME}_2.pkl')
            new_bloom.set_ground_truth(gt_index)

            faiss_blocking = FAISSBlocking('hnsw')
            for k in range(10):
                blocks = faiss_blocking.build_blocks(new_bloom, top_k=k+1)
                cnt = 0
                for val in blocks.values():
                    cnt += len(val)

                ev = faiss_blocking.evaluate(blocks, verbose = False)
                my_ev = {'name': 'FAISS HNSW Blocking', 'candidate_pairs': cnt, 'top_k': k}


                for keys, items in ev.items():
                    my_ev[keys] = round(items, 2) if isinstance(items, float) else items

                my_ev['config'] = configuration
                df = pd.DataFrame([my_ev])
                df.to_csv(f'{DIR}/{FILENAME}', mode='a+', index=False, header=not loaded)
                loaded = os.path.exists(f"{DIR}/{FILENAME}")
