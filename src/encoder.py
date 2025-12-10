"""Encoder Evaluation """


import os
from itertools import product

import pandas as pd
from tqdm import tqdm

from pprl.encoder import BloomFilterConfig, BloomEncodedData, BloomFilter



def evaluate():
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


    attributes = [['name']]
    size = [64, 128, 256, 512, 1024]
    qgrams = [2,3,4]

    TOTAL_OUTER = len(attributes) * len(size) * len(qgrams)

    for a, s, q in tqdm(product(attributes, size, qgrams),
                total=TOTAL_OUTER,
                desc='Configuration building', position=0, leave=True):
        total_inner = int(s/(8*q))
        for n in tqdm(range(total_inner),
                    total=total_inner,
                    desc='Configuration building 2', position=1, leave=False):
            configuration = {
                'attributes': a,
                'size': s,
                'offset': 0,
                'num_hashes': n+1,
                'hashing_type': 'salted_qgrams',
                'salt': '',
                'qgrams': q
            }


            BLOOM_NAME = 'encoder'
            config = BloomFilterConfig(**configuration)
            bl = BloomFilter(config)
            bloom_filters : BloomEncodedData = bl.encode(d1)
            bloom_filters.to_file(f'{BLOOM_NAME}_1.pkl')

            bloom_filters : BloomEncodedData = bl.encode(d2)
            bloom_filters.to_file(f'{BLOOM_NAME}_2.pkl')
            new_bloom = BloomEncodedData.from_file(f'{BLOOM_NAME}_1.pkl', f'{BLOOM_NAME}_2.pkl')
            new_bloom.set_ground_truth(gt_index)


            for adjacent_bits in [1,2,4,8,10,12]:

                DIR = "results"
                FILENAME = f"blocking_encoder_{adjacent_bits}.csv"

                loaded = os.path.exists(f"{DIR}/{FILENAME}")


                conf_list = []



                ev: dict = new_bloom.evaluate(adjacent_bits = adjacent_bits, verbose = True)

                candidate_pairs: list = new_bloom.get_candidate_pairs(adjacent_bits=adjacent_bits)


                my_ev = {'name': 'Encoder',
                        'candidate_pairs': len(candidate_pairs),
                    }

                for keys, items in ev.items():
                    my_ev[keys] = round(items, 2) if isinstance(items, float) else items

                my_ev['config'] = configuration
                df = pd.DataFrame([my_ev])
                df.to_csv(f'{DIR}/{FILENAME}', mode='a+', index=False, header=not loaded)
                loaded = os.path.exists(f"{DIR}/{FILENAME}")


def get_results():
    i_list = [1, 2, 4, 8, 10, 12]
    df_list = []
    for i in i_list:
        filename = f'results/encoder_{i}.csv'
        df = pd.read_csv(filename)
        df['adjacent_bits'] = i
        df_list.append(df)

    df = pd.concat(df_list, ignore_index=False)

    df = df.sort_values(by=['candidate_pairs', 'Recall %'], ascending=[True, False])

    df = df.head(100)
    df.to_csv('results/encoder_top_100.csv', index=False)




if __name__ == '__main__':
    evaluate()