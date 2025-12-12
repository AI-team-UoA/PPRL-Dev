"""Utils for Results Generation"""

from itertools import product
import pandas as pd
import ray
import ray.actor

from pprl.comparison_cleaning import AbstractMetablocking
from pprl.encoder import BloomFilterConfig, BloomEncodedData, BloomFilter

def load_dataframe(path: str, files: tuple, sep: str):
    """Loads Dataframe"""

    file, file2 = files
    d1 = pd.read_csv(f"{path}/{file}.csv", sep=sep)
    d2 = pd.read_csv(f"{path}/{file2}.csv", sep=sep)

    gt = pd.read_csv(f'{path}/gtclean.csv' , sep=sep)
    # Add index columns for lookup
    df_a = d1.reset_index().rename(columns={"index": "index_A"})
    df_b = d2.reset_index().rename(columns={"index": "index_B"})

    df_a = df_a[["index_A", "id"]]
    df_b = df_b[["index_B", "id"]]

    gt_index = gt.merge(left_on='D1', right=df_a, right_on='id')

    gt_index = gt_index.drop(columns=['id', 'D1'])
    gt_index.columns = ['D2', 'D1']

    gt_index = gt_index.merge(left_on='D2', right=df_b, right_on='id')
    gt_index = gt_index.drop(columns=['id', 'D2'])
    gt_index.columns = ['D1', 'D2']

    return d1, d2, gt_index


def config_bloom_filter(attributes,
        size, qgrams, bloom_name, loaded_dataframe):
    """BF Gerator Method"""

    for a, s, q in product(attributes, size, qgrams):
        n = 20
        # for n in range(int(s/(8*q))):
            # print(f'{n}/{int(s/(8*q))}')
        configuration = {
            'attributes': a,
            'size': s,
            'offset': 0,
            'num_hashes': n,
            'hashing_type': 'salted_qgrams',
            'salt': '',
            'qgrams': q
        }

        config = BloomFilterConfig(**configuration)
        bl = BloomFilter(config)
        bloom_filters : BloomEncodedData = bl.encode(loaded_dataframe[0])
        bloom_filters.to_file(f'{bloom_name}_1.pkl')

        bloom_filters : BloomEncodedData = bl.encode(loaded_dataframe[1])
        bloom_filters.to_file(f'{bloom_name}_2.pkl')
        new_bloom = BloomEncodedData.from_file(f'{bloom_name}_1.pkl', f'{bloom_name}_2.pkl')

            new_bloom.set_ground_truth(loaded_dataframe[2])
            yield configuration, new_bloom


@ray.remote
def comparison_cleaning_exp(cc: callable, new_bloom: BloomEncodedData,
            configuration: dict):
    """Experiments for comparison Cleaning"""
    wep : AbstractMetablocking = cc()

    metablocking_method = cc.__name__
    metablocking_results = []

    for i in [1,2,4,8,10,12]:

        new_candidates = wep.process(encoded_data=new_bloom,
                        adjacent_bits=i)
        ev : dict = wep.evaluate(new_candidates, verbose = False)

        cnt = 0
        for val in new_candidates.values():
            cnt += len(val)

        my_ev = {"name" : metablocking_method,
                'candidate_pairs': cnt,
                'adjacent_bits': i}

        for keys, items in ev.items():
            my_ev[keys] = round(items, 2) if isinstance(items, float) else items

        my_ev['config'] = configuration
        metablocking_results.append(my_ev)

    return metablocking_method, metablocking_results
        new_bloom.set_ground_truth(loaded_dataframe[2])

        yield configuration, new_bloom
