"""Evaluation script"""
import sys
import pandas as pd
from networkx import Graph
from src.pprl.encoder import BloomFilterConfig, BloomEncodedData, BloomFilter
from src.pprl.blocking import FAISSBlocking
from src.pprl.comparison_cleaning import WeightedEdgePruning, \
    CardinalityEdgePruning, CardinalityNodePruning, WeightedNodePruning, \
    BLAST
from src.pprl.matching import Matcher
from pybloom_live import BloomFilter as Bloom
import itertools
import math
from tqdm import tqdm
from pathlib import Path
import ast


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

    actual_df = pd.read_csv("find_best_run.csv")
    actual_df['cofig'] = actual_df['cofig'].apply(ast.literal_eval)

    configuration = {'attributes': ['name'], 'size': 512, 'offset': 0, 'num_hashes': 10,
                    'hashing_type': 'salted_skipqgrams', 'salt': 'left', 'qgrams': 2}
        
    config = BloomFilterConfig(**configuration)
    bl = BloomFilter(config)
    bloom_filters : BloomEncodedData = bl.encode(d1)
    bloom_filters.to_file('bloom_1.pkl')

    bloom_filters : BloomEncodedData = bl.encode(d2)
    bloom_filters.to_file('bloom_2.pkl')
    new_bloom = BloomEncodedData.from_file('bloom_1.pkl', 'bloom_2.pkl')
    new_bloom.set_ground_truth(gt_index)


    faiss_blocking = FAISSBlocking()
    blocks = faiss_blocking.build_blocks(new_bloom, top_k=5)
    cnt = 0
    for val in blocks.values():
        cnt += len(val)

    
    
    ev = faiss_blocking.evaluate(blocks, verbose = True)
    print(ev)
    matcher = Matcher(batch_size=100,threshold=0.8, metric='scm')
    graph : Graph = matcher.predict(encoded_data=new_bloom, blocks=blocks)
    ev = matcher.evaluate(graph, verbose = True)
    print(ev)




    for cc in [WeightedEdgePruning,
               WeightedNodePruning,
               CardinalityNodePruning,
               CardinalityEdgePruning,
            BLAST]:
        comparison_cleaning_list = []
        my_ev = {"name" : cc.__name__ }

        wep = WeightedEdgePruning()
        new_candidates = wep.process(encoded_data=new_bloom, adjacent_bits=10)

        ev = wep.evaluate(new_candidates)
        print(ev)        
        matcher = Matcher(batch_size=100,threshold=0.8, metric='scm')
        graph : Graph = matcher.predict(encoded_data=new_bloom, blocks=new_candidates)
        
        ev = matcher.evaluate(graph)
        exit(10)
