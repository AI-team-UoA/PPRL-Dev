from src.pprl.encoder import BloomFilter, BloomFilterConfig
from src.pprl.datamodel import  Block, BloomEncodedData
from typing import Dict
import itertools
import pandas as pd
from src.pprl.blocking import LSHBlocker, BitBlocker, FAISSBlocking
from ordered_set import OrderedSet
import numpy as np
from typing import List, Tuple
import bitarray
from networkx import Graph
from src.pprl.matching import Matcher
from collections import defaultdict
from src.pprl.comparison_cleaning import WeightedNodePruning, CardinalityNodePruning


# # Helper to generate random strings
# def random_string(length=10):
#     return ''.join(random.choice(string.ascii_letters) for _ in range(length))



if __name__ == '__main__':
    path = "data/test"

    d1 = pd.read_csv(f"{path}/test1.csv", sep =',')
    config = BloomFilterConfig(size=2048,
        num_hashes=20,
        offset=0,qgrams=4,
        hashing_type='salted_string')
    
    bl = BloomFilter(config)

    bloom_filters : BloomEncodedData = bl.encode(d1)
    bloom_filters.to_file('bloom_1.pkl')
    bloom_filters.to_file('bloom_2.pkl')

    d2 = pd.read_csv(f"{path}/test2.csv", sep=',')
    bloom_filters : BloomEncodedData = bl.encode(d2)

    new_bloom = BloomEncodedData.from_file('bloom_1.pkl', 'bloom_2.pkl')


    gt = pd.read_csv(f'{path}/gt.csv' , sep=',')
    new_bloom.set_ground_truth(gt)


    faiss_blocking = FAISSBlocking()
    blocks = faiss_blocking.build_blocks(new_bloom, new_bloom.attributes, top_k=1)

    print(blocks)

    wep = WeightedNodePruning()
    new_candidates = wep.process(encoded_data=new_bloom)

    print(new_candidates)
    matcher = Matcher(batch_size=1, threshold=1.0, metric='dice')
    graph : Graph = matcher.predict(encoded_data=new_bloom, blocks=new_candidates)
    ev = matcher.evaluate(graph)


    print(new_candidates)
    exit(10)


    blocker = LSHBlocker(psi=4, lmbda=10, prune_ratio=0.6)
    blocks : Dict[str, Block]= blocker.build_blocks(new_bloom)


    for u, v in graph.edges:
        print(u,v)

