from src.pprl.encoder import BloomFilter
from src.pprl.datamodel import Data, Block, BloomEncodedData
from typing import Dict
import itertools
import pandas as pd
from src.pprl.blocking import LSHBlocker, BitBlocker
from ordered_set import OrderedSet
import numpy as np
from typing import List, Tuple
import bitarray
from networkx import Graph
from src.pprl.matching import Matcher


# # Helper to generate random strings
# def random_string(length=10):
#     return ''.join(random.choice(string.ascii_letters) for _ in range(length))

    

if __name__ == '__main__':
    path = "data/test" 
    d1 = pd.read_csv(f"{path}/test1.csv")
    bl = BloomFilter(size=2048,
                num_hashes = 100,
                offset = 0,
                qgrams=2,
                hashing_type="salted_metaphone")

    bloom_filters : BloomEncodedData = bl.encode(d1)
    # print(bloom_filters)
    bloom_filters.to_file('bloom_1.pkl')
    
    # new_bloom = BloomEncodedData.from_file('bloom_1.pkl')

    # print(new_bloom)
    d2 = pd.read_csv(f"{path}/test2.csv")
    bloom_filters : BloomEncodedData = bl.encode(d2)
    bloom_filters.to_file('bloom_2.pkl')

    
    new_bloom = BloomEncodedData.from_file('bloom_1.pkl', 'bloom_2.pkl')
    # print(new_bloom.bittaray_dict)
    
    # exit(10)
    
    # print('\n', bloom_filters)

    
    blocker = LSHBlocker(psi=32, lmbda=10)
    blocks : Dict[str, Block]= blocker.build_blocks(new_bloom)
    
    for key, block in blocks.items():
        print(key, block) 

    exit(10)
    matcher = Matcher(batch_size=1, threshold=1.0, metric='scm')
    graph : Graph = matcher.predict(data, encoded_data=bloom_filters)

    # TO DO : Add extract pairs and evaluation
    pairs_list = list()   
    for edge in graph.edges:
        node1, node2 = edge
        if node1 < data.dataset_limit:
            id1 = data._gt_to_ids_reversed_1[node1]
            id2 = data._gt_to_ids_reversed_2[node2]
        
        else:
            id1 = data._gt_to_ids_reversed_1[node2]
            id2 = data._gt_to_ids_reversed_2[node1]

        pairs_list.append((id1, id2))

    
    pairs_df = pd.DataFrame(pairs_list, columns=['id1', 'id2'])
    
    print(pairs_df)
    exit(10)
   
""" MATCHING """

# def dice_coefficient(bf1: BloomFilter, bf2:BloomFilter):
#         bit1 = bf1.bitarray
#         bit2 = bf2.bitarray
    
#         intersection = (bit1 & bit2).count()
#         bf1_len = bit1.count()
#         bf2_len = bit2.count()
#         return 2 * intersection / (bf1_len + bf2_len)

#     matches = []
#     for i in range(len(d1)):
#         for j in range(len(d1), len(d2) + len(d1)): 
#             bloom_filter_1 = bloom_filters[i]
#             bloom_filter_2 = bloom_filters[j]
#             score = 0.0
#             for key, _ in bloom_filter_1.items():
#                 score += dice_coefficient(bloom_filter_1[key], bloom_filter_2[key])
#             avg_score = score / len(bloom_filter_1)
#             print(f"{i} {j} {avg_score}")
#             if avg_score == 1.0 :
#                 matches.append((i,j))
        
#     is_dirty_er = data.is_dirty_er
#     gt_to_ids_reversed_1 = data._gt_to_ids_reversed_1
#     gt_to_ids_reversed_2 = data._gt_to_ids_reversed_2
#     ids_1 = list()
#     ids_2 = list()
#     for i, j in matches: 
#         id1 = gt_to_ids_reversed_1[i]
#         id2 = gt_to_ids_reversed_2[j]
#         ids_1.append(int(id1))
#         ids_2.append(int(id2))
#     pairs_df = pd.DataFrame({"id1" : ids_1, "id2" : ids_2}) 
    
#     merged_df = pd.merge(left=pairs_df, left_on="id1", right=d1, right_on="id")
#     merged_df = merged_df.drop(columns=["id1", "id"])
#     merged_df = pd.merge(left=merged_df, left_on="id2", right=d2, right_on="id")
    
#     merged_df = merged_df.drop(columns=["id2", "id"])
#     print(merged_df)
   
#     exit(10)