from src.pprl.encoder import BloomFilter
from src.pprl.datamodel import Data
import itertools
import pandas as pd
from rbloom import Bloom
from src.pprl.blocking import LSHBlocker
import bitarray
# # Helper to generate random strings
# def random_string(length=10):
#     return ''.join(random.choice(string.ascii_letters) for _ in range(length))

    
def dice_coefficient(bf1: Bloom, bf2:Bloom):
    intersection = (bf1 & bf2)
    return 2 * intersection.approx_items / (bf1.approx_items + bf2.approx_items)


if __name__ == '__main__':
    path = "data/test" 
    d1 = pd.read_csv(f"{path}/test1.csv")
    d2 = pd.read_csv(f"{path}/test2.csv")
    print(type(d1['id'][0]))
    data = Data(dataset_1=d1, dataset_2=d2, id_column_name_1='id', id_column_name_2='id')

    bl = BloomFilter(size=1000, qgrams=2, hashing_type="salted_string")
    bloom_filters = bl.encode(data)
    
    # print(bloom_filters)
    
    
    
    if attributes: 
        for id, (_, row) in enumerate(d1.iterrows()): 
            print(id)
            for attribute in attributes:
                print(f"type: {type(row[attribute])} row: {row[attribute]}")                    
    
    exit(10)
    bl.encode(data)
    # print(len(bl.bf_a))
    # for bf1 in bl.bf_a: 
    #     for bf2 in bl.bf_b: 
    #         score = dice_coefficient(bf1, bf2) 
    #         print(f"matching {score}")

    bf1 = bl.bf_a
    bf2 = bl.bf_b

    blocker = LSHBlocker(psi=1, lmbda=10)
    blocker.fit(bf1 + bf2)
    
    blocks_a = blocker.block_all(bf1)
    blocks_b = blocker.block_all(bf2)

    print(blocks_a)
    print(blocks_b)
    
    candidates = set()
    for key in blocks_a:
        ids_a = blocks_a.get(key, [])
        ids_b = blocks_b.get(key, [])
        for i, j in itertools.product(ids_a, ids_b):
            candidates.add((i, j))
    print(candidates)
    
    def dice_sim(b1: bitarray, b2: bitarray) -> float:
        inter = (b1 & b2).count()
        return (2 * inter) / (b1.count() + b2.count()) if (b1.count() + b2.count()) else 0.0

    sim_scores = [(i, j, dice_sim(bf1[i], bf2[j])) for i, j in candidates]
    print(sim_scores)
    # blocking = LSHBlocker



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