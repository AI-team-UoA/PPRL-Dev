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

    data = Data(dataset_1=d1, dataset_2=d2, id_column_name_1='id', id_column_name_2='id')

    bl = BloomFilter(size=1000, qgrams=2, hashing_type="seeded_qgrams")
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
