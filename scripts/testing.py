import pandas as pd

from pprl.encoder import BloomFilter, BloomFilterConfig
from pprl.datamodel import  Block
from pprl.encoded_data import BloomEncodedData



# # Helper to generate random strings
# def random_string(length=10):
#     return ''.join(random.choice(string.ascii_letters) for _ in range(length))



if __name__ == '__main__':
    path = "data/test"

    d1 = pd.read_csv(f"{path}/test1.csv", sep =',')
    config = BloomFilterConfig(size=256,
        num_hashes=4,
        offset=0,qgrams=2,
        hashing_type='salted_qgrams',
        attributes = ["name"])

    bl = BloomFilter(config)

    bloom_filters : BloomEncodedData = bl.encode(d1)
    bloom_filters.to_file('bloom_1.pkl')

    d2 = pd.read_csv(f"{path}/test2.csv", sep=',')
    bloom_filters : BloomEncodedData = bl.encode(d2)
    bloom_filters.to_file('bloom_2.pkl')

    new_bloom = BloomEncodedData.from_file('bloom_1.pkl', 'bloom_2.pkl')


    gt = pd.read_csv(f'{path}/gt.csv' , sep=',')
    new_bloom.set_ground_truth(gt)
    new_bloom.serial_evaluate(adjacent_bits=1, verbose=True)
    print(new_bloom.candidate_pairs)
    print(len(new_bloom.candidate_pairs))

    exit(10)
    # new_enc_bitarray = new_bloom.bitarray_dict.copy()
    # print(new_bloom.bitarray_dict)
    lsh = LSHBlocker(psi=4, lmbda=2, seed=1, prune_ratio=0.9)
    blocks = lsh.build_blocks(new_bloom)

    exit(10)
    blocks = faiss_blocking.build_blocks(new_bloom, new_bloom.attributes, top_k=2)
    faiss_blocking.evaluate(blocks)

    matcher = Matcher(batch_size=1, threshold=0.7, metric='dice', concat_bf=False)
    graph : Graph = matcher.predict(encoded_data=new_bloom, blocks=blocks)
    ev = matcher.evaluate(graph)


    print(graph.edges)

    exit(10)

    wep = WeightedEdgePruning()
    new_candidates = wep.process(encoded_data=new_bloom, adjacent_bits=8)
    print(new_candidates)

    exit(10)


    blocker = LSHBlocker(psi=4, lmbda=10, prune_ratio=0.6)
    blocks : Dict[str, Block]= blocker.build_blocks(new_bloom)


    for u, v in graph.edges:
        print(u,v)

