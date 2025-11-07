import pandas as pd
from src.pprl.encoder import BloomFilterConfig, BloomEncodedData, BloomFilter
from src.pprl.blocking import FAISSBlocking
from src.pprl.comparison_cleaning import WeightedEdgePruning, \
    CardinalityEdgePruning, CardinalityNodePruning, WeightedNodePruning, \
    BLAST
from src.pprl.matching import Matcher
from networkx import Graph

if __name__ == '__main__':
    path = "data/ccer/D2"

    d1 = pd.read_csv(f"{path}/abtclean.csv", sep ='|')
    config = BloomFilterConfig(size=1048,
        num_hashes=20,
        offset=0,qgrams=4,
        hashing_type='salted_string')

    bl = BloomFilter(config)

    bloom_filters : BloomEncodedData = bl.encode(d1)
    bloom_filters.to_file('bloom_1.pkl')
    bloom_filters.to_file('bloom_2.pkl')

    d2 = pd.read_csv(f"{path}/buyclean.csv", sep='|')
    bloom_filters : BloomEncodedData = bl.encode(d2)

    new_bloom = BloomEncodedData.from_file('bloom_1.pkl', 'bloom_2.pkl')


    gt = pd.read_csv(f'{path}/gtclean.csv' , sep='|')
    # Add index columns for lookup
    df_A = d1.reset_index().rename(columns={"index": "index_A"})
    df_B = d2.reset_index().rename(columns={"index": "index_B"})

    # Merge ground truth with both datasets to get indices
    gt_index = (
        gt
        .merge(df_A[["index_A", "id"]], left_on="D1", right_on="id", how="inner")
    .merge(df_B[["index_B", "id"]], left_on="D2", right_on="id", how="inner")
    [["D1", "D2"]]
    )

    print(gt_index.head(10))
    new_bloom.set_ground_truth(gt)


    faiss_blocking = FAISSBlocking()
    blocks = faiss_blocking.build_blocks(new_bloom, new_bloom.attributes, top_k=30)
    my_ev = {"name" : 'faiss_blocking'}
    ev = faiss_blocking.evaluate(blocks) 


    for keys, items in ev.items():
        my_ev[keys] = items
        
    matcher = Matcher(batch_size=100,threshold=0.8, metric='dice')
    graph : Graph = matcher.predict(encoded_data=new_bloom, blocks=blocks)
    ev = matcher.evaluate(graph)
    print(ev)

    for keys, items in ev.items():
        my_ev[f'matcher_{keys}'] = items
        
    df = pd.DataFrame([my_ev])
    
    df.to_csv('evaluator.csv', mode='a+', index=False, header=False)
    # print(blocks)



    # for cc in [WeightedEdgePruning,
    #            WeightedNodePruning,
    #            CardinalityNodePruning,
    #            CardinalityEdgePruning,
    #         BLAST]:
    #     comparison_cleaning_list = []
    #     my_ev = {"name" : cc.__name__ } 

    #     wep = cc()
    #     new_candidates = wep.process(encoded_data=new_bloom)

    #     ev = wep.evaluate(new_candidates)
    #     for keys, items in ev.items():
    #         my_ev[keys] = items
        
    #     matcher = Matcher(batch_size=100,threshold=0.4, metric='dice')
    #     graph : Graph = matcher.predict(encoded_data=new_bloom, blocks=new_candidates)
    #     ev = matcher.evaluate(graph)

    #     for keys, items in ev.items():
    #         my_ev[f'matcher_{keys}'] = items
        
    #     comparison_cleaning_list.append(my_ev)

    #     df = pd.DataFrame([my_ev])
    
    #     df.to_csv('evaluator.csv', mode='a+', index=False, header=False)

    exit(10)
