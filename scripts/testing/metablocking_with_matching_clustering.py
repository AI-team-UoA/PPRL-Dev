"""FAISS Results """

import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from pprl.clustering import (
    ConnectedComponentsClustering,
    KiralyMSMApproximateClustering,
    UniqueMappingClustering
)
from pprl.matching import Matcher

from pprl.comparison_cleaning import (
    WeightedEdgePruning,
    WeightedNodePruning,
    CardinalityEdgePruning,
    CardinalityNodePruning,
)

from utils import (
    config_bloom_filter_with_configuration,
    load_dataframe,
)
# from pprl.blocking import FAISSBlocking

if __name__ == '__main__':


    DIR = "D2"
    PATH = f"data/ccer/{DIR}"
    FILE = 'abtclean'
    FILE2 = 'buyclean'
    attributes = ['name']
    SEP = "|"

    loaded_dataframe = load_dataframe(PATH, (FILE, FILE2), SEP)
    BLOOM_NAME = 'metablocking'
    print(f"Running METABLOCKING for {DIR}")

    wep = CardinalityNodePruning()


    size = 1024
    qgrams = 4

    kiraly = KiralyMSMApproximateClustering()

    num_hashes = 1
    configuration = {'attributes': attributes,
        'size': size,
        'offset': 0,
        'num_hashes': num_hashes,
        'hashing_type': 'salted_qgrams',
        'salt': '',
        'qgrams': qgrams
    }
    new_bloom = config_bloom_filter_with_configuration(configuration,
                        BLOOM_NAME, loaded_dataframe)

    blocks = wep.process(encoded_data = new_bloom,
                adjacent_bits = 12)
    ev = wep.evaluate(blocks, verbose=True)


    metric = 'cosine'
    thr = 0.7
    matcher = Matcher(batch_size=10_000,
                threshold=thr, metric=metric)
    graph = matcher.predict(new_bloom, blocks)
    ev = matcher.evaluate(graph, verbose=True)
    exit(10)


        # new_blocks = {}
        #     if len(blocks) > new_bloom.bounds[0]:
        #         for id, blocks_set in blocks.items():
        #             if id >= new_bloom.bounds[0]:
        #                 if id in new_blocks:
        #                     new_blocks[id].update(blocks_set)
        #                 else:
        #                     for new_block_id in blocks_set:
        #                         if new_block_id < new_bloom.bounds[0]:
        #                             new_blocks[new_block_id].add(id)
        #             else:
        #                 for block_id in blocks_set:
        #                     if block_id >= new_bloom.bounds[0]:
        #                         if id in new_blocks:
        #                             new_blocks[id].add(block_id)
        #                         else:
        #                             new_blocks[id] = {block_id}

        #     blocks = new_blocks
        #     matcher = None
        #         for similarity in tqdm(
        #             range(5,100,5), leave=False, position=2,
        #             desc=f"Matching Similarity {metric}", total=19):
        #             if matcher is None:
        #             else:
        #                 graph = matcher.update_graph(graph, thr)

        #             edges, _ = graph
        #             matcher_ev = {
        #                 'name': f'Matching with {metric}',
        #                 'adjacent_bits': 12,
        #                 'total_matches': edges.shape[0],
        #                 'similarity_threshold': thr}

        #             for keys, items in ev.items():
        #                 matcher_ev[keys] = round(items, 2) if isinstance(items, float) else items

        #             matcher_ev['config'] = configuration
        #             matcher_filename = f"{metric}.csv"
        #             loaded = os.path.exists(f"{RESULTS}/matching/{cc.__class__.__name__}/{matcher_filename}")
        #             df = pd.DataFrame([matcher_ev])
        #             df.to_csv(f'{RESULTS}/matching/{cc.__class__.__name__}/{matcher_filename}', mode='a+', index=False,
        #                 header=not loaded)
        #             for clustering in [kiraly, connected, unique]:
        #                 clustering_name = clustering.__class__.__name__
        #                 clustering_filename = f"{clustering_name}_{metric}.csv"
        #                 clusters = clustering.process(graph, new_bloom, thr)
        #                 ev_cl = clustering.evaluate(clusters, verbose=False)
        #                 cluster_ev = {
        #                     'name': f'Clustering with {clustering_name}',
        #                     'adjacent_bits': 12,
        #                     'total_clusters': len(clusters),
        #                     'threshold': thr,
        #                     'matcher': metric,
        #                 }
        #                 for keys, items in ev_cl.items():
        #                     cluster_ev[keys] = round(items, 2) if isinstance(items, float) else items
        #                 cluster_ev['config'] = configuration
        #                 loaded = os.path.exists(f"{RESULTS}/clustering/{cc.__class__.__name__}/{clustering_filename}")
        #                 df = pd.DataFrame([cluster_ev])
        #                 df.to_csv(f'{RESULTS}/clustering/{cc.__class__.__name__}/{clustering_filename}', mode='a+', index=False,
        #                     header=not loaded)
