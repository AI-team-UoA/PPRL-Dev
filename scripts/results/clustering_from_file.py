from __future__ import annotations
import os
import pandas as pd
from tqdm import tqdm
from utils import load_dataframe, config_bloom_filter
import rmm
# rmm.reinitialize(pool_allocator=False)

# Enable managed memory to allow oversubscribing GPU memory
rmm.reinitialize(pool_allocator=True, managed_memory=True)
from rmm.allocators.cupy import rmm_cupy_allocator
import cupy as cp
cp.cuda.set_allocator(rmm_cupy_allocator)
import pickle
# from pprl.matching_gpu import Matcher
from pprl.clustering_gpu import UniqueMappingClustering, \
    ConnectedComponentsClustering, KiralyMSMApproximateClustering


if __name__ == '__main__':

    DIR = "D10"
    PATH = f"data/ccer/{DIR}"
    FILE = 'imdbclean'
    FILE2 = 'dbpediaclean'
    attributes = [['title', 'starring']]
    SEP = "|"

    RESULTS = f"results/matching/{DIR}"
    RESULTS_CLUSTERING = f"results/clustering/{DIR}"

    loaded_dataframe = load_dataframe(PATH, (FILE, FILE2), SEP)
    BLOOM_NAME = 'matching'
    print(f"Running MATCHING & CLUSTERING for {DIR}")


    size = [1024 // 2]
    qgrams = [4]
    TOTAL_LENGTH = 32
    adjacent_config = {'0' :[]}

    FINISHED = 1
    for num_hashes in range(FINISHED):
        config_dict = {'attributes': attributes[0],
                'size': size,
                'offset': 0,
                'num_hashes': num_hashes+1,
                'hashing_type': 'salted_qgrams',
                'salt': '',
                'qgrams': qgrams}
        adjacent_config['0'].append(config_dict)

    for configuration, new_bloom in tqdm(
        config_bloom_filter(attributes, size, qgrams,
                BLOOM_NAME, loaded_dataframe, adjacent_config),
        total=TOTAL_LENGTH):
        if new_bloom is None:
            continue

        num_hashes = configuration['num_hashes']
        if num_hashes == 8:
            break
        graph_path = f".graphs/bloom_{num_hashes}"
        for metric in ['cosine', 'dice', 'jaccard']:
            for similarity in tqdm(range(5, 100, 5), total=19):
                graph_index = similarity // 5
                graph = None
                graph_file_name = f'{graph_path}/{metric}/graph_{graph_index}.pkl'

                for cl in [KiralyMSMApproximateClustering,
                        ConnectedComponentsClustering,
                        UniqueMappingClustering
                    ]:
                    with open(graph_file_name, "rb") as f:
                        graph = pickle.load(f)

                    edges = cp.asarray(graph['edges'])
                    weights = graph['weights']

                    n_graph = edges, weights

                    thr = similarity / 100


                    clustering_method = cl()
                    clusters = clustering_method.process(n_graph, new_bloom, thr)
                    ev_cl = clustering_method.evaluate(clusters, verbose=False)
                    cluster_ev = {
                        'name': f'Clustering with {cl.__name__}',
                        'total_clusters': len(clusters),
                        'threshold': thr,
                        'matcher': metric,
                    }

                    for keys, items in ev_cl.items():
                        cluster_ev[keys] = round(items, 2) \
                        if isinstance(items, float) else items

                    cluster_ev['config'] = configuration
                    clustering_name = cl.__name__
                    evaluations = [cluster_ev]
                    filename = f"{clustering_name}_{metric}.csv"
                    loaded = os.path.exists(f"{RESULTS_CLUSTERING}/{filename}")
                    df = pd.DataFrame(evaluations)
                    df.to_csv(f'{RESULTS_CLUSTERING}/{filename}',
                    mode='a+', index=False, header=not loaded)
    print(f"Finished {DIR}")



