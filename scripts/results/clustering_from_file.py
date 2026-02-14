from __future__ import annotations
import os
import pandas as pd
from tqdm import tqdm
from utils import load_dataframe, config_bloom_filter, \
    config_bloom_filter_with_configuration
import cupy as cp
# Enable managed memory to allow oversubscribing GPU memory
import pickle
from pprl.clustering import UniqueMappingClustering, \
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

    num_hashes = 2
    configuration = {'attributes': attributes[0],
        'size': size,
        'offset': 0,
        'num_hashes': num_hashes,
        'hashing_type': 'salted_qgrams',
        'salt': '',
        'qgrams': qgrams
    }

    new_bloom = config_bloom_filter_with_configuration(configuration,
                        BLOOM_NAME, loaded_dataframe)

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

                edges = graph['edges']
                weights = cp.asnumpy(graph['weights'])

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



