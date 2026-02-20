from __future__ import annotations
import gc

import os
import pandas as pd
from tqdm import tqdm
from utils import load_dataframe, config_bloom_filter, \
    config_bloom_filter_with_configuration
import cupy as cp
from pathlib import Path

# Enable managed memory to allow oversubscribing GPU memory
import pickle
from pprl.clustering import UniqueMappingClustering, \
    ConnectedComponentsClustering, KiralyMSMApproximateClustering
from pprl.matching_gpu import Matcher


if __name__ == '__main__':

    DIR = "D10"
    PATH = f"data/ccer/{DIR}"
    FILE = 'imdbclean'
    FILE2 = 'dbpediaclean'
    attributes = ['title', 'starring']
    SEP = "|"

    RESULTS = f"results/matching/{DIR}"
    RESULTS_CLUSTERING = f"results/clustering/{DIR}"

    loaded_dataframe = load_dataframe(PATH, (FILE, FILE2), SEP)
    BLOOM_NAME = 'matching'
    print(f"Running MATCHING & CLUSTERING for {DIR}")


    size = 1024 // 2
    qgrams = 4
    TOTAL_LENGTH = 32
    DONE = 7

    for num_hashes in tqdm(range(1, 33), total=32):
        if num_hashes < DONE:
            continue

        configuration = {'attributes': attributes,
            'size': size,
            'offset': 0,
            'num_hashes': num_hashes,
            'hashing_type': 'salted_qgrams',
            'salt': '',
            'qgrams': qgrams
        }

        Path(f".graphs/bloom_{configuration['num_hashes']}").mkdir(parents=True,
                                                            exist_ok=True)
        new_bloom = config_bloom_filter_with_configuration(configuration,
                    BLOOM_NAME, loaded_dataframe)

        graph_path = f".graphs/bloom_{num_hashes}"
        for metric in ['cosine', 'dice', 'jaccard']:
            Path(
                f".graphs/bloom_{configuration['num_hashes']}/{metric}"
                ).mkdir(parents=True, exist_ok=True)
            graph = None
            for similarity in tqdm(range(5, 100, 5), total=19):
                graph_index = similarity // 5
                graph_file_name = f'{graph_path}/{metric}/graph_{graph_index}.pkl'
                thr = similarity / 100
                if graph is None:

                    matcher = Matcher(batch_size=1_000_000, threshold=thr, metric=metric)
                    graph = matcher.predict(new_bloom)
                    edges_gpu, weights_gpu = graph
                    edges = cp.asnumpy(edges_gpu)
                    weights = cp.asnumpy(weights_gpu)

                    graph = edges, weights

                    del edges_gpu, weights_gpu
                    cp.get_default_memory_pool().free_all_blocks()

                    dict_to_pickle = {
                        'edges': edges,
                        'weights': weights,
                        'sim': thr,
                        'metric': metric,
                        'num_hashes': configuration['num_hashes']
                    }
                    graph_index = similarity//5

                    with open(f"{graph_file_name}", "wb+") as f:
                        pickle.dump(dict_to_pickle, f)
                    del dict_to_pickle
                    gc.collect()

                else:
                    edges, weights = graph
                    mask = thr <= weights
                    del graph
                    gc.collect()

                    new_edges = edges[mask]
                    new_weights = weights[mask]

                    del edges, weights, mask
                    gc.collect()
                    edges = new_edges
                    graph = edges, new_weights
                    matcher.matcher_config.threshold = thr
                ev = matcher.evaluate(graph, verbose=False)
                matcher_ev = {
                    'name': f'Matching with {metric}',
                    'total_matches': edges.shape[0],
                    'similarity_threshold': thr}
                for keys, items in ev.items():
                    matcher_ev[keys] = round(items, 2) if isinstance(items, float) else items




                matcher_ev['config'] = configuration
                filename = f"{metric}.csv"
                loaded = os.path.exists(f"{RESULTS}/{filename}")
                df = pd.DataFrame([matcher_ev])
                df.to_csv(f'{RESULTS}/{filename}',
                mode='a+', index=False, header=not loaded)
            del graph
            gc.collect()
        del new_bloom
        gc.collect()

    print(f"Finished {DIR}")



