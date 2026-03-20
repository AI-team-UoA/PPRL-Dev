"""FAISS Results """


import os

from pathlib import Path
import pandas as pd
from tqdm import tqdm
from privjedai.gpu.clustering import ConnectedComponentsClustering, KiralyMSMApproximateClustering, UniqueMappingClustering
from privjedai.gpu.matching import Matcher
from utils import config_bloom_filter_with_configuration, load_dataframe, config_bloom_filter
from privjedai.blocking import FAISSBlocking

if __name__ == '__main__':


    DIR = "D10"
    PATH = f"data/ccer/{DIR}"
    FILE = 'imdbclean'
    FILE2 = 'dbpediaclean'
    attributes = ['title','starring']
    SEP = "|"

    RESULTS = f"results/faiss/{DIR}"
    loaded_dataframe = load_dataframe(PATH, (FILE, FILE2), SEP)
    BLOOM_NAME = 'faiss'
    print(f"Running FAISS for {DIR}")

    for faiss_mode in ['flat', 'hnsw', 'lsh']:
        Path(
            f"{RESULTS}/matching/{faiss_mode}"
        ).mkdir(parents=True, exist_ok=True)
        Path(
            f"{RESULTS}/clustering/{faiss_mode}"
        ).mkdir(parents=True, exist_ok=True)

    size = 1024 // 2
    qgrams = 4
    TOTAL_LENGTH = 32
    kiraly = KiralyMSMApproximateClustering()
    connected = ConnectedComponentsClustering()
    unique = UniqueMappingClustering()


    for num_hashes in tqdm(range(1, 33), total=TOTAL_LENGTH,
                position=0, desc="Total Progress", leave=True):
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

        for faiss_mode in ['flat', 'hnsw', 'lsh']:
            faiss_blocking = FAISSBlocking(faiss_mode)
            matcher = None
            for k in tqdm(range(10), desc=f"{faiss_mode.upper()} Top-K",
                          position=1, leave=False,
                          total=10):
                blocks = faiss_blocking.build_blocks(new_bloom, top_k=k+1)
                for metric in ['cosine', 'dice', 'jaccard']:
                    for similarity in tqdm(
                        range(5,100,5), leave=False, position=2,
                        desc=f"Matching Similarity {metric}", total=19):
                        thr = similarity / 100
                        if matcher is None:
                            matcher = Matcher(batch_size=1_000_000,
                                threshold=thr, metric=metric)
                            graph = matcher.predict(new_bloom, blocks)
                        else:
                            graph = matcher.update_graph(graph, thr)
                        ev = matcher.evaluate(graph, verbose=False)
                        edges, _ = graph
                        matcher_ev = {
                            'name': f'Matching with {metric}',
                            'top_k': k+1,
                            'total_matches': edges.shape[0],
                            'similarity_threshold': thr}

                        for keys, items in ev.items():
                            matcher_ev[keys] = round(items, 2) if isinstance(items, float) else items

                        matcher_ev['config'] = configuration
                        matcher_filename = f"{metric}.csv"
                        loaded = os.path.exists(f"{RESULTS}/matching/{faiss_mode}/{matcher_filename}")
                        df = pd.DataFrame([matcher_ev])
                        df.to_csv(f'{RESULTS}/matching/{faiss_mode}/{matcher_filename}', mode='a+', index=False,
                            header=not loaded)
                        for clustering in [kiraly, connected, unique]:
                            clustering_name = clustering.__class__.__name__
                            clustering_filename = f"{clustering_name}_{metric}.csv"
                            clusters = clustering.process(graph, new_bloom, thr)
                            ev_cl = clustering.evaluate(clusters, verbose=False)
                            cluster_ev = {
                                'name': f'Clustering with {clustering_name}',
                                'total_clusters': len(clusters),
                                'threshold': thr,
                                'matcher': metric,
                                'top_k': k+1
                            }
                            for keys, items in ev_cl.items():
                                cluster_ev[keys] = round(items, 2) if isinstance(items, float) else items
                            cluster_ev['config'] = configuration
                            loaded = os.path.exists(f"{RESULTS}/clustering/{faiss_mode}/{clustering_filename}")
                            df = pd.DataFrame([cluster_ev])
                            df.to_csv(f'{RESULTS}/clustering/{faiss_mode}/{clustering_filename}', mode='a+', index=False,
                                header=not loaded)
