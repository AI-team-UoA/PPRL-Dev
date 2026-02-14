"""Matching & Clustering Results """
from __future__ import annotations
import ast
import os
import pandas as pd
from tqdm import tqdm
from utils import load_dataframe, config_bloom_filter
import rmm
rmm.reinitialize(pool_allocator=False)
from pathlib import Path
import pickle

from pprl.matching_gpu import Matcher
from pprl.clustering_gpu import UniqueMappingClustering, \
    ConnectedComponentsClustering, KiralyMSMApproximateClustering


if __name__ == '__main__':


    DIR = "D10"
    PATH = f"data/ccer/{DIR}"
    FILE = 'imdbclean'
    FILE2 = 'dbpediaclean'
    attributes = [['title', 'starring']]
    SEP = "|"

    # # DIR = "D10"
    # PATH = f"data/ccer/{DIR}"
    # FILE = 'dblpclean'
    # FILE2 = 'scholarclean'
    # attributes = [['title']]
    # SEP = ">"


    # DIR = "D9"
    # PATH = f"data/ccer/{DIR}"
    # FILE = 'dblpclean'
    # FILE2 = 'scholarclean'
    # attributes = [['title']]
    # SEP = ">"

    RESULTS = f"results/matching/{DIR}"
    RESULTS_CLUSTERING = f"results/clustering/{DIR}"

    loaded_dataframe = load_dataframe(PATH, (FILE, FILE2), SEP)
    BLOOM_NAME = 'matching'
    print(f"Running MATCHING & CLUSTERING for {DIR}")


    size = [1024 // 2]
    qgrams = [4]
    TOTAL_LENGTH = 32
    # futures = []
    # MAX_CONCURRENCY = 24
    # CLUSTER_MAX_CONCURRENCY = 5


    adjacent_config = {}

    matched_done = set()
    clustered_done = set()
    skip = set()
    # for metric in ['dice', 'jaccard', 'cosine']:
    #     matching_df = pd.read_csv(f"{RESULTS}/{metric}.csv")
    #     matching_df['config'] = matching_df['config'].apply(ast.literal_eval)
    #     matching_df['metric'] = metric
    #     matching_df['num_hashes'] = matching_df['config'].apply(lambda x: x['num_hashes'])
    #     list_done = set(zip(
    #         matching_df['metric'],
    #         matching_df['num_hashes'],
    #         matching_df['similarity_threshold']))
    #     matched_done.update(list_done)
    #     max_num_hashes = matching_df['num_hashes'].max()


    #     def clustering_done_list(clustering_method: str):
    #         filename = f"{clustering_method}_{metric}.csv"
    #         if not os.path.exists(f"{RESULTS_CLUSTERING}/{filename}"):
    #             return False
    #         clustering_df = pd.read_csv(f"{RESULTS_CLUSTERING}/"
    #             f"{clustering_method}_{metric}.csv")
    #         clustering_df['config'] = clustering_df['config'].apply(
    #             ast.literal_eval)
    #         clustering_df['num_hashes'] = clustering_df['config'].apply(
    #             lambda x: x['num_hashes'])
    #         clustering_df['method'] = clustering_method
    #         list_done_cl = set(zip(
    #                 clustering_df['method'],
    #                 clustering_df['matcher'],
    #                 clustering_df['num_hashes'],
    #                 clustering_df['matching_thr'],
    #                 clustering_df['clustering_threshold']))
    #         return list_done_cl


    #     for hashing in range(1, max_num_hashes + 1):
    #         flag = True
    #         for similarity in range(5, 100, 5):
    #             if flag == False:
    #                 break
    #             if (metric, hashing, similarity / 100) not in matched_done:
    #                 flag = False
    #                 break
    #             for cl_method in [
    #                 'ConnectedComponentsClustering',
    #                 'UniqueMappingClustering',
    #                 'KiralyMSMApproximateClustering']:

    #                 cl_done = clustering_done_list(cl_method)
    #                 clustered_done.update(cl_done if cl_done else set())
    #                 if flag == False:
    #                     continue
    #                 if cl_done:
    #                     for cl_similarity in range(10, 100, 10):
    #                         if (cl_method, metric, hashing,
    #                             similarity / 100,
    #                             cl_similarity / 100) not in cl_done:
    #                             flag = False
    #                             break
    #                 else:
    #                     flag = False
    #                     break
    #         if flag:
    #             skip.add((metric, hashing))



    # if not ray.is_initialized():
    #     logging.getLogger("ray").setLevel(logging.ERROR)
    #     ray.init(log_to_driver=False, logging_level=logging.ERROR,
    #             local_mode=False)

    for configuration, new_bloom in tqdm(
        config_bloom_filter(attributes, size, qgrams,
                BLOOM_NAME, loaded_dataframe, adjacent_config),
        total=TOTAL_LENGTH):

        Path(f".graphs/bloom_{configuration['num_hashes']}").mkdir(parents=True,
                                                            exist_ok=True)

        # new_bloom_ref = ray.put(new_bloom)
        for metric in ['dice', 'jaccard', 'cosine']:
            Path(
                f".graphs/bloom_{configuration['num_hashes']}/{metric}"
                ).mkdir(parents=True, exist_ok=True)

            graph_path = f".graphs/bloom_{configuration['num_hashes']}/{metric}"

            if (metric, configuration['num_hashes']) in skip:
                continue
            print(f"Running for {metric}")
            graph = None
            for similarity in tqdm(range(5,100,5), desc="Matching Similarity", total=10):
                thr = similarity / 100
                if graph is None:
                    matcher = Matcher(batch_size=1_000_000, threshold=thr, metric=metric)
                    graph = matcher.predict(new_bloom)
                else:
                    edges, weights = graph
                    mask = thr <= weights
                    edges = edges[mask.get()]
                    weights = weights[mask.get()]
                    graph = edges, weights
                    matcher.matcher_config.threshold = thr

                if (metric, configuration['num_hashes'], similarity / 100) not in matched_done:

                    edges, weights = graph
                    dict_to_pickle = {
                        'edges': edges,
                        'weights': weights,
                        'sim': thr,
                        'metric': metric,
                        'num_hashes': configuration['num_hashes']
                    }
                    graph_index = similarity//5

                    with open(f"{graph_path}/graph_{graph_index}.pkl", "wb+") as f:
                        pickle.dump(dict_to_pickle, f)


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




                # for cl in [KiralyMSMApproximateClustering, ConnectedComponentsClustering,
                #         UniqueMappingClustering
                #         ]:
                #     clustering_method = cl()
                #     clusters = clustering_method.process(graph, new_bloom, thr)
                #     ev_cl = clustering_method.evaluate(clusters, verbose=False)
                #     cluster_ev = {
                #         'name': f'Clustering with {cl.__name__}',
                #         'total_clusters': len(clusters),
                #         'threshold': thr,
                #         'matcher': metric,
                #     }

                #     for keys, items in ev_cl.items():
                #         cluster_ev[keys] = round(items, 2) \
                #         if isinstance(items, float) else items

                #     cluster_ev['config'] = configuration
                #     clustering_name = cl.__name__
                #     evaluations = [cluster_ev]
                #     filename = f"{clustering_name}_{metric}.csv"
                #     loaded = os.path.exists(f"{RESULTS_CLUSTERING}/{filename}")
                #     df = pd.DataFrame(evaluations)
                #     df.to_csv(f'{RESULTS_CLUSTERING}/{filename}',
                #     mode='a+', index=False, header=not loaded)
    print(f"Finished {DIR}")
