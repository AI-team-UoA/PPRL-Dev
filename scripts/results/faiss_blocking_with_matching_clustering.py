"""FAISS Results """

import os

from typing import Literal, List, Dict
import pandas as pd
from tqdm import tqdm
from utils import load_dataframe, config_bloom_filter
from privjedai.blocking import FAISSBlocking


def evaluate_faiss(f_mode,
                   f_blocks, f_k, f_ev,
                   f_configuration, f_results,
                   f_m_value=32) -> None:
    candidate_pairs = 0
    for val in f_blocks.values():
        candidate_pairs += len(val)
    if f_mode == 'hnsw':
        faiss_evaluation: Dict[str, float | str | int] = {
            'name': f'FAISS {f_mode.upper()} Blocking',
            'candidate_pairs': candidate_pairs,
            'M': f_m_value,
            'top_k': f_k, }
    else:
        faiss_evaluation: Dict[str, float | str | int] = {
            'name': f'FAISS {f_mode.upper()} Blocking',
            'candidate_pairs': candidate_pairs,
            'top_k': f_k}

    for keys, items in f_ev.items():
        faiss_evaluation[keys] = round(items, 2) if isinstance(items, float) else items
    faiss_evaluation['config'] = f_configuration

    filename = f"faiss_{f_mode}.csv"
    df = pd.DataFrame([faiss_evaluation])
    loaded = os.path.exists(f"{f_results}/{filename}")
    df.to_csv(f'{f_results}/{filename}', mode='a+', index=False, header=not loaded)


if __name__ == '__main__':

    DIR = "D8"
    PATH = f"data/ccer/{DIR}"
    FILE = 'walmartclean'
    FILE2 = 'amazonclean'
    attributes = [['title']]
    SEP = "|"

    VALID_FAISS_MDOES: List[Literal['flat', 'hnsw', 'multihash']] = ['flat', 'hnsw', 'multihash']

    RESULTS = f"results/faiss/{DIR}"
    loaded_dataframe = load_dataframe(PATH, (FILE, FILE2), SEP)
    BLOOM_NAME = 'faiss'
    print(f"Running FAISS for {DIR}")

    size = [64, 128, 256, 512, 1024]
    qgrams = [2, 3, 4]
    TOTAL_LENGTH = 266

    for configuration, new_bloom in tqdm(
            config_bloom_filter(
                attributes,
                size,
                qgrams,
                BLOOM_NAME,
                loaded_dataframe,
                {}),
            desc="Blooms Generator",
            position=0,
            leave=True,
            total=TOTAL_LENGTH):

        for faiss_mode in VALID_FAISS_MDOES:
            if faiss_mode == 'hnsw':
                for m_value in [32, 64]:
                    faiss_blocking = FAISSBlocking(faiss_mode)
                    faiss_blocking.configure_hsnw(m_value)
                    for k in tqdm(range(1, 11, 1),
                                  desc=f"{faiss_mode.upper()} Top-K",
                                  position=1,
                                  leave=False,
                                  total=10):
                        blocks = faiss_blocking.build_blocks(new_bloom, top_k=k)
                        ev: dict = faiss_blocking.evaluate(blocks, verbose=False)
                        evaluate_faiss(faiss_mode, blocks, k, ev, configuration, RESULTS, m_value)
            else:
                faiss_blocking = FAISSBlocking(faiss_mode)
                for k in tqdm(range(1, 11, 1),
                              desc=f"{faiss_mode.upper()} Top-K",
                              position=1,
                              leave=False,
                              total=10):
                    blocks = faiss_blocking.build_blocks(new_bloom, top_k=k)
                    ev: dict = faiss_blocking.evaluate(blocks, verbose=False)
                    evaluate_faiss(faiss_mode, blocks, k, ev, configuration, RESULTS)
