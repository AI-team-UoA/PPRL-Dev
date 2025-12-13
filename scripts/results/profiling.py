"""Profiling to produce faster Results"""

import ast
import os

import pandas as pd
from tqdm import tqdm
import ray

from utils import load_dataframe, config_bloom_filter


if __name__ == '__main__':
    DIR = "D2"
    PATH = f"data/ccer/{DIR}"
    FILE = 'abtclean'
    FILE2 = 'buyclean'
    SEP = '|'


    DIR = f"results/metablocking/{DIR}"


    loaded_dataframe = load_dataframe(PATH, (FILE, FILE2), SEP)



    attributes = [['name']]
    size = [64, 128, 256, 512, 1024]
    qgrams = [2,3,4]

    TOTAL_LENGTH = len(attributes) * len(size) * len(qgrams)
    for configuration, new_bloom in tqdm(
        config_bloom_filter(attributes, size, qgrams, 'metablocking', loaded_dataframe),
        total=TOTAL_LENGTH):
        x = new_bloom.to_entities()
        exit(10)
