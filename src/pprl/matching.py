from typing import Literal, List, Callable, Dict
from src.pprl.utils import _dice, _scm
from src.pprl.datamodel import Data, Block
from pybloom_live import BloomFilter as Bloom
from networkx import Graph
from ordered_set import OrderedSet
import numpy as np

class Matcher: 
    
    def __init__(self,
        batch_size : int = 10_000,
        threshold : float = 0.6,
        metric : Literal["dice", "scm"] = "dice", 
        attributes : List[str] = None,
    ) -> None:
        metrics_dict = {"dice" : _dice, 
                        "scm" : _scm}
        
        self.batch_size : int = batch_size 
        self.threshold : float = threshold
        self._metric : Callable = metrics_dict[metric]
        self.attributes : List[str] = attributes
        
    def predict(self, data: Data, 
                bloom_filters : Dict[int, Dict[str, Bloom]],
                blocks: Dict[str, Block] = None)  -> Graph:
        """
        Vectorized  similarity computation for Bloom filters.

        np.ndarray
            Array of shape (n_candidates, 3) with columns [eid1_idx, eid2_idx, similarity]
        """
        candidates = set()
        unique_ids = OrderedSet()

        if blocks: 
            for _, block in blocks.items():
                for eid1 in block.entities_D1: 
                    for eid2 in block.entities_D2: 
                        candidates.add((eid1,eid2))
                        unique_ids.add(eid1)
                        unique_ids.add(eid2)
        else: 
            print("EDW")
            data_limit = data.dataset_limit
            print(data_limit)
            end_of_dict = len(bloom_filters) 
            print(end_of_dict - data_limit)
            for i in range(data_limit):
                for j in range(data_limit, end_of_dict):
                    print(j)
                    candidates.add((i, j))
                    unique_ids.add(i)
                    unique_ids.add(j)

        print(list(candidates))
        attributes = self.attributes            

                   
        if not candidates:
            return np.array([]).reshape(0, 3)
        
        for _, bloom_dicts in bloom_filters.items():
            
            if attributes == None: 
                attributes = list(bloom_dicts.keys())
            
            for _, bloom in bloom_dicts.items():
                bf_length = len(bloom.bitarray)
                break
            break

        n_attributes = len(attributes)

        id_to_idx = {eid: idx for idx, eid in enumerate(unique_ids)}
        idx_to_id = {idx: eid for idx, eid in enumerate(unique_ids)}
        
        
        # Create matrices: (n_entities, n_attributes, bf_length)
        bloom_matrix = np.zeros((len(unique_ids), n_attributes, bf_length), dtype=np.uint8)
        # Fill the bloom matrix
        for eid_idx, eid in enumerate(unique_ids):
            for attr_idx, attr in enumerate(attributes):
                bitarr = bloom_filters[eid][attr].bitarray
                bloom_matrix[eid_idx, attr_idx] = np.array(bitarr.tolist(), dtype=np.uint8)
        
        # Prepare candidate indices
        idx_pairs = np.array([(id_to_idx[eid1], id_to_idx[eid2]) 
                            for eid1, eid2 in candidates])
        n_candidates = len(idx_pairs)

        graph = Graph()
        cnt = 0
        for start in range(0, n_candidates, self.batch_size):
            cnt += 1
            end = min(start + self.batch_size, n_candidates)
            batch_pairs = idx_pairs[start:end]
            blooms_1 = bloom_matrix[batch_pairs[:,0]]
            blooms_2 = bloom_matrix[batch_pairs[:,1]]
            print(idx_to_id[7], idx_to_id[4])
            avg_similarities : np.array = self._metric(blooms_1, blooms_2)
            edges = [(idx_to_id[i], idx_to_id[j], float(sim))
                    for (i,j), sim in zip(batch_pairs, avg_similarities)
                    if sim >= self.threshold]
            graph.add_weighted_edges_from(edges)

        return graph
        
    