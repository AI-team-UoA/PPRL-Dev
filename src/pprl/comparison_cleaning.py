"""PPRL Metablocking Methods"""
import sys
from queue import PriorityQueue
from itertools import chain
from collections import defaultdict
from math import log10, sqrt
from time import time
from abc import ABC, abstractmethod
from typing import Dict
from scipy.sparse import csr_matrix
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from pprl.evaluation import Evaluation
from pprl.datamodel import PPRLFeature
from pprl.encoded_data import BloomEncodedData
from pprl.utils import chi_square



class AbstractComparisonCleaning(PPRLFeature):
    """Abstract class for Block cleaning
    """

    def __init__(self) -> None:
        super().__init__()
        self._valid_entities: set = set()
        self.encoded_data : BloomEncodedData
        self.entity_id_block_key_pairs : list
        # self._inverted_index : dict
        self.candidate_pairs : dict

    def process(
            self,
            encoded_data: BloomEncodedData,
            adjacent_bits : int = 1
    ) -> dict:
        """Main method for comparison cleaning

        Args:
            encoded_data (BloomEncodedData): bloom filters created from previous steps of pprl

        Returns:
            dict: cleaned blocks
        """
        start_time = time()
        self.encoded_data = encoded_data

        self.candidate_pairs : Dict[int, set] = defaultdict(set)

        self.entity_ids, self.block_keys, = encoded_data.get_entity_id_block_key_pairs(adjacent_bits)

        entity_indices = self.entity_ids

        num_of_entities = len(self.encoded_data.encoded_dict)
        limit = self.encoded_data.bounds[0]
        entity_indices = self.entity_ids

        _, block_indices = np.unique(self.block_keys,
                            return_inverse=True)

        num_blocks = block_indices.max() + 1 \
            if len(block_indices) > 0 else 0


        data = np.ones(entity_indices.shape[0], dtype=bool)
        M = csr_matrix(
            (data, (entity_indices, block_indices)),
            shape=(num_of_entities, num_blocks),
            dtype=bool
        )

        self.M1 = M[:limit, :]
        self.M2 = M[limit:, :]

        self.candidate_pairs = self._apply_main_processing()

        self.execution_time = time() - start_time

        return self.candidate_pairs

    def report(self) -> None:
        """Prints Block Building method configuration
        """
        _conf = self._configuration()
        print(
            "Method name: " + self._method_name +
            "\nMethod info: " + self._method_info +
            ("\nParameters: \n" + ''.join([f'\t{k}: {v}\n' for k, v in _conf.items()])
            if _conf.items() else "\nParameters: Parameter-Free method\n") +
            "Attributes:\n\t" + ', '.join(c for c in self.encoded_data.attributes) +
            f"\nRuntime: {self.execution_time:2.4f} seconds"
        )

    @abstractmethod
    def _apply_main_processing(self) -> dict:
        pass

    @abstractmethod
    def _configuration(self) -> dict:
        pass

    def evaluate(self,
                 prediction : dict,
                 export_to_df: bool = False,
                 with_classification_report: bool = False,
                 verbose: bool = True) -> any:
        """Function to evaluate meta-blocking methods f1-score, recall and precision

        Args:
            prediction (dict) : Blocks predicted from the blocking method
            export_to_df (bool) : Create evaluation dataframe
            with_classification_report (bool) : Printing Info for the blocking method
            with_stats (bool) : Printing Method's Statistics

        Returns:
            Evaluation : Evaluation Object with F1, Recall, Precision etc.
        """

        if self.encoded_data is None:
            raise AttributeError("Can not proceed to evaluation without data object.")

        if self.encoded_data.skip_ground_truth:
            raise AttributeError("Can not proceed to evaluation without a ground-truth file."
                        "Data object has not been initialized with the ground-truth file")

        eval_obj = Evaluation(self.encoded_data)
        true_positives = 0
        total_matching_pairs = 0
        for block in prediction.values():
            total_matching_pairs += len(block)



        for _, (id1, id2) in self.encoded_data.ground_truth.iterrows():
            id2 = self.encoded_data.bounds[0] + id2
            if id1 in prediction and id2 in prediction[id1]:
                true_positives += 1

        eval_obj.calculate_scores(true_positives=true_positives,
                                  total_matching_pairs=total_matching_pairs)
        return eval_obj.report(self.method_configuration(),
                                export_to_df,
                                with_classification_report,
                                verbose)

    def export_to_df(self, prediction: dict) -> pd.DataFrame:
        """creates a dataframe with the predicted pairs

        Args:
            prediction (dict): Predicted candidate pairs

        Returns:
            pd.DataFrame: Dataframe with the predicted pairs
        """
        pairs_list = []
        for id1, candidates in prediction.items():
            for candiadate_id in candidates:
                id2 = candiadate_id - self.encoded_data.bounds[0]
                pairs_list.append({'id1': id1, 'id2': id2})

        pairs_df = pd.DataFrame(pairs_list) if len(pairs_list) > 0 else pd.DataFrame(
                columns=['id1', 'id2'])

        return pairs_df


class AbstractMetablocking(AbstractComparisonCleaning, ABC):
    """Restructure a redundancy-positive block collection into a new
        one that contains substantially lower number of redundant
        and superfluous comparisons, while maintaining the original number of matching ones
    """


    def __init__(self) -> None:
        super().__init__()
        self._flags: np.array
        self._counters: np.array
        self._comparisons_per_entity: np.array
        self._distinct_comparisons: int
        self._neighbors: set = set()
        self.weighting_scheme: str

    def _apply_main_processing(self) -> dict:
        self._counters = np.empty([self.encoded_data.bounds[1]], dtype=float)
        self._flags = np.empty([self.encoded_data.bounds[1]], dtype=int)
        self._set_statistics()
        self._set_threshold()

        return self._prune_edges()

    def _get_weight(self, entity_id: int, neighbor_id: int) -> float:
        ws = self.weighting_scheme
        _num_of_bits = len(self._inverted_index)
        weight_dict = {
            'CN-CBS': lambda: self._counters[neighbor_id],
            'CBS': lambda: self._counters[neighbor_id],
            'SN-CBS': lambda: self._counters[neighbor_id],
            # CARDINALITY_NORM_COSINE, SIZE_NORM_COSINE
            'CNC': lambda: self._counters[neighbor_id] /
                sqrt(self._comparisons_per_entity[entity_id] *
                    self._comparisons_per_entity[neighbor_id]),
            'SNC': lambda: self._counters[neighbor_id] /
                sqrt(self._comparisons_per_entity[entity_id] *
                    self._comparisons_per_entity[neighbor_id]),
            # SIZE_NORM_DICE, CARDINALITY_NORM_DICE
            'SND' : lambda: 2 *
                self._counters[neighbor_id] / (
                self._comparisons_per_entity[entity_id] +
                self._comparisons_per_entity[neighbor_id] ) ,
            'CND' : lambda: 2 *
                self._counters[neighbor_id] / (
                self._comparisons_per_entity[entity_id] +
                self._comparisons_per_entity[neighbor_id] ),
            # CARDINALITY_NORM_JS, SIZE_NORM_JS
            'CNJ' : lambda : self._counters[neighbor_id] / (
                self._comparisons_per_entity[entity_id] +
                self._comparisons_per_entity[neighbor_id] -
                self._counters[neighbor_id] ),
            'SNJ' : lambda : self._counters[neighbor_id] / (
                self._comparisons_per_entity[entity_id] +
                self._comparisons_per_entity[neighbor_id] -
                self._counters[neighbor_id] ),
            'COSINE': lambda :  self._counters[neighbor_id] /
                (sqrt(len(self._entity_index[entity_id]) * len(self._entity_index[neighbor_id]))),
            'DICE': lambda : 2 * self._counters[neighbor_id] /
                (len(self._entity_index[entity_id]) + len(self._entity_index[neighbor_id])),
            'ECBS': lambda : float( self._counters[neighbor_id] *
                log10((_num_of_bits / len(self._entity_index[entity_id]))) *
                log10((_num_of_bits / len(self._entity_index[neighbor_id])))
            ),
            'JS': lambda : self._counters[neighbor_id] / (len(self._entity_index[entity_id]) + \
                len(self._entity_index[neighbor_id]) - self._counters[neighbor_id]),
            'EJS': lambda : self._ejs(entity_id, neighbor_id),
            'X2': lambda : self._x2(entity_id, neighbor_id, _num_of_bits)
        }
        if ws not in weight_dict:
            raise ValueError("This weighting scheme does not exist")

        return weight_dict[ws]()

    def _ejs(self, entity_id, neighbor_id):
        probability = self._counters[neighbor_id] / (len(self._entity_index[entity_id]) + \
                        len(self._entity_index[neighbor_id]) - self._counters[neighbor_id])
        return float(probability * \
                log10(self._distinct_comparisons / self._comparisons_per_entity[entity_id]) * \
                log10(self._distinct_comparisons / self._comparisons_per_entity[neighbor_id]))

    def _x2(self, entity_id, neighbor_id, _num_of_bits):
        observed = [int(self._counters[neighbor_id]),
            int(len(self._entity_index[entity_id])-self._counters[neighbor_id])]
        expected = [int(len(self._entity_index[neighbor_id])-observed[0]),
            int(_num_of_bits - (observed[0] + observed[1] - self._counters[neighbor_id]))]
        return chi_square(np.array([observed, expected]))

    def _normalize_neighbor_entities(self, bit_pos: int, entity_id: int) -> None:
        self._neighbors.clear()
        _limit = self.encoded_data.bounds[0]
        if entity_id < _limit:
            self._neighbors.update(self._inverted_index[bit_pos][1])
        else:
            self._neighbors.update(self._inverted_index[bit_pos][0])


    def _get_vectorized_weights(self, C) -> np.array:
        _limit = self.encoded_data.bounds[0]

        c_data = C.data
        rows = C.row
        cols = C.col

        blocks_1 = self.M1.getnnz(axis=1)
        blocks_2 = self.M2.getnnz(axis=1)


        b1 = blocks_1[rows]
        b2 = blocks_2[cols]
        if self.weighting_scheme in ['SNC' , 'CNC', 'CND', 'SND',
                            'CNJ', 'SNJ', 'EJS']:
            comp_1 = self._comparisons_per_entity[:_limit]
            comp_2 = self._comparisons_per_entity[_limit:]
            c1 = comp_1[rows]
            c2 = comp_2[cols]

        _num_of_bits = self.M1.shape[1]
        weighting_dict = {
            'CN-CBS': lambda : c_data,
            'CBS': lambda : c_data,
            'SN-CBS': lambda : c_data,
            'SNC': lambda : c_data / np.sqrt(c1 * c2),
            'CNC': lambda : c_data / np.sqrt(c1 * c2),
            'SND' : lambda : 2 * c_data / (c1 + c2) ,
            'CND' : lambda : 2 * c_data / (c1 + c2) ,
            'CNJ' : lambda : c_data / (c1 + c2 - c_data) ,
            'SNJ' : lambda : c_data / (c1 + c2 - c_data) ,
            'COSINE': lambda :  c_data / (np.sqrt(b1 * b2)),
            'DICE': lambda : 2 * c_data / (b1 + b2),
            'ECBS': lambda : c_data * np.log10(_num_of_bits / b1) * \
                np.log10(_num_of_bits / b2),
            'JS': lambda : c_data / (b1 + b2 - c_data),
            'EJS': lambda : (c_data / (b1 + b2 - c_data)) * \
                np.log10(self._distinct_comparisons / c1) * \
                np.log10(self._distinct_comparisons / c2),
            'X2': lambda : np.divide(
                float(_num_of_bits) * (c_data * float(_num_of_bits) - b1 * b2) ** 2,
                b1 * b2 * (float(_num_of_bits) - b1) * (float(_num_of_bits) - b2),
                out=np.zeros_like(c_data, dtype=float),
                where=(b1 * b2 * (float(_num_of_bits) - b1) * (float(_num_of_bits) - b2))!=0
            )
        }
        if self.weighting_scheme not in weighting_dict:
            raise ValueError("This weighting scheme does not exist")
        return weighting_dict[self.weighting_scheme]()

    def _set_statistics(self) -> None:
        if self.weighting_scheme not in {'EJS',
                'CNC', 'SNC', 'SND', 'CND', 'CNJ', 'SNJ'} :
            return
        M1 = self.M1
        M2 = self.M2

        C = M1.dot(M2.T)
        comp_e1 = C.getnnz(axis=1)
        comp_e2 = C.getnnz(axis=0)
        self._comparisons_per_entity = np.concatenate([comp_e1, comp_e2]).astype(float)
        self._distinct_comparisons = float(C.nnz)

    def _get_neighbor_entities(self, block_id: int, entity_id: int) -> set:
        _limit = self.encoded_data.bounds[1]
        return self._inverted_index[block_id][1] \
            if (entity_id < _limit) else \
                self._inverted_index[block_id][0]

    def _get_bit_assignmets(self) -> int:
        bit_assignments = 0
        for datasets_dict in self._inverted_index.values():
            bit_assignments += len(datasets_dict[0]) + len(datasets_dict[1])
        return bit_assignments

    @abstractmethod
    def _set_threshold(self):
        pass

    @abstractmethod
    def _prune_edges(self) -> dict:
        pass


class WeightedEdgePruning(AbstractMetablocking):
    """A Meta-blocking method that retains all comparisons \
        that have a weight higher than the average edge weight in the blocking graph.
    """

    _method_name = "Weighted Edge Pruning"
    _method_short_name: str = "WEP"
    _method_info = "A Meta-blocking method that retains all comparisons " + \
                "that have a weight higher than the average edge weight in the blocking graph."

    def __init__(self, weighting_scheme: str = 'CBS') -> None:
        super().__init__()
        self.weighting_scheme = weighting_scheme
        self._node_centric = False
        self._num_of_edges: float
        self._threshold: float
        self._retained_neighbors: set = set()

    def _prune_edges(self) -> dict:

        _limit = self.encoded_data.bounds[0] if not self._node_centric \
            else len(self.encoded_data.encoded_dict)
        invalid_mask = self._sim_csr.data < self._threshold

        self._sim_csr.data[invalid_mask] = 0.0
        self._sim_csr.eliminate_zeros()

        indptr = self._sim_csr.indptr
        indices = self._sim_csr.indices
        for entity_id in range(_limit):
            start_idx = indptr[entity_id]
            end_idx = indptr[entity_id + 1]
            if start_idx < end_idx:
                valid_neighbors = indices[start_idx:end_idx] + _limit
                self.candidate_pairs[entity_id].update(valid_neighbors)

        return self.candidate_pairs

    def _process_entity(self, entity_id: int) -> None:
        pass


    def _update_counters(self) -> None:
        s1 = self.M1.sum(axis=0).A1
        s2 = self.M2.sum(axis=0).A1
        if self.weighting_scheme in {'CN-CBS', 'CNC', 'CND', 'CNJ'}:
            denominator = s1 * s2
            self._block_cardinalites = np.divide(1.0, denominator,
                            out = np.zeros_like(denominator, dtype=float),
                            where = denominator!=0)
        elif self.weighting_scheme in {'SN-CBS', 'SNC', 'SND', 'SNJ'}:
            denominator = s1 + s2
            self._block_cardinalites = np.divide(1.0, denominator,
                            out = np.zeros_like(denominator, dtype=float),
                            where = denominator!=0)
        else:
            self._block_cardinalites = np.ones_like(s1, dtype=float)


    def _set_threshold(self):
        self._num_of_edges = 0.0
        self._threshold = 0.0
        self._update_counters()

        processed_M1 = self.M1.multiply(self._block_cardinalites)
        _sim_matrix = processed_M1.dot(self.M2.T)
        self._num_of_edges = _sim_matrix.nnz

        C = _sim_matrix.tocoo()
        C.data = self._get_vectorized_weights(C)

        self._threshold = C.data.sum() / self._num_of_edges
        self._sim_csr = C.tocsr()


    def _configuration(self) -> dict:
        return {
            "Node centric" : self._node_centric,
            "Weighting scheme" : self.weighting_scheme
        }


class CardinalityEdgePruning(WeightedEdgePruning):
    """A Meta-blocking method that retains the comparisons \
        that correspond to the top-K weighted edges in the bit graph.
    """

    _method_name = "Cardinality Edge Pruning"
    _method_short_name: str = "CEP"
    _method_info = "A Meta-blocking method that retains the comparisons " + \
                        "that correspond to the top-K weighted edges in the bit graph."

    def __init__(self, weighting_scheme: str = 'JS') -> None:
        super().__init__(weighting_scheme)
        self._minimum_weight: float = sys.float_info.min
        self._top_k_edges: PriorityQueue

    def _prune_edges(self) -> dict:
        _limit = self.encoded_data.bounds[0] if not self._node_centric \
            else len(self.encoded_data.encoded_dict)

        self._update_counters()
        processed_M1 = self.M1.multiply(self._block_cardinalites)
        _sim_matrix = processed_M1.dot(self.M2.T)
        self._num_of_edges = _sim_matrix.nnz

        C = _sim_matrix.tocoo()
        C.data = self._get_vectorized_weights(C)
        self._sim_csr = C.tocsr()

        K = int(self._threshold)
        print("K: ", K)
        C = self._sim_csr.tocoo()

        data = C.data
        rows = C.row
        cols = C.col + _limit

        num_edges = len(data)

        if num_edges > K > 0:
            top_k_indices = np.argpartition(data, -K)[-K:]
            top_rows = rows[top_k_indices]
            top_cols = cols[top_k_indices]
        else:
            top_rows = rows
            top_cols = cols

        for entity_id, neighbor_id in zip(top_rows, top_cols):
            self.candidate_pairs[entity_id].add(neighbor_id)

        return self.candidate_pairs

    def _set_threshold(self) -> None:
        self._threshold = self.block_keys.shape[0] / 2


    def _verify_valid_entities(self, entity_id: int) -> None:
        if entity_id not in self._entity_index:
            return

        for neighbor_id in self._valid_entities:
            weight = self._get_weight(entity_id, neighbor_id)
            if weight >= self._minimum_weight:
                self._top_k_edges.put(
                    (weight, entity_id, neighbor_id)
                )
                if self._threshold < self._top_k_edges.qsize():
                    self._minimum_weight = self._top_k_edges.get()[0]

class CardinalityNodePruning(CardinalityEdgePruning):
    """A Meta-blocking method that retains for every entity, \
        the comparisons that correspond to its top-k weighted edges in the blocking graph."
    """

    _method_name = "Cardinality Node Pruning"
    _method_short_name: str = "CNP"
    _method_info = "A Meta-blocking method that retains for every entity, " + \
                    "the comparisons that correspond to its top-k" + \
                    " weighted edges in the blocking graph."
    def __init__(self, weighting_scheme: str = 'CBS') -> None:
        super().__init__(weighting_scheme)
        self._nearest_entities: dict
        self._node_centric = True
        self._top_k_edges: PriorityQueue

    def _prune_edges(self) -> dict:
        _limit = self.encoded_data.bounds[0] if not self._node_centric \
            else len(self.encoded_data.encoded_dict)

        self._update_counters()
        processed_M1 = self.M1.multiply(self._block_cardinalites)
        _sim_matrix = processed_M1.dot(self.M2.T)
        self._num_of_edges = _sim_matrix.nnz

        C = _sim_matrix.tocoo()
        C.data = self._get_vectorized_weights(C)
        self._sim_csr = C.tocsr()


        K = int(self._threshold)
        indptr = self._sim_csr.indptr
        indices = self._sim_csr.indices
        data = self._sim_csr.data
        offset = self.encoded_data.bounds[0]



        sources = []
        targets = []
        for i in range(0, self.encoded_data.bounds[0]):
            start_idx = indptr[i]
            end_idx = indptr[i + 1]
            row_len = end_idx - start_idx

            if row_len == 0:
                continue


            row_data = data[start_idx:end_idx]
            row_indices = indices[start_idx:end_idx]

            if row_len > K:
                top_k_pos = np.argpartition(row_data, -K)[-K:]
                top_neighbors = row_indices[top_k_pos]
            else:
                top_neighbors = row_indices
            sources.append(np.full(len(top_neighbors), i, dtype=int))
            targets.append(top_neighbors + offset)

        if self._node_centric:
            sim_csc = self._sim_csr.tocsc()
            indptr_col = sim_csc.indptr
            indices_col = sim_csc.indices
            data_col = sim_csc.data

            num_d2_entities = self.encoded_data.bounds[1] - self.encoded_data.bounds[0]
            for j in range(num_d2_entities):
                start = indptr_col[j]
                end = indptr_col[j + 1]

                if end > start:
                    col_data = data_col[start:end]
                    col_indices = indices_col[start:end] # These are D1 IDs

                    if (end - start) > K:
                        top_k_pos = np.argpartition(col_data, -K)[-K:]
                        top_neighbors = col_indices[top_k_pos]
                    else:
                        top_neighbors = col_indices

                    # The entity is 'j + limit_d1', voting for 'col_indices'
                    sources.append(np.full(len(top_neighbors), j + offset, dtype=np.int32))
                    targets.append(top_neighbors)


        self.candidate_pairs = defaultdict(set)
        sources = np.concatenate(sources)
        targets = np.concatenate(targets)
        edges = np.column_stack((sources, targets))
        self._nearest_entities = {}

        for src, tgt in edges:
            self._nearest_entities.setdefault(src, set()).add(tgt)

        for src, tgt in edges:
            if self._is_valid_comparison(src, tgt):
                self.candidate_pairs[src].add(tgt)

        return self.candidate_pairs

    def _retain_valid_comparisons(self) -> dict:
        self.candidate_pairs = defaultdict(set)
        num_of_entities = len(self.encoded_data.encoded_dict)
        for i in range(0, num_of_entities):
            if i in self._nearest_entities:
                for neighbor_id in self._nearest_entities[i]:
                    # Comparison is a triple: (id1, id2, weight)
                    if self._is_valid_comparison(i, neighbor_id):
                        self.candidate_pairs[i].add(neighbor_id)
        return self.candidate_pairs

    def _is_valid_comparison(self, entity_id: int, neighbor_id: int) -> bool:
        if neighbor_id not in self._nearest_entities:
            return True
        if entity_id in self._nearest_entities[neighbor_id]:
            return entity_id < neighbor_id
        return True

    def _set_threshold(self) -> None:
        num_of_entities = len(self.encoded_data.encoded_dict)
        bit_assignments = self.block_keys.shape[0]
        self._threshold = max(1, bit_assignments / num_of_entities)

    def _verify_valid_entities(self, entity_id: int) -> None:
        if entity_id not in self._entity_index:
            return
        self._top_k_edges = PriorityQueue(int(2*self._threshold))
        self._minimum_weight = sys.float_info.min
        for neighbor_id in self._valid_entities:
            weight = self._get_weight(entity_id, neighbor_id)
            if weight >= self._minimum_weight:
                self._top_k_edges.put(
                    (weight, entity_id, neighbor_id)
                )
                if self._threshold < self._top_k_edges.qsize():
                    self._minimum_weight = self._top_k_edges.get()[0]
        if self._top_k_edges:
            self._nearest_entities.setdefault(entity_id, set())
        while not self._top_k_edges.empty():
            comparison = self._top_k_edges.get()
            self._nearest_entities[entity_id].add(comparison[2])


class ReciprocalCardinalityNodePruning(CardinalityNodePruning):
    """A Meta-blocking method that retains the comparisons \
        that correspond to edges in the blocking graph that are among the top-k weighted  \
        ones for both adjacent entities/nodes.
    """

    _method_name = "Reciprocal Cardinality Node Pruning"
    _method_short_name: str = "RCNP"
    _method_info = "A Meta-blocking method that retains the comparisons " + \
                    "that correspond to edges in the blocking graph that are among " + \
                    "the top-k weighted ones for both adjacent entities/nodes."

    def __init__(self, weighting_scheme: str = 'CN-CBS') -> None:
        super().__init__(weighting_scheme)

    def _is_valid_comparison(self, entity_id: int, neighbor_id: int) -> bool:
        if neighbor_id not in self._nearest_entities:
            return False
        if entity_id in self._nearest_entities[neighbor_id]:
            return  entity_id < neighbor_id
        return False

class WeightedNodePruning(WeightedEdgePruning):
    """A Meta-blocking method that retains for every entity, the comparisons \
        that correspond to edges in the blocking graph that are exceed \
        the average edge weight in the respective node neighborhood.
    """

    _method_name = "Weighted Node Pruning"
    _method_short_name: str = "WNP"
    _method_info = "A Meta-blocking method that retains for every entity, the comparisons \
                    that correspond to edges in the blocking graph that are exceed \
                    the average edge weight in the respective node neighborhood."

    def __init__(self, weighting_scheme: str = 'CBS') -> None:
        super().__init__(weighting_scheme)
        self._average_weight: np.array
        self._node_centric = True

    def _get_valid_weight(self, entity_id: int, neighbor_id: int) -> float:
        weight = self._get_weight(entity_id, neighbor_id)
        return weight if ((self._average_weight[entity_id] <= weight or \
                             self._average_weight[neighbor_id] <= weight) and
                                entity_id < neighbor_id) else 0


    def _prune_edges(self) -> dict:

        C = self._sim_matrix.tocoo()
        weights = self._get_vectorized_weights(C)

        mask = ((self._average_weight[C.row] <= weights) | \
                (self._average_weight[C.col] <= weights))

        valid_rows = C.row[mask]
        valid_cols = C.col[mask] + self.encoded_data.bounds[0]

        for entity_id, neighbor_id in zip(valid_rows, valid_cols):
            self.candidate_pairs[entity_id].add(neighbor_id)
        return self.candidate_pairs


    def _set_threshold(self):
        _limit = self.encoded_data.bounds[0] if not self._node_centric \
            else len(self.encoded_data.encoded_dict)

        self._average_weight = np.zeros(_limit, dtype=float)

        self._update_counters()

        processed_M1 = self.M1.multiply(self._block_cardinalites)
        self._sim_matrix = processed_M1.dot(self.M2.T)
        self._num_of_edges = self._sim_matrix.nnz

        C = self._sim_matrix.tocoo()
        C.data = self._get_vectorized_weights(C)
        data = C.data
        row = C.row
        for i in range(0, _limit):
            self._average_weight[i] = data[row == i].mean() \
                if np.any(row == i) else 0.0


    def _update_threshold(self, entity_id: int) -> None:
        if entity_id not in self._entity_index:
            return
        self._average_weight[entity_id] = 0.0
        for neighbor_id in self._valid_entities:
            self._average_weight[entity_id] += super()._get_weight(entity_id, neighbor_id)
        self._average_weight[entity_id] /= len(self._valid_entities)

    def _verify_valid_entities(self, entity_id: int) -> None:
        if entity_id not in self._entity_index:
            return
        self._retained_neighbors.clear()
        for neighbor_id in self._valid_entities:
            _weight = self._get_valid_weight(entity_id, neighbor_id)
            if _weight:
                self._retained_neighbors.add(neighbor_id)
        if len(self._retained_neighbors) > 0:
            self.candidate_pairs[entity_id] = self._retained_neighbors.copy()

class BLAST(WeightedNodePruning):
    """Meta-blocking method that retains the comparisons \
        that correspond to edges in the blocking graph that are exceed 1/4 of the sum \
        of the maximum edge weights in the two adjacent node neighborhoods.
    """

    _method_name = _method_short_name = "BLAST"
    _method_info = "Meta-blocking method that retains the comparisons " + \
                "that correspond to edges in the blocking graph that are exceed 1/4 of the sum " + \
                "of the maximum edge weights in the two adjacent node neighborhoods."

    def __init__(self, weighting_scheme: str = 'X2') -> None:
        super().__init__(weighting_scheme)

    def _get_valid_weight(self, entity_id: int, neighbor_id: int) -> float:
        weight = self._get_weight(entity_id, neighbor_id)
        edge_threshold = (self._average_weight[entity_id] + self._average_weight[neighbor_id]) / 4
        return edge_threshold <= weight and entity_id < neighbor_id

    def _update_threshold(self, entity_id: int) -> None:
        if entity_id not in self._entity_index:
            return
        self._average_weight[entity_id] = 0.0
        for neighbor_id in self._valid_entities:
            self._average_weight[entity_id] = \
                max(self._average_weight[entity_id], self._get_weight(entity_id, neighbor_id))

class ReciprocalWeightedNodePruning(WeightedNodePruning):
    """Meta-blocking method that retains the comparisons\
        that correspond to edges in the blocking graph that are \
        exceed the average edge weight in both adjacent node neighborhoods.
    """

    _method_name = "Reciprocal Weighted Node Pruning"
    _method_short_name: str = "RWNP"
    _method_info = "Meta-blocking method that retains the comparisons " + \
                    "that correspond to edges in the blocking graph that are " + \
                    "exceed the average edge weight in both adjacent node neighborhoods."

    def __init__(self, weighting_scheme: str = 'CN-CBS') -> None:
        super().__init__(weighting_scheme)

    def _get_valid_weight(self, entity_id: int, neighbor_id: int) -> float:
        weight = self._get_weight(entity_id, neighbor_id)
        return weight if ((self._average_weight[entity_id] <= weight and \
                             self._average_weight[neighbor_id] <= weight) and
                                entity_id < neighbor_id) else 0
