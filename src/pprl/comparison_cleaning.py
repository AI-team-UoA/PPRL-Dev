"""PPRL Metablocking Methods"""
import sys
from queue import PriorityQueue
from itertools import chain
from collections import defaultdict
from math import log10, sqrt
from time import time
from abc import ABC, abstractmethod
from typing import Dict
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
        self._entity_index : dict
        self._inverted_index : dict
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

        for attr in encoded_data.attributes:
            self._entity_index, self._inverted_index = encoded_data.to_entities(attr, adjacent_bits)
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
        self._counters = np.empty([len(self._entity_index)], dtype=float)
        self._flags = np.empty([len(self._entity_index)], dtype=int)
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

    def _set_statistics(self) -> None:
        if self.weighting_scheme not in {'EJS',
                'CNC', 'SNC', 'SND', 'CND', 'CNJ', 'SNJ'} :
            return

        num_of_entities = len(self.encoded_data.encoded_dict)
        self._distinct_comparisons = 0
        self._comparisons_per_entity = np.empty([num_of_entities], dtype=float)
        distinct_neighbors = set()
        for entity_id in range(0, num_of_entities, 1):
            if entity_id in self._entity_index:
                associated_blocks = self._entity_index[entity_id]
                distinct_neighbors.clear()
                distinct_neighbors = set(chain.from_iterable(
                    self._get_neighbor_entities(block_id, entity_id)
                    for block_id in associated_blocks
                ))
                self._comparisons_per_entity[entity_id] = len(distinct_neighbors)

                self._distinct_comparisons += self._comparisons_per_entity[entity_id]
        self._distinct_comparisons /= 2

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
        for i in tqdm(range(0, _limit), desc="Processing entities", position=1, leave=False):
            self._process_entity(i)
            self._verify_valid_entities(i)

        return self.candidate_pairs

    def _process_entity(self, entity_id: int) -> None:
        if entity_id not in self._entity_index:
            return
        self._valid_entities.clear()
        self._flags[:] = -1
        associated_bits = self._entity_index[entity_id]
        for bit_pos in associated_bits:
            self._normalize_neighbor_entities(bit_pos, entity_id)
            for neighbor_id in self._neighbors:
                if self._flags[neighbor_id] != entity_id:
                    self._counters[neighbor_id] = 0
                    self._flags[neighbor_id] = entity_id
                if self.weighting_scheme in {'CN-CBS', 'CNC', 'CND', 'CNJ'}:
                    cardinality = len(self._inverted_index[bit_pos][0]) * \
                        len(self._inverted_index[bit_pos][1])
                    self._counters[neighbor_id] += 1 / cardinality
                if self.weighting_scheme in {'SN-CBS', 'SNC', 'SND', 'SNJ'}:
                    block_size = len(self._inverted_index[bit_pos][0]) + \
                        len(self._inverted_index[bit_pos][1])
                    self._counters[neighbor_id] += 1 / block_size
                else:
                    self._counters[neighbor_id] += 1
                self._valid_entities.add(neighbor_id)

    def _update_threshold(self, entity_id: int) -> None:
        if entity_id not in self._entity_index:
            return

        self._num_of_edges += len(self._valid_entities)
        for neighbor_id in self._valid_entities:
            self._threshold += self._get_weight(entity_id, neighbor_id)

    def _set_threshold(self):
        self._num_of_edges = 0.0
        self._threshold = 0.0
        _limit = self.encoded_data.bounds[0] if not self._node_centric \
            else len(self.encoded_data.encoded_dict)
        for i in range(_limit):
            self._process_entity(i)
            self._update_threshold(i)

        self._threshold /= self._num_of_edges

    def _verify_valid_entities(self, entity_id: int) -> None:
        if entity_id not in self._entity_index:
            return

        self._retained_neighbors.clear()
        for neighbor_id in self._valid_entities:
            weight = self._get_weight(entity_id, neighbor_id)
            if self._threshold <= weight:
                self._retained_neighbors.add(neighbor_id)

        if len(self._retained_neighbors) > 0:
            self.candidate_pairs[entity_id].update(self._retained_neighbors.copy())


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

        self._top_k_edges = PriorityQueue(int(2*self._threshold))
        for i in range(0, _limit):
            self._process_entity(i)
            self._verify_valid_entities(i)
        while not self._top_k_edges.empty():
            comparison = self._top_k_edges.get()
            self.candidate_pairs[comparison[1]].add(comparison[2])

        return self.candidate_pairs

    def _set_threshold(self) -> None:
        bit_assignments = self._get_bit_assignmets()
        self._threshold = bit_assignments / 2

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
        self._number_of_nearest_neighbors : int = None

    def _prune_edges(self) -> dict:
        _limit = self.encoded_data.bounds[0] if not self._node_centric \
            else len(self.encoded_data.encoded_dict)

        self._nearest_entities = {}
        for i in range(0, _limit):
            self._process_entity(i)
            self._verify_valid_entities(i)
        return self._retain_valid_comparisons()

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
        if self._number_of_nearest_neighbors is None:
            num_of_entities = len(self.encoded_data.encoded_dict)
            bit_assignments = self._get_bit_assignmets()
            self._threshold = max(1, bit_assignments / num_of_entities)
        else:
            self._threshold = self._number_of_nearest_neighbors

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

    def _set_threshold(self):
        _limit = self.encoded_data.bounds[0] if not self._node_centric \
            else len(self.encoded_data.encoded_dict)

        self._average_weight = np.zeros(_limit, dtype=float)
        for i in range(0, _limit):
            self._process_entity(i)
            self._update_threshold(i)

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
