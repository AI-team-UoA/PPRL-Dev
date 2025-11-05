"""PPRL Metablocking Methods"""

from itertools import chain
from collections import defaultdict
from math import log10, sqrt
from time import time
from abc import ABC, abstractmethod
from typing import Dict
import pandas as pd
import numpy as np

from src.pprl.evaluation import Evaluation
from src.pprl.datamodel import BloomEncodedData, PPRLFeature
from src.pprl.utils import chi_square



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
            self._entity_index, self._inverted_index = encoded_data.to_entities(attr)
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
            if (id1 in prediction and id2 in prediction[id1]) or   \
                (id2 in prediction and id1 in prediction[id2]):
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
        _limit = self.encoded_data.bounds[0]
        for i in range(0, _limit):
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
        _limit = self.encoded_data.bounds[0]
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
