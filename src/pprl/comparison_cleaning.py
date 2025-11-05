import sys
import warnings
import pandas as pd
from itertools import chain
from collections import defaultdict
from logging import warning
from math import log10, sqrt
from queue import PriorityQueue
from time import time

import numpy as np
import math
from tqdm.autonotebook import tqdm

from .evaluation import Evaluation

from .datamodel import BloomEncodedData, PPRLFeature
# from .utils import chi_square, create_entity_index, get_sorted_blocks_shuffled_entities, PositionIndex, canonical_swap, sorted_enumerate

from abc import ABC, abstractmethod
from typing import Tuple, List, Dict


class AbstractComparisonCleaning(PPRLFeature):
    """Abstract class for Block cleaning
    """

    def __init__(self) -> None:
        super().__init__()
        self._limit: int
        self._num_of_blocks: int
        self._valid_entities: set = set()
        self._entity_index: dict
        self._weighting_scheme: str
        self._inverted_index : dict
        self.inverted_index : dict = None
        # self._blocks: dict # initial blocks
        # self.blocks = dict() # blocks after CC
        self._node_centric: bool

    def process(
            self,
            # blocks: dict,
            encoded_data: BloomEncodedData,
            tqdm_disable: bool = False,
            store_weights: bool = False
    ) -> dict:
        """Main method for comparison cleaning

        Args:
            encoded_data (BloomEncodedData): bloom filters created from previous steps of pprl
            tqdm_disable (bool, optional): Disables tqdm progress bars. Defaults to False.

        Returns:
            dict: cleaned blocks
        """
        start_time = time()
        self.tqdm_disable, self.encoded_data, self.store_weights = tqdm_disable, encoded_data, store_weights
        self._limit = self.encoded_data.bounds[0]
        self._progress_bar = tqdm(
            total=self._limit,
            desc=self._method_name,
            disable=self.tqdm_disable
        )

        self._stored_weights =  defaultdict(float) if self.store_weights else None

        self.candidate_pairs : Dict[int, set] = defaultdict(set)

        for attr in encoded_data.attributes:
            self._entity_index, self._inverted_index = encoded_data.to_entities(attr)
            print(self._entity_index)
            print(self._inverted_index)

            self._num_of_bits = len(self._inverted_index)
            # print(self.candidate_pairs)
            self.candidate_pairs = self._apply_main_processing()

        self.execution_time = time() - start_time
        self._progress_bar.close()

        return self.candidate_pairs

    def report(self) -> None:
        """Prints Block Building method configuration
        """
        print(
            "Method name: " + self._method_name +
            "\nMethod info: " + self._method_info +
            ("\nParameters: \n" + ''.join(['\t{0}: {1}\n'.format(k, v) for k, v in self._configuration().items()]) if self._configuration().items() else "\nParameters: Parameter-Free method\n") +
            "Attributes from D1:\n\t" + ', '.join(c for c in (self.attributes_1 if self.attributes_1 is not None \
                else self.data.dataset_1.columns)) +
            ("\nAttributes from D2:\n\t" + ', '.join(c for c in (self.attributes_2 if self.attributes_2 is not None \
                else self.data.dataset_2.columns)) if not self.data.is_dirty_er else "") +
            "\nRuntime: {:2.4f} seconds".format(self.execution_time)
        )

    @abstractmethod
    def _apply_main_processing(self) -> dict:
        pass

    @abstractmethod
    def _configuration(self) -> dict:
        pass
    
    def stats(self) -> None:
        pass
    
    def get_precalculated_weight(self, entity_id: int, neighbor_id: int) -> float:
        """Returns the precalculated weight for given pair

        Args:
            entity_id (int): Entity ID
            neighbor_id (int): Neighbor ID

        Raises:
            AttributeError: Given pair has no precalculated weigth

        Returns:
            float: Pair weigth
        """
        if(not self.store_weights): raise AttributeError("No precalculated weights.")
        _weight = self._stored_weights.get(canonical_swap(entity_id, neighbor_id), KeyError(f"Pair [{entity_id},{neighbor_id}] has no precalculated weight"))
        return _weight
        
    def evaluate(self,
                 prediction,
                 export_to_df: bool = False,
                 export_to_dict: bool = False,
                 with_classification_report: bool = False,
                 verbose: bool = True) -> any:

        if self.data is None:
            raise AttributeError("Can not proceed to evaluation without data object.")

        if self.data.ground_truth is None:
            raise AttributeError("Can not proceed to evaluation without a ground-truth file. " +
                    "Data object has not been initialized with the ground-truth file")

        eval_obj = Evaluation(self.data)
        true_positives = 0
        total_matching_pairs = sum([len(block) for block in prediction.values()])
        for _, (id1, id2) in self.data.ground_truth.iterrows():
            id1 = self.data._ids_mapping_1[id1]
            id2 = self.data._ids_mapping_1[id2] if self.data.is_dirty_er \
                                                else self.data._ids_mapping_2[id2]
            if (id1 in prediction and id2 in prediction[id1]) or   \
                (id2 in prediction and id1 in prediction[id2]):
                true_positives += 1

        eval_obj.calculate_scores(true_positives=true_positives, 
                                  total_matching_pairs=total_matching_pairs)
        return eval_obj.report(self.method_configuration(),
                                export_to_df,
                                export_to_dict,
                                with_classification_report,
                                verbose)

    def export_to_df(self, prediction) -> pd.DataFrame:
        """creates a dataframe with the predicted pairs

        Args:
            prediction (any): Predicted candidate pairs

        Returns:
            pd.DataFrame: Dataframe with the predicted pairs
        """
        pairs_df = pd.DataFrame(columns=['id1', 'id2'])
        for entity_id, candidates in prediction.items():
            id1 = self.data._gt_to_ids_reversed_1[entity_id]                                           
            for candiadate_id in candidates:
                id2 = self.data._gt_to_ids_reversed_1[candiadate_id] if self.data.is_dirty_er \
                        else self.data._gt_to_ids_reversed_2[candiadate_id]
                pairs_df = pd.concat([pairs_df, pd.DataFrame([{'id1':id1, 'id2':id2}], index=[0])], ignore_index=True)

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
        self._flags: np.array
        self._comparisons_per_entity: np.array
        self._node_centric: bool
        self._threshold: float
        self._distinct_comparisons: int
        self._comparisons_per_entity: np.array
        self._neighbors: set = set()
        self._retained_neighbors: set = set()
        self._block_assignments: int = 0
        self.weighting_scheme: str

    def _apply_main_processing(self) -> dict:
        self._counters = np.empty([len(self._entity_index)], dtype=float)
        self._flags = np.empty([len(self._entity_index)], dtype=int)
        if(self._comparisons_per_entity_required()):
            self._set_statistics()
        self._set_threshold()

        return self._prune_edges()


    def _comparisons_per_entity_required(self):
        return (self.weighting_scheme == 'EJS' or 
                self.weighting_scheme == 'CNC' or
                self.weighting_scheme == 'SNC' or
                self.weighting_scheme == 'SND' or
                self.weighting_scheme == 'CND' or
                self.weighting_scheme == 'CNJ' or
                self.weighting_scheme == 'SNJ')
        
    def _get_weight(self, entity_id: int, neighbor_id: int) -> float:
        ws = self.weighting_scheme
        if ws == 'CN-CBS' or ws == 'CBS' or ws == 'SN-CBS':
            return self._counters[neighbor_id]
        # CARDINALITY_NORM_COSINE, SIZE_NORM_COSINE
        elif ws == 'CNC' or ws == 'SNC':
            return self._counters[neighbor_id] / float(sqrt(self._comparisons_per_entity[entity_id] * self._comparisons_per_entity[neighbor_id]))
        # SIZE_NORM_DICE, CARDINALITY_NORM_DICE
        elif ws == 'SND' or ws == 'CND':
            return 2 * self._counters[neighbor_id] / float(self._comparisons_per_entity[entity_id] + self._comparisons_per_entity[neighbor_id])
        # CARDINALITY_NORM_JS, SIZE_NORM_JS
        elif ws == 'CNJ' or ws == 'SNJ':
            return  self._counters[neighbor_id] / float(self._comparisons_per_entity[entity_id] + self._comparisons_per_entity[neighbor_id] - self._counters[neighbor_id])
        elif ws == 'COSINE':
            return self._counters[neighbor_id] / float(sqrt(len(self._entity_index[entity_id]) * len(self._entity_index[neighbor_id])))
        elif ws == 'DICE':
            return 2 * self._counters[neighbor_id] / float(len(self._entity_index[entity_id]) + len(self._entity_index[neighbor_id]))
        elif ws == 'ECBS':
            return float(
                self._counters[neighbor_id] *
                log10(float(self._num_of_blocks / len(self._entity_index[entity_id]))) *
                log10(float(self._num_of_blocks / len(self._entity_index[neighbor_id])))
            )
        elif ws == 'JS':
            return self._counters[neighbor_id] / (len(self._entity_index[entity_id]) + \
                    len(self._entity_index[neighbor_id]) - self._counters[neighbor_id])
        elif ws == 'EJS':
            probability = self._counters[neighbor_id] / (len(self._entity_index[entity_id]) + \
                            len(self._entity_index[neighbor_id]) - self._counters[neighbor_id])
            return float(probability * \
                    log10(self._distinct_comparisons / self._comparisons_per_entity[entity_id]) * \
                    log10(self._distinct_comparisons / self._comparisons_per_entity[neighbor_id]))
        elif ws == 'X2':
            observed = [int(self._counters[neighbor_id]),
                        int(len(self._entity_index[entity_id])-self._counters[neighbor_id])]
            expected = [int(len(self._entity_index[neighbor_id])-observed[0]),
                        int(self._num_of_blocks - (observed[0] + observed[1] - self._counters[neighbor_id]))]
            return None
            # return chi_square(np.array([observed, expected]))
        else:
            raise ValueError("This weighting scheme does not exist")

    def _normalize_neighbor_entities(self, bit_pos: int, entity_id: int) -> None:
        self._neighbors.clear()
        if entity_id < self._limit:
            self._neighbors.update(self._inverted_index[bit_pos][1])
        else:
            self._neighbors.update(self._inverted_index[bit_pos][0])

    def _set_statistics(self) -> None:
        self._distinct_comparisons = 0
        self._comparisons_per_entity = np.empty([self.data.num_of_entities], dtype=float)
        distinct_neighbors = set()
        for entity_id in range(0, self.data.num_of_entities, 1):
            if entity_id in self._entity_index:
                associated_blocks = self._entity_index[entity_id]
                distinct_neighbors.clear()
                # distinct_neighbors = set().union(*[
                #     self._get_neighbor_entities(block_id, entity_id) for block_id in associated_blocks
                # ])
                distinct_neighbors = set(chain.from_iterable(
                    self._get_neighbor_entities(block_id, entity_id) for block_id in associated_blocks
                ))
                # for block_id in associated_blocks:
                #     distinct_neighbors = set.union(
                #         distinct_neighbors,
                #         self._get_neighbor_entities(block_id, entity_id)
                #     )
                self._comparisons_per_entity[entity_id] = len(distinct_neighbors)

                if self.data.is_dirty_er:
                    self._comparisons_per_entity[entity_id] -= 1

                self._distinct_comparisons += self._comparisons_per_entity[entity_id]
        self._distinct_comparisons /= 2

    def _get_neighbor_entities(self, block_id: int, entity_id: int) -> set:
        return self._blocks[block_id].entities_D2 \
            if (not self.data.is_dirty_er and entity_id < self.data.dataset_limit) else \
                self._blocks[block_id].entities_D1

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

    def _prune_edges(self) -> dict:
        for i in range(0, self._limit):
            self._process_entity(i)
            self._verify_valid_entities(i)
            self._progress_bar.update(1)

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
                if self.weighting_scheme == 'CN-CBS' or self.weighting_scheme == 'CNC' or self.weighting_scheme == 'CND' or self.weighting_scheme == 'CNJ':
                    cardinality = len(self._inverted_index[bit_pos][0]) * len(self._inverted_index[bit_pos][1])
                    self._counters[neighbor_id] += 1 / cardinality
                if self.weighting_scheme == 'SN-CBS' or self.weighting_scheme == 'SNC' or self.weighting_scheme == 'SND' or self.weighting_scheme == 'SNJ':
                    block_size = len(self._inverted_index[bit_pos][0]) + len(self._inverted_index[bit_pos][1])
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
        for i in range(0, self._limit):
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
                if self.store_weights:
                    self._stored_weights[canonical_swap(entity_id, neighbor_id)] = weight

        if len(self._retained_neighbors) > 0:
            self.candidate_pairs[entity_id].update(self._retained_neighbors.copy())

    def _configuration(self) -> dict:
        return {
            "Node centric" : self._node_centric,
            "Weighting scheme" : self.weighting_scheme
        }