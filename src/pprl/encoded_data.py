"""Encoded data class file"""

import pickle
from typing import Dict, List, Tuple
from collections import defaultdict
import itertools
import logging

import pandas as pd
import ray

from bitarray import bitarray
import numpy as np

from pprl.utils import  are_matching
from pprl.evaluation import Evaluation
from pprl.datamodel import EncodedData, PPRLFeature

@ray.remote
class ConfusionMatrixRay:
    """ConfusionMatrix Ray"""

    length: int
    ground_truth: list
    adjacent_bits: int
    _entities_d1: dict
    _entities_d2: dict
    total_length: int
    limit: int
    adjacent_list : list

    def __init__(self, length,
        ground_truth, adjacent_bits, total_length):
        self.length = length
        self.ground_truth = ground_truth
        self.adjacent_bits = adjacent_bits
        self.total_length = total_length
        self.adjacent_list = list(range(self.adjacent_bits))



    def _get_concat_bloom_filter(self, entity_id: int) -> set:
        _set_concat_bf = set()

        dict_to_use: Dict[any, dict] = self._entities_d1 \
            if entity_id in self._entities_d1 \
            else self._entities_d2

        for i, bloom_filter  in enumerate(
            dict_to_use[entity_id].values()):
            bloom_filter_concat = [(bloom_filter_index + self.length*i)
                    for bloom_filter_index in bloom_filter]
            _set_concat_bf.update(set(bloom_filter_concat))

        return _set_concat_bf

    def set_encoded_dict(self, encoded_dict, limit):
        """Sets encoded Dict for Confusion Matrix"""

        self._entities_d1 = { key: value
            for key, value in encoded_dict.items()
            if key < limit }

        self._entities_d2 = {key: value
            for key, value in encoded_dict.items()
            if key >= limit }
        self.limit = limit

    def calculate(self,
        index : int, batch_size: int):
        """Calculates Confusion Matrix With Ray"""
        total_matching_pairs = 0
        true_positives = 0


        for i in range(index, batch_size + index):
            _entity_d1 = self._get_concat_bloom_filter(i)
            matching_pairs, positives = self._calculate_total(
                        _entity_d1,
                        i)
            total_matching_pairs += matching_pairs
            true_positives += positives

        return total_matching_pairs, true_positives


    def _calculate_total(self,
        _entity_d1: set,
        index: int):

        total_matching_pairs = 0
        true_positives = 0

        for index_d2 in self._entities_d2:
            _entity_d2 = self._get_concat_bloom_filter(index_d2)

            matching = (index, index_d2-self.limit) in self.ground_truth

            matching_pairs, true_positive = self._calculate_one(
                    _entity_d1, _entity_d2, matching
            )
            total_matching_pairs += matching_pairs
            true_positives += true_positive

        return total_matching_pairs, true_positives

    def _calculate_one(self,
        _entity_d1: set,
        _entity_d2: set,
        matching: bool) -> tuple:

        total_matching_pairs = 0
        true_positive = 0
        for bit in range((self.total_length +
                          self.adjacent_bits - 1) //  self.adjacent_bits):
            adj_list = [(i + bit * self.adjacent_bits) for i in self.adjacent_list]
            cont_a = [i for i in adj_list if i in _entity_d1]
            cont_b = [i for i in adj_list if i in _entity_d2]
            if len(cont_b) == 0 or len(cont_a) == 0:
                continue
            if cont_a == cont_b:
                total_matching_pairs += 1
                if matching and not true_positive:
                    true_positive += 1

        return total_matching_pairs, true_positive


class BloomEncodedData(EncodedData, PPRLFeature):
    """
    Encoded data using Bloom filters for PPRL.

    Stores entity data encoded as Bloom filters with inverted index support
    for efficient similarity computation.
    """

    _method_name = "BloomEncodedData"
    _method_info = "BloomEncodedData"
    _method_short_name = "BED"

    dataset_limit : int
    bounds : List[int]
    inverted_index : Dict[int, Dict[str, set]]
    candidate_pairs: set
    """inverted_index :   position_of_bit : entities {D0: x,y,z,...} and {D1: x2,y2,z2,...}"""

    def __init__(self, data: Dict[int, Dict[str, List[int]]] = None, length: int = 0):
        """Initialize BloomEncodedData with optional pre-encoded data and bit length."""
        super().__init__()
        self.execution_time  = 0.0
        self.inverted_index = None
        self.skip_ground_truth = True
        if data:
            self.length = length
            self.encoded_dict : Dict[int, Dict[str, List[int]]] = data
            if self.encoded_dict:
                self.bitarray_dict : Dict[int, Dict[str, bitarray]] = {}
                for key, value in self.encoded_dict.items():
                    self.bitarray_dict[key] = {}
                    for attr, indices in value.items():
                        b = bitarray(length)
                        b.setall(0)
                        if indices:
                            b[indices] = True
                        self.bitarray_dict[key][attr] = b

                first_attr = next(iter(self.encoded_dict.values()), {})
                if first_attr:
                    self.attributes = list(first_attr.keys())

    def __str__(self):
        return f'{self.encoded_dict}'

    def __getstate__(self):
        """Customize what gets pickled."""
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        """Customize how to restore from pickle."""
        self.__dict__.update(state)


    def to_file(self, filename:str) :
        with open(filename, "wb+") as f:
            pickle.dump(self, f)

    def set_ground_truth(self, df: pd.DataFrame) :
        """
        Sets ground truth for evaluation
        Ground truth must be the indexes of the records


        Args:
            filename (str): Ground truth file path
        """

        self.ground_truth = df
        self.skip_ground_truth = False

    def to_entities(self,
            adjacent_bits: int = 1) -> Tuple[dict, dict]:
        """
        Convert to entity-bit mapping and create inverted index.
        Working only with concatenated bitarrays
        e.g if name : (0001) and email (1111)
        then will check 0001111 for as a whole bloom filter

        Args:
            attribute: Specific attribute to process, or None for all

        Returns:
            Tuple of (entity_to_bits, inverted_index)
        """
        to_entities : dict = {}
        inverted_index : Dict[int, Dict[int, set]]= {}
        dataset = 0


        for idx, blooms in self.encoded_dict.items():
            if dataset < len(self.bounds) and idx >= self.bounds[dataset]:
                dataset += 1

            attributes = self.attributes
            length_encoded_dict = self.length * len(attributes)
            concatenated_bloom = []
            for i,attr in enumerate(attributes):
                concatenated_bloom += [bl_i + i * self.length for bl_i in blooms[attr]]

            key_set = set()
            adjacent_list = list(range(adjacent_bits))
            for index in range((length_encoded_dict + adjacent_bits - 1) //  adjacent_bits):
                adj_list = [(i + index * adjacent_bits) for i in adjacent_list]
                bit_key = f'{index}'
                for i in adj_list:
                    if i in concatenated_bloom:
                        bit_key = f'{bit_key},{i}'
                if bit_key != f'{index}':
                    key_set.add(bit_key)
                if bit_key not in inverted_index:
                    inverted_index[bit_key] = {key: set() for key in range(len(self.bounds))}
                inverted_index[bit_key][dataset].add(idx)

            to_entities[idx] = [*key_set]

        return to_entities, dict(inverted_index)


    @classmethod
    def from_file(cls, *filenames) -> "BloomEncodedData":
        """Load a new instance from file."""
        ret : BloomEncodedData = None
        dataset_num = 0
        for filename in filenames:
            with open(filename, "rb") as f:
                obj : BloomEncodedData = pickle.load(f)
                if not ret:
                    ret = obj
                    ret.bounds = [len(ret.encoded_dict)]
                    # ret.dataset_indexes[dataset_num] = (0, len(ret.encoded_dict)-1)
                else:
                    offset = len(ret.encoded_dict)
                    encoded_dict = obj.encoded_dict
                    bitarray_dict = obj.bitarray_dict
                    ret.encoded_dict = {**ret.encoded_dict,
                                **{key + offset :  value for key, value in encoded_dict.items()} }
                    ret.bitarray_dict = {**ret.bitarray_dict,
                                **{key + offset :  value for key, value in bitarray_dict.items()} }
                    ret.bounds.append(offset + len(obj.encoded_dict))

            dataset_num += 1

        ret.dataset_limit = offset
        ret.inverted_index = None

        return ret

    def _create_inverted_index(self) -> None:
        self.inverted_index = defaultdict(lambda : defaultdict(set))
        dataset = 0
        for i, encoded in self.encoded_dict.items():
            if  i >= self.bounds[dataset]:
                dataset += 1
            for attr_bit_positions in encoded.values():
                for bit_position in attr_bit_positions:
                    self.inverted_index[bit_position][dataset].add(i)


    def _serial_evaluate(self, entity_index: dict):
        true_positives = 0
        for _, (id1, id2) in self.ground_truth.iterrows():
            id2 = id2 + self.bounds[0]
            if id1 in entity_index and    \
            id2 in entity_index and are_matching(entity_index, id1, id2):
                true_positives += 1

        return true_positives

    def calculate_candidate_pairs(self, inverted_index: dict) -> int:
        """Returns length of candidate pairs"""
        candidate_pairs  = set()
        for block in inverted_index.values():
            candidate_pairs.update(set(
                itertools.product(block[0], block[1])))
        return len(candidate_pairs)

    def _get_set_concat_bloom_filter(self, entity_id: int) -> set:
        _set_concat_bf = set()
        for i, bloom_filter  in enumerate(
            self.encoded_dict[entity_id].values()):
            bloom_filter_concat = [(bloom_filter_index + self.length*i)
                    for bloom_filter_index in bloom_filter]
            _set_concat_bf.update(set(bloom_filter_concat))


        return _set_concat_bf



    @ray.remote
    def _ray_get_candidate_pairs(self, index: int, adjacent_bits: int) -> list:
        adjacent_list = list(range(adjacent_bits))
        total_length = len(self.attributes) * self.length
        _entity_d1 = self._get_set_concat_bloom_filter(index)
        _entities_d2 = {key: value
                for key, value in self.encoded_dict.items()
                if key >= self.bounds[0]}

        candidate_pairs = []

        for index_d2 in _entities_d2:
            _entity_d2 = self._get_set_concat_bloom_filter(index_d2)
            for bit in range((total_length + adjacent_bits - 1) //  adjacent_bits):
                adj_list = [(i + bit * adjacent_bits) for i in adjacent_list]
                cont_a = [i for i in adj_list if i in _entity_d1]
                cont_b = [i for i in adj_list if i in _entity_d2]
                if len(cont_b) == 0 or len(cont_a) == 0:
                    continue
                if cont_a == cont_b:
                    candidate_pairs.append((index, index_d2))
                    break

        return candidate_pairs





    def get_candidate_pairs(self, adjacent_bits: int = 1) -> list:
        """
        Function that produces candidate pairs based on blocking of adjacent bits
        Uses Ray
        Args:
            adjacent_bits (int): Adjacent bits that are used as blocking keys

        Returns:
            list: All the candidate pairs

        """


        if not ray.is_initialized():
            logging.getLogger("ray").setLevel(logging.ERROR)
            ray.init(log_to_driver=False, logging_level=logging.ERROR)

        self_ref = ray.put(self)
        futures = []

        for i in range(self.bounds[0]):
            future = self._ray_get_candidate_pairs.remote(self_ref,
                index=i,
                adjacent_bits = adjacent_bits,
            )
            futures.append(future)

        results = ray.get(futures)
        candidate_pairs = []
        for pairs in results:
            candidate_pairs.extend(pairs)

        return candidate_pairs


    def serial_evaluate(self,
        adjacent_bits: int = 1,
        export_to_df: bool = False,
        with_classification_report: bool = False,
        verbose: bool = True) -> any:
        """Function to evaluate encoder treating it
        as a blocking method: f1-score, recall and precision

        Args:
            export_to_df (bool) : Create evaluation dataframe
            with_classification_report (bool) : Printing Info for the blocking method
            with_stats (bool) : Printing Method's Statistics

        Returns:
            Evaluation : Evaluation Object with F1, Recall, Precision etc.
        """
        if self.skip_ground_truth:
            raise AttributeError("Can not proceed to evaluation without a ground-truth file."
                    "Data object has not been initialized with the ground-truth file")

        eval_obj = Evaluation(self)

        ground_truth_list = list(self.ground_truth.itertuples(index=False, name=None))
        total_matching_pairs = 0
        true_positives = 0
        total_length = len(self.attributes) * self.length

        self.candidate_pairs = set()

        bloom_filter_matrix = np.zeros(
            (len(self.encoded_dict), total_length),
            dtype=bool )



        for entity_id, fields in self.encoded_dict.items():
            for i, bloom_filter_indices in enumerate(fields.values()):
                offset = i * self.length
                if bloom_filter_indices:
                    bloom_filter_matrix[entity_id,
                        np.array(bloom_filter_indices) + offset]  = True


        for id_1 in range(self.bounds[0]):
            entity_1 = bloom_filter_matrix[id_1]
            for id_2 in range(self.bounds[0], self.bounds[1]):
                entity_2 = bloom_filter_matrix[id_2]
                matching_pairs = self._get_matching_pairs(
                    entity_1, entity_2,
                    adjacent_bits
                )
                total_matching_pairs += matching_pairs
                if (id_1, id_2) not in self.candidate_pairs and \
                    matching_pairs:
                    if (id_1, id_2 - self.bounds[0]) in ground_truth_list:
                        true_positives += 1
                    self.candidate_pairs.add((id_1, id_2))

        eval_obj.calculate_scores(true_positives=true_positives,
                                total_matching_pairs=total_matching_pairs)
        eval_result = eval_obj.report(self.method_configuration(),
                                export_to_df,
                                with_classification_report,
                                verbose)

        return eval_result

    def _get_matching_pairs(self, entity_1: np.array,
                    entity_2: np.array,
                    adjacent_bits):
        matching_pairs = 0

        n_slices = len(entity_1) // adjacent_bits
        limit = n_slices * adjacent_bits

        sliced_1: np.array = entity_1[:limit].reshape(n_slices, adjacent_bits)
        sliced_2: np.array = entity_2[:limit].reshape(n_slices, adjacent_bits)

        sliced_1_has_active_bits = sliced_1.any(axis=1)
        sliced_2_has_active_bits = sliced_2.any(axis=1)

        matches = (sliced_1 == sliced_2).all(axis=1)

        valid_matches = matches & \
            sliced_1_has_active_bits & sliced_2_has_active_bits

        matching_pairs = np.count_nonzero(valid_matches)

        if len(entity_1) % adjacent_bits != 0:
            remaining_1 = entity_1[limit:]
            remaining_2 = entity_2[limit:]
            remaining_1_has_active_bits = remaining_1.any()
            remaining_2_has_active_bits = remaining_2.any()
            matches = np.array_equal(remaining_1, remaining_2)
            if matches and remaining_1_has_active_bits \
                and remaining_2_has_active_bits:
                    matching_pairs += 1



        return matching_pairs


    def evaluate(self,
            adjacent_bits: int = 1,
            batch_size: int = 100,
            export_to_df: bool = False,
            with_classification_report: bool = False,
            verbose: bool = True) -> any:
        """Function to evaluate encoder treating it
        as a blocking method: f1-score, recall and precision

        Args:
            export_to_df (bool) : Create evaluation dataframe
            with_classification_report (bool) : Printing Info for the blocking method
            with_stats (bool) : Printing Method's Statistics

        Returns:
            Evaluation : Evaluation Object with F1, Recall, Precision etc.
        """
        if self.skip_ground_truth:
            raise AttributeError("Can not proceed to evaluation without a ground-truth file."
                    "Data object has not been initialized with the ground-truth file")

        eval_obj = Evaluation(self)

        ground_truth_list = list(self.ground_truth.itertuples(index=False, name=None))
        total_matching_pairs = 0
        true_positives = 0


        if not ray.is_initialized():
            logging.getLogger("ray").setLevel(logging.ERROR)
            ray.init(log_to_driver=False, logging_level=logging.ERROR)

        encoded_dict_ref = ray.put(self.encoded_dict)

        total_length = len(self.attributes) * self.length

        num_actors = 15

        actors = []
        set_encoded_dict = []

        for _ in range(num_actors):
            actor = ConfusionMatrixRay.remote(self.length,
                                ground_truth_list,
                                adjacent_bits, total_length)
            future = actor.set_encoded_dict.remote(encoded_dict_ref, self.bounds[0])
            actors.append(actor)
            set_encoded_dict.append(future)

        ray.get(set_encoded_dict)


        actor_index = 0


        futures = []
        for i in range(0, self.bounds[0], batch_size):
            actor = actors[actor_index % num_actors]
            future = actor.calculate.remote(i, batch_size)
            futures.append(future)
            actor_index += 1

        results = ray.get(futures)
        for total, posititves in results:
            total_matching_pairs += total
            true_positives += posititves

        eval_obj.calculate_scores(true_positives=true_positives,
                                total_matching_pairs=total_matching_pairs)
        eval_result = eval_obj.report(self.method_configuration(),
                                export_to_df,
                                with_classification_report,
                                verbose)

        return eval_result

    def _configuration(self):
        return {}


class HomomorphicEcnodedData(EncodedData):
    """
    Encoded data using homomorphic encryption for PPRL.

    Stores entity data encoded with homomorphic encryption schemes,
    allowing computation on encrypted data without decryption.

    NOT YET IMPLEMENTED
    """
    def __init__(self, data: Dict[int, Dict[str, List[int]]]):
        self.encoded_dict : Dict[int, Dict[str, List[int]]] = data
        if self.encoded_dict:
            first_attr = next(iter(self.encoded_dict.values()), {})
            if first_attr:
                self.attributes = list(first_attr.keys())

    def __str__(self):
        return f'{self.encoded_dict}'
