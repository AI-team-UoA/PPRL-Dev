"""Encoded data class file"""

import pickle
from typing import Dict, List, Tuple
import pandas as pd
from bitarray import bitarray

import pandas as pd
from src.pprl.utils import  are_matching, get_blocks_cardinality
from src.pprl.evaluation import Evaluation
from src.pprl.datamodel import EncodedData


class BloomEncodedData(EncodedData):
    """
    Encoded data using Bloom filters for PPRL.

    Stores entity data encoded as Bloom filters with inverted index support
    for efficient similarity computation.
    """

    dataset_limit : int
    bounds : List[int]
    inverted_index : Dict[int, Dict[str, set]]
    """inverted_index :   position_of_bit : entities {D0: x,y,z,...} and {D1: x2,y2,z2,...}"""

    def __init__(self, data: Dict[int, Dict[str, List[int]]] = None, length: int = 0):
        """Initialize BloomEncodedData with optional pre-encoded data and bit length."""

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

    def to_entities(self, attributes: List[str] = None,
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

            attributes = attributes if attributes else self.attributes
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


    def evaluate(self,
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

        entity_index, inverted_index = self.to_entities()

        new_dict = {}

        for key, value in entity_index.items():
            new_dict[key] = set(value)

        entity_index = new_dict
        print(entity_index[0])

        total_matching_pairs = sum(len(entities[0]) * len(entities[1])
                    for entities in inverted_index.values())
        print(total_matching_pairs)
        exit(10)

        true_positives = self._serial_evaluate(entity_index)
        total_matching_pairs = get_blocks_cardinality(eval_blocks)

        # eval_obj.calculate_scores(true_positives=true_positives,
        #                         total_matching_pairs=total_matching_pairs)
        # eval_result = eval_obj.report(self.method_configuration(),
        #                         export_to_df,
        #                         with_classification_report,
        #                         verbose)

        return eval_result




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
