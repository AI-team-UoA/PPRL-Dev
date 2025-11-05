"""Datamodel of pprl.
"""
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List, Tuple
import pickle
from ordered_set import OrderedSet
import pandas as pd
from bitarray import bitarray


class PPRLFeature(ABC):
    """
    Abstract base class for Privacy-Preserving Record Linkage (PPRL) feature methods.

    This class defines the interface and common functionality for all PPRL feature
    extraction and evaluation methods. Concrete subclasses must implement the
    abstract methods for specific feature evaluation techniques.

    Attributes:
        _method_name (str): Formal name of the method.
        _method_info (str): Detailed description of the method.
        _method_short_name (str): Abbreviated name for the method.
        execution_time (float): Time taken to execute the method in seconds.
    """

    _method_name: str
    _method_info: str
    _method_short_name: str

    def __init__(self) -> None:
        super().__init__()
        self.execution_time: float

    @abstractmethod
    def _configuration(self) -> dict:
        pass

    @abstractmethod
    def evaluate(self,
                 prediction: dict,
                 export_to_df: bool = False,
                 with_classification_report: bool = False,
                 verbose: bool = True) -> any:
        """
        Evaluate the PPRL method on given predictions.

        This abstract method must be implemented by subclasses to perform
        the actual evaluation of the PPRL feature method.

        Args:
            prediction: The prediction results to evaluate. Format depends on
                        the specific implementation.
            export_to_df (bool): If True, export results as pandas DataFrame.
            with_classification_report (bool): If True, include detailed
                classification metrics in the output.
            verbose (bool): If True, print progress and results to console.

        Returns:
            any: Evaluation results. The exact return type depends on the
                    export flags and specific implementation. Could be a
                    DataFrame, dictionary, or custom result object.

        Raises:
            NotImplementedError: If not implemented by subclass.
            ValueError: If prediction data is invalid or missing.
        """

    def method_configuration(self) -> dict:
        """Returns configuration details
        """
        return {
            "name" : self._method_name,
            "parameters" : self._configuration(),
            "runtime": self.execution_time
        }

    def report(self) -> None:
        """Prints method configuration
        """
        _configuration = self._configuration()
        parameters = ("\n" + ''.join([f'\t{k}: {v}\n' for k, v in _configuration.items()])) \
                        if len(self._configuration().items()) != 0 else ' None'
        print(
            "Method name: " + self._method_name +
            "\nMethod info: " + self._method_info +
            "\nParameters: " + parameters +
            f"\nRuntime: {self.execution_time:2.4f} seconds"
        )


class Block:
    """The main module used for storing entities in the blocking steps of pyjedai module. \
        Consists of 2 sets of profile entities 1 for Dirty ER and 2 for Clean-Clean ER.
    """
    def __init__(self) -> None:
        # self.entities = {"D0": OrderedSet, "D1": OrederedSet, ...}
        self.entities : Dict[str, OrderedSet] = defaultdict(OrderedSet)

    def get_cardinality(self) -> int:
        """Returns block cardinality.

        Args:
            is_dirty_er (bool): Dirty or Clean-Clean ER.

        Returns:
            int: Cardinality
        """
        return len(self.entities['D0']) * len(self.entities['D1'])

    def __str__(self):
        return f"{dict(self.entities)}"

class EncodedData(ABC):
    """
    Abstract base class representing encoded data for Privacy-Preserving Record Linkage (PPRL).

    This class provides a common interface for handling encoded or anonymized data
    in PPRL workflows. It stores the encoded representations of entities along with
    metadata and ground truth information for evaluation purposes.

    Attributes:
        encoded_dict (Dict[int, Dict[str, any]]): Dictionary containing encoded entities.
            Keys are entity identifiers, values are dictionaries of encoded attributes.
        bounds (List): List containing boundary information for the encoded data,
            typically used for spatial or range-based encodings.
        skip_ground_truth (bool): Flag indicating whether ground truth data should
            be processed or skipped during operations.
        ground_truth (pd.DataFrame): DataFrame containing the ground truth matching
            pairs for evaluation purposes.
    """
    encoded_dict: Dict[int, Dict[str, any]]
    bounds : List
    skip_ground_truth : bool
    ground_truth: pd.DataFrame

    def get_cardinality(self) -> int:
        """
        Get the number of encoded entities in the dataset.

        Returns:
            int: The cardinality (number of entities) in the encoded dataset.

        """
        return len(self.encoded_dict)

    @abstractmethod
    def to_file(self, filename: str):
        """
        Save the encoded data to a file.

        This abstract method must be implemented by subclasses to provide
        serialization functionality for the encoded data.

        Args:
            filename (str): Path to the output file where encoded data will be saved.

        """

    @classmethod
    @abstractmethod
    def from_file(cls, *kwargs):
        """
        Load encoded data from a file.

        This abstract class method must be implemented by subclasses to provide
        deserialization functionality for encoded data files.

        Args:
            *kwargs: Variable length argument list. Specific arguments depend on
                    the implementation but typically include:
                filename (str): Path to the input file containing encoded data.
                Other implementation-specific parameters.

        Returns:
            EncodedData: An instance of the specific EncodedData subclass loaded
                        with data from the file.
        """


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

    def to_entities(self, attribute: str = None) -> Tuple[dict, dict]:
        """
        Convert to entity-bit mapping and create inverted index.

        Args:
            attribute: Specific attribute to process, or None for all

        Returns:
            Tuple of (entity_to_bits, inverted_index)
        """
        to_entities : dict = {}
        inverted_index : Dict[int, Dict[int, set]]= {}
        dataset = 0
        if attribute:
            for idx, blooms in self.encoded_dict.items():
                if dataset < len(self.bounds) and idx >= self.bounds[dataset]:
                    dataset += 1
                to_entities[idx] = blooms[attribute]
                for bit in blooms[attribute]:
                    if bit not in inverted_index:
                        inverted_index[bit] = {key: set() for key in range(len(self.bounds))}
                    inverted_index[bit][dataset].add(idx)
        else:
            for idx, blooms in self.encoded_dict.items():
                if  dataset < len(self.bounds) and idx >= self.bounds[dataset]:
                    dataset += 1
                bit_set = set()
                for bits in blooms.values():
                    for bit in bits:
                        bit_set.add(bit)
                        if bit not in inverted_index:
                            inverted_index[bit] = {key: set() for key in range(len(self.bounds))}
                        inverted_index[bit][dataset].add(idx)

                to_entities[idx] = [*bit_set]

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
