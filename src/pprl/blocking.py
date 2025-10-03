import random
import numpy as np
from bitarray import bitarray
from collections import defaultdict
from typing import List, Dict, Union
from pybloom_live import BloomFilter as Bloom  
from src.pprl.datamodel import Data, Block, PPRLFeature
import time
from abc import abstractmethod

class AbstractBlockProcessing(PPRLFeature):
    """ 
    Abstract Class for blocking 
    """

    def __init__(self):
        super().__init__()
        self.blocks : dict

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

    """
    def _serial_evaluate(self, entity_index) -> int:
        true_positives = 0
        for _, (id1, id2) in self.data.ground_truth.iterrows():
            id1 = self.data._ids_mapping_1[id1]
            id2 = self.data._ids_mapping_1[id2] if self.data.is_dirty_er else self.data._ids_mapping_2[id2]
            if id1 in entity_index and    \
                id2 in entity_index and are_matching(entity_index, id1, id2):
                    true_positives += 1 

        return true_positives
    

    def evaluate(self,
                 prediction,
                 export_to_df: bool = False,
                 export_to_dict: bool = False,
                 with_classification_report: bool = False,
                 verbose: bool = True,
                 with_stats: bool = False) -> any:

        if prediction is None:
            if self.blocks is None:
                raise AttributeError("Can not proceed to evaluation without build_blocks.")
            else:
                eval_blocks = self.blocks
        else:
            eval_blocks = prediction
            
        if self.data is None:
            raise AttributeError("Can not proceed to evaluation without data object.")

        if self.data.ground_truth is None:
            raise AttributeError("Can not proceed to evaluation without a ground-truth file. " + 
                    "Data object has not been initialized with the ground-truth file")

        eval_obj = Evaluation(self.data)
        entity_index = eval_obj._create_entity_index_from_blocks(eval_blocks)
        
        if self.disable_ray and self.disable_eval_ray: 
            true_positives = self._serial_evaluate(entity_index)
        else: 
            true_positives = self._ray_evaluate(entity_index)
              
        
        total_matching_pairs = get_blocks_cardinality(eval_blocks, self.data.is_dirty_er)
        eval_obj.calculate_scores(true_positives=true_positives, total_matching_pairs=total_matching_pairs)
        eval_result = eval_obj.report(self.method_configuration(),
                                export_to_df,
                                export_to_dict,
                                with_classification_report,
                                verbose)
        if with_stats:
            self.stats(eval_blocks)
        return eval_result
        """
    
    def stats(self, blocks: dict, verbose: bool = True) -> dict:
        return None


class AbstractBlockBuilding(AbstractBlockProcessing):
    """Abstract class for the block building method
    """

    _method_name: str
    _method_info: str
    _method_short_name: str
    
    
    def __init__(self, seed : int = 42):
        super().__init__()
        self.seed : int = seed

        
    def build_blocks(
            self,
            data: Data,
            bloom_filters: Dict[int, Dict[str, Bloom]],
            attributes : List[str] = None,
    ) -> Dict[Union[str, int], Block]:
        """Main method of Blocking in a dataset

            Args:
                data (Data): Data module that contaiins the processed dataset
                attributes (list, optional): Attributes columns of the datasets
                    that will be processed. Defaults to None. \
                    If not provided, all attributes are selected.

                tqdm_disable (bool, optional): Disables all tqdm at processing. Defaults to False.

            Returns:
                Dict[token, Block]: Dictionary of blocks.
        """

        _start_time = time.time()
        self.data = data

        _, first_bloom_filters = next(iter(bloom_filters.items()))
        bloom_attributes = list(first_bloom_filters.keys())

        if attributes:
            if set(attributes) > set(bloom_attributes): 
                raise ValueError(f"Attributes must be a subset of the attributes in Bloom Filters : {bloom_attributes}")
            self.attributes = attributes
        else: 
            self.attributes = bloom_attributes


        # self._progress_bar = tqdm(
        #     total=data.num_of_entities, desc=self._method_name, disable=tqdm_disable
        # )
        
        
        self._fit(bloom_filters)

        blocks = defaultdict(Block)
        for idx, bloom_dict in bloom_filters.items():
            for _, bloom in bloom_dict.items():
                for key in self._block_record(bloom.bitarray):
                    if idx < self.data.dataset_limit: 
                        blocks[str(key)].entities_D1.add(idx)
                    else: 
                        blocks[str(key)].entities_D2.add(idx)

                                    
        self.original_num_of_blocks = len(blocks)
        self.blocks = blocks
        # self.blocks = self._clean_blocks(blocks)
        # self.num_of_blocks_dropped = len(blocks) - len(self.blocks)
        self.execution_time = time.time() - _start_time
        # self._progress_bar.close()
        
        return self.blocks

        # blocks = defaultdict(Block)
        # for eid, entity in enumerate(self._entities_d1):
        #     for token in entity:
        #         blocks[token].entities_D1.add(eid)
        #     self._progress_bar.update(1)
        
        # if not data.is_dirty_er:
        #     for eid_off, entity in enumerate(self._entities_d2):
        #         eid = self.data.dataset_limit + eid_off
        #         for token in entity:
        #             blocks[token].entities_D2.add(eid)
        #         self._progress_bar.update(1)


    def report(self) -> None:
        """Prints Block Building method configuration
        """
        print(
            "Method name: " + self._method_name +
            "\nMethod info: " + self._method_info +
            ("\nParameters: \n" + ''.join(['\t{0}: {1}\n'.format(k, v) for k, v in self._configuration().items()]) if self._configuration().items() else "\nParameters: Parameter-Free method\n") +
            "Attributes:\n\t" + ', '.join(c for c in self.attributes ) + 
            "\nRuntime: {:2.4f} seconds".format(self.execution_time)
        )

    @abstractmethod
    def _fit(self, bloom_filters: dict) -> None: 
        pass

    @abstractmethod
    def _block_record(self, bloom: bitarray) -> List[str]: 
        pass

    # @abstractmethod
    # def _clean_blocks(self, blocks: dict) -> dict:
    #     pass

    # @abstractmethod
    # def _configuration(self) -> dict:
    #     pass    



class LSHBlocker(AbstractBlockBuilding):
    """
    LSH-based blocking for Bloom-filter-encoded records.
    Each record produces Λ blocking keys of length Ψ bits.
    """
    def __init__(
        self,
        psi: int = 36,
        lmbda: int = 3,
        prune_ratio: float = 0.5,
        prune_sample: int = 1000,
        seed: int = 42
    ):
        """
        :param psi: number of bit positions per key (Ψ).
        :param lmbda: number of keys per record (Λ).
        :param prune_ratio: ratio of most frequent/uncommon bit positions to prune.
        :param prune_sample: number of records to sample for frequency estimation.
        """
        super().__init__(seed=seed)
        self.rng : random.Random
        self.psi = psi
        self.lmbda = lmbda
        self.prune_ratio = prune_ratio
        self.prune_sample = prune_sample
        self.bit_positions = None  # will hold usable bit indices

    def _select_bit_positions(self, bflist: List[bitarray]) -> List[int]:
        """
        Determine a candidate pool of bit positions excluding too frequent or rare bits.
        Frequency estimated from initial sample.
        """
        m = len(bflist[0])
        freq = np.zeros(m, dtype=int)
        sample = bflist if len(bflist) <= self.prune_sample else bflist[: self.prune_sample]
        for bf in sample:
            freq += np.array(bf.tolist(), dtype=int)
        
        # normalize to frequency ratio
        freq_ratio = freq / len(sample)
        mask = (freq_ratio > self.prune_ratio) | (freq_ratio < (1 - self.prune_ratio))
        # candidate positions are those not masked

        candidates = [i for i, bad in enumerate(mask) if not bad]

        if len(candidates) < self.psi:
            raise ValueError("Not enough bit positions after pruning")
        return candidates

    def _fit(self, bloom_filters : Dict[int, Dict[str, Bloom]]):

        bitarray_list: List[bitarray] = [
            bloom.bitarray for bloom_dict in bloom_filters.values()
            for bloom in bloom_dict.values()
        ]
        """Initialize blocker by sampling to prune and selecting usable bit positions."""
        self.rng = random.Random(self.seed)
        self.bit_positions = self._select_bit_positions(bitarray_list)

    def _block_record(self, bf: bitarray) -> List[str]:
        """
        Generate Λ blocking keys for a single Bloom filter record.
        Each key is Ψ bits sampled from bf at selected positions.
        """
        if self.bit_positions is None:
            raise RuntimeError("Call fit() before blocking")
        keys = []
        for i in range(self.lmbda):
            positions = self.rng.sample(self.bit_positions, self.psi)
            bits = ''.join('1' if bf[j] else '0' for j in positions)
            # optionally, can hash this bit string to reduce size
            keys.append(bits)
        return keys

    
class BitBlocker(AbstractBlockBuilding):

    def __init__(self, 
            psi: int = 36,
            lmbda: int = 3,
            seed : int = 42):
        super().__init__(seed=seed)
        if psi < 1 or lmbda < 1 :
            raise ValueError(f"Both Values psi and lmbda must be positive numbers : {psi} , {lmbda}")

        self.psi = psi
        self.lmbda = lmbda

    def _fit(self, bloom_filters: Dict[int, Dict[str, Bloom]]) -> None:
        _, bloom_dicts = next(iter(bloom_filters.items()))
        _, bloom  = next(iter(bloom_dicts.items()))

        self.hash_len = len(bloom.bitarray)

        self.rng = random.Random(self.seed)
        print(self.hash_len)
        self.hash_indices = tuple(self.rng.sample(range(self.hash_len), self.psi)
                                  for _ in range(self.lmbda))

        return         
    
    def _block_record(self, bf: bitarray) -> List[int]:
        block_keys = list()
        for i, table_indices in enumerate(self.hash_indices):
            vals = map(bf.__getitem__, table_indices)
            table_block = sum(b << j for j, b in enumerate(vals))
            block_keys.append(table_block * len(self.hash_indices) + i)

        return block_keys


## JACCARD BLOCKER (?)

class ContinuousBlocker(AbstractBlockBuilding):

    def __init__(self, 
            radius : float = 0.5):
        super().__init__()
        if radius <= 0.0:
            raise ValueError(f"radius must be positive number : {radius}")
        self.radius = radius



    def _fit(self, bloom_filters: Dict[int, Dict[str, Bloom]]) -> None:
        return         
    
    def _block_record(self, bf: bitarray) -> List[int]:
        block_keys = list()
        for i, table_indices in enumerate(self.hash_indices):
            vals = map(bf.__getitem__, table_indices)
            table_block = sum(b << j for j, b in enumerate(vals))
            block_keys.append(table_block * len(self.hash_indices) + i)

        return block_keys