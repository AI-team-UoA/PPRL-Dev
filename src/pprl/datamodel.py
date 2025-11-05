"""Datamodel of pprl.
"""
import pandas as pd
from pandas import DataFrame
from abc import ABC, abstractmethod
from tqdm import tqdm
from collections import defaultdict
import nltk
from nltk.corpus import stopwords
import re
from ordered_set import OrderedSet
from typing import Dict, List, Tuple
from pybloom_live import BloomFilter as Bloom
import pickle
from bitarray import bitarray 
from collections import defaultdict


class PPRLFeature(ABC):

    _method_name: str
    _method_info: str
    _method_short_name: str

    def __init__(self) -> None:
        super().__init__()
        self._progress_bar: tqdm
        self.execution_time: float
        self.tqdm_disable: bool

    # @abstractmethod
    # def _configuration(self) -> dict:
    #     pass

    # @abstractmethod
    # def evaluate(self,
    #              prediction=None,
    #              export_to_df: bool = False,
    #              export_to_dict: bool = False,
    #              with_classification_report: bool = False,
    #              verbose: bool = True) -> any:
    #     pass
    
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
        parameters = ("\n" + ''.join(['\t{0}: {1}\n'.format(k, v) for k, v in self._configuration().items()])) \
                        if len(self._configuration().items()) != 0 else ' None'
        print(
            "Method name: " + self._method_name +
            "\nMethod info: " + self._method_info +
            "\nParameters: " + parameters +
            "\nRuntime: {:2.4f} seconds".format(self.execution_time)
        )

    def _configuration(self)-> dict:
        pass
    
    @abstractmethod
    def stats(self) -> None:
        pass

# class Data: 
#     """The corpus of the dataset that will be processed with PPRL. \
#         Contains all the information of the dataset and will be passed to each step \
#         of the PPRL workflow.
#     """

#     def __init__(
#                 self,
#                 dataset_1: DataFrame,
#                 id_column_name_1: str,                
#                 dataset_2: DataFrame,
#                 id_column_name_2: str,
#                 attributes_1: list = None,
#                 dataset_name_1: str = None,
#                 attributes_2: list = None,
#                 dataset_name_2: str = None,
#                 ground_truth: DataFrame = None,
#                 skip_ground_truth_processing: bool = False
#     ) -> None:
#         # Original Datasets as pd.DataFrame
#         if isinstance(dataset_1, pd.DataFrame):
#             self.dataset_1 = dataset_1
#         else:
#             raise AttributeError("Dataset 1 must be a pandas DataFrame")

#         if dataset_2 is not None:
#             if id_column_name_2 is None:
#                 raise AttributeError("Must provide datasets 2 id column")

#             if isinstance(dataset_2, pd.DataFrame):
#                 self.dataset_2 = dataset_2
#             else:
#                 raise AttributeError("Dataset 2 must be a pandas DataFrame")

#         # Processed dataframes to lists (all attribute columns)
#         # Tranformed to list for optimization (list)
#         self.entities_d1: list
#         self.entities_d2: list = None

#         # D1 and D2 dataframes concatenated
#         self.entities: DataFrame

#         # Datasets specs
#         self.is_dirty_er = dataset_2 is None
#         self.dataset_limit = self.num_of_entities_1 = len(dataset_1)
#         self.num_of_entities_2: int = len(dataset_2) if dataset_2 is not None else 0
#         self.num_of_entities: int = self.num_of_entities_1 + self.num_of_entities_2

#         self.id_column_name_1 = id_column_name_1
#         self.id_column_name_2 = id_column_name_2
        
#         self.dataset_name_1 = dataset_name_1
#         self.dataset_name_2 = dataset_name_2
        
#         # Fill NaN values with empty string
#         self.dataset_1.fillna("", inplace=True)
#         self.dataset_1 = self.dataset_1.astype(str)
#         if not self.is_dirty_er:
#             self.dataset_2.fillna("", inplace=True)
#             self.dataset_2 = self.dataset_2.astype(str)
            
#         # Attributes
#         if attributes_1 is None:
#             if dataset_1.columns.values.tolist():
#                 self.attributes_1 = dataset_1.columns.values.tolist()
#                 if self.id_column_name_1 in self.attributes_1:
#                     self.attributes_1.remove(self.id_column_name_1)
#             else:
#                 raise AttributeError(
#                     "Dataset 1 must contain column names if attributes_1 is empty.")
#         else:
#             self.attributes_1: list = attributes_1

#         if dataset_2 is not None:
#             if attributes_2 is None:
#                 if dataset_2.columns.values.tolist():
#                     self.attributes_2 = dataset_2.columns.values.tolist()
#                     if self.id_column_name_2 in self.attributes_2:
#                         self.attributes_2.remove(self.id_column_name_2)
#                 else:
#                     raise AttributeError("Dataset 2 must contain column names if attributes_2 is empty.")
#             else:
#                 self.attributes_2: list = attributes_2

#         # Ground truth data
#         self.skip_ground_truth_processing = skip_ground_truth_processing
#         if ground_truth is not None and not skip_ground_truth_processing:
#             self.ground_truth = ground_truth.astype(str)
#             self.ground_truth.drop_duplicates(inplace=True)
#             self._ids_mapping_1: dict
#             self._gt_to_ids_reversed_1: dict
#             self._ids_mapping_2: dict
#             self._gt_to_ids_reversed_2: dict
#         else:
#             self.ground_truth = None
        
#         self.entities = self.dataset_1 = self.dataset_1.astype(str)
        
#         # Concatenated columns into new dataframe
#         self.entities_d1 = self.dataset_1[self.attributes_1]
        
#         if not self.is_dirty_er:
#             self.dataset_2 = self.dataset_2.astype(str)
#             self.entities_d2 = self.dataset_2[self.attributes_2]
#             self.entities = pd.concat([self.dataset_1, self.dataset_2],
#                                         ignore_index=True)
#         # if not skip_ground_truth_processing:
#         self._create_gt_mapping()
#         if ground_truth is not None:
#             if skip_ground_truth_processing:
#                 self.ground_truth = ground_truth
#             else:
#                 self._store_pairs()
#         else:
#             self.ground_truth = None
#         # self.bloom_filters : Dict[int, Dict[str, Bloom]] = None
    
#     def _store_pairs(self) -> None:
#         """Creates a mapping:
#             - pairs_of : ids of first dataset to ids of true matches from second dataset"""
#         self.duplicate_of = defaultdict(set)
        
#         for _, row in self.ground_truth.iterrows():
#             id1, id2 = (row.iloc[0], row.iloc[1])
#             if id1 in self.duplicate_of: self.duplicate_of[id1].add(id2)
#             else: self.duplicate_of[id1] = {id2}
            
#     def _are_true_positives(self, id1 : int, id2 : int):
#         """Checks if given pair of identifiers represents a duplicate.
#             Identifiers must be inorder, first one belonging to the first and the second to the second dataset

#         Args:
#             id1 (int, optional): Identifier from the first dataframe. 
#             id2 (int, optional): Identifier from the second dataframe.

#         Returns:
#             _type_: _description_
#         """
#         return id1 in self.duplicate_of and id2 in self.duplicate_of[id1]

#     def _create_gt_mapping(self) -> None:
#         """Creates two mappings:
#             - _ids_mapping_X: ids from initial dataset to index
#             - _gt_to_ids_reversed_X (inversed _ids_mapping_X): index number \
#                             from range to initial dataset id
#         """
#         if self.ground_truth is not None:
#             self.ground_truth = self.ground_truth.astype(str)
#         # else:
#         #     return
#         self._ids_mapping_1 = dict(
#             zip(
#                 self.dataset_1[self.id_column_name_1].tolist(),
#                 range(0, self.num_of_entities_1)
#             )
#         )

#         self._gt_to_ids_reversed_1 = dict(
#             zip(
#                 self._ids_mapping_1.values(),
#                 self._ids_mapping_1.keys()
#             )
#         )

#         if not self.is_dirty_er:
#             self._ids_mapping_2 = dict(
#                 zip(
#                     self.dataset_2[self.id_column_name_2].tolist(),
#                     range(self.num_of_entities_1, self.num_of_entities_1+self.num_of_entities_2)
#                 )
#             )

#             self._gt_to_ids_reversed_2 = dict(
#                 zip(
#                     self._ids_mapping_2.values(),
#                     self._ids_mapping_2.keys()
#                 )
#             )
            
#     def get_pyjedai_id_of(self, dataset_id: any) -> int:
#         pass

#     def get_real_id_of(self, pyjedai_id: int) -> any:
#         if pyjedai_id < self.dataset_limit:
#             return self._gt_to_ids_reversed_1[pyjedai_id]
#         else:
#             return self._gt_to_ids_reversed_2[pyjedai_id]
        
#     def print_specs(self) -> None:
#         """Dataset report.
#         """
#         def calculate_memory_usage_of_pandas(dataframe: pd.DataFrame) -> float:
#             memory_usage = dataframe.memory_usage(deep=True).sum()
#             if memory_usage > 1024**4:
#                 memory_usage /= (1024**4)
#                 unit = "TB"
#             elif memory_usage > 1024**3:
#                 memory_usage /= (1024**3)
#                 unit = "GB"
#             elif memory_usage > 1024**2:
#                 memory_usage /= (1024**2)
#                 unit = "MB"
#             elif memory_usage > 1024:
#                 memory_usage /= (1024)
#                 unit = "KB"
#             else:
#                 unit = "B"
            
#             return memory_usage, unit

#         print('*' * 123)
#         print(' ' * 50, 'Data Report')
#         print('*' * 123)
#         print("Type of Entity Resolution: ", "Dirty" if self.is_dirty_er else "Clean-Clean" )
#         name1 = self.dataset_name_1 if self.dataset_name_1 is not None else "D1"
#         print("Dataset 1 (" + name1 + "):")
#         print("\tNumber of entities: ", self.num_of_entities_1)
#         print("\tNumber of NaN values: ", self.dataset_1.isnull().sum().sum())
#         memory_usage, unit = calculate_memory_usage_of_pandas(self.dataset_1)
#         print("\tMemory usage [" + unit + "]: ", "{:.2f}".format(memory_usage))
#         print("\tAttributes:")
#         for attr in self.attributes_1:
#             print("\t\t", attr)
#         if not self.is_dirty_er:
#             name2 = self.dataset_name_2 if self.dataset_name_2 is not None else "D2"
#             print("Dataset 2 (" + name2 + "):")
#             print("\tNumber of entities: ", self.num_of_entities_2)
#             print("\tNumber of NaN values: ", self.dataset_2.isnull().sum().sum())
#             memory_usage, unit = calculate_memory_usage_of_pandas(self.dataset_2)
#             print("\tMemory usage [" + unit + "]: ", "{:.2f}".format(memory_usage))
#             print("\tAttributes:")
#             for attr in self.attributes_2:
#                 print("\t\t", attr)
#         print("\nTotal number of entities: ", self.num_of_entities)
#         if self.ground_truth is not None:
#             print("Number of matching pairs in ground-truth: ", len(self.ground_truth))
#         print(u'\u2500' * 123)

#     # Functions that removes stopwords, punctuation, uni-codes, numbers from the dataset
#     def clean_dataset(self, 
#                         remove_stopwords: bool = True, 
#                         remove_punctuation: bool = True, 
#                         remove_numbers:bool = True,
#                         remove_unicodes: bool = True) -> None:
#         """Removes stopwords, punctuation, uni-codes, numbers from the dataset.
#         """
#         nltk.download('stopwords')

#         # Make self.dataset_1 and self.dataset_2 lowercase
#         self.dataset_1 = self.dataset_1.applymap(lambda x: x.lower())
#         if not self.is_dirty_er:
#             self.dataset_2 = self.dataset_2.applymap(lambda x: x.lower())
            
#         if remove_numbers:
#             self.dataset_1 = self.dataset_1.applymap(lambda x: re.sub(r'\d+', '', x))
#             if not self.is_dirty_er:
#                 self.dataset_2 = self.dataset_2.applymap(lambda x: re.sub(r'\d+', '', x))    
                
#         if remove_unicodes:
#             self.dataset_1 = self.dataset_1.applymap(lambda x: re.sub(r'[^\x00-\x7F]+', '', x))
#             if not self.is_dirty_er:
#                 self.dataset_2 = self.dataset_2.applymap(lambda x: re.sub(r'[^\x00-\x7F]+', '', x))
            
#         if remove_punctuation:
#             self.dataset_1  = self.dataset_1.applymap(lambda x: re.sub(r'[^\w\s]','',x))
#             if not self.is_dirty_er:
#                 self.dataset_2 = self.dataset_2.applymap(lambda x: re.sub(r'[^\w\s]','',x))
        
#         if remove_stopwords:
#             self.dataset_1 = self.dataset_1.applymap(lambda x: ' '.join([word for word in x.split() if word not in (stopwords.words('english'))]))
#             if not self.is_dirty_er:
#                 self.dataset_2 = self.dataset_2.applymap(lambda x: ' '.join([word for word in x.split() if word not in (stopwords.words('english'))]))    

#         self.entities = self.dataset_1 = self.dataset_1.astype(str)
        
#         # Concatenated columns into new dataframe
#         self.entities_d1 = self.dataset_1[self.attributes_1]

#         if not self.is_dirty_er:
#             self.dataset_2 = self.dataset_2.astype(str)
#             self.entities_d2 = self.dataset_2[self.attributes_2]
#             self.entities = pd.concat([self.dataset_1, self.dataset_2],
#                                         ignore_index=True)

#     def stats_about_data(self) -> None:
        
#         stats_df = pd.DataFrame(columns=['word_count_1', 'word_count_2'])
        
#         # Calculate the average number of words per line
#         stats_df['word_count_1'] = self.dataset_1.apply(lambda row: len(row.str.split()), axis=1)
#         print(stats_df['word_count_1'])
#         average_words_per_line_1 = stats_df['word_count_1'].mean()
#         print(average_words_per_line_1)
        
#         if not self.is_dirty_er:
#             stats_df['word_count_2'] = self.dataset_2.apply(lambda row: len(row.str.split()), axis=1)
#             average_words_per_line_2 = stats_df['word_count_2'].mean()
#             print(average_words_per_line_2)
            
#         return stats_df

class Block:
    """The main module used for storing entities in the blocking steps of pyjedai module. \
        Consists of 2 sets of profile entities 1 for Dirty ER and 2 for Clean-Clean ER.
    """
    def __init__(self) -> None:
        self.entities : Dict[str, OrderedSet] = defaultdict(OrderedSet)

    # def get_cardinality(self, is_dirty_er) -> int:
    #     """Returns block cardinality.

    #     Args:
    #         is_dirty_er (bool): Dirty or Clean-Clean ER.

    #     Returns:
    #         int: Cardinality
    #     """
    #     if is_dirty_er:
    #         return len(self.entities_D1)*(len(self.entities_D1)-1)/2
    #     return len(self.entities_D1) * len(self.entities_D2)

    # def get_size(self) -> int:
    #     """Returns block size.

    #     Returns:
    #         int: Block size
    #     """
    #     return len(self.entities_D1) + len(self.entities_D2)

    # def verbose(self, key: any, is_dirty_er: bool) -> None:
    #     """Prints contents of a block

    #     Args:
    #         key (any): Block key
    #         is_dirty_er (bool): Dirty or Clean-Clean ER.
    #     """
    #     print("\nBlock ", "\033[1;32m"+key+"\033[0m", " has cardinality ", str(self.get_cardinality(is_dirty_er)) ," and contains entities with ids: ")
    #     if is_dirty_er:
    #         print("Dirty dataset: " + "[\033[1;34m" + \
    #          str(len(self.entities_D1)) + " entities\033[0m]")
    #         print(self.entities_D1)
    #     else:
    #         print("Clean dataset 1: " + "[\033[1;34m" + \
    #          str(len(self.entities_D1)) + " entities\033[0m]")
    #         print(self.entities_D1)
    #         print("Clean dataset 2: " + "[\033[1;34m" + str(len(self.entities_D2)) + \
    #         " entities\033[0m]")
    #         print(self.entities_D2)
    
    def __str__(self): 
        return f"{dict(self.entities)}"
    # f"D1: {self.entities_D1} D2: {self.entities_D2}"
    
class EncodedData(ABC): 
    encoded_dict: Dict[int, Dict[str, any]]
    bounds : List
    skip_ground_truth : bool
    ground_truth: pd.DataFrame
    
    def get_cardinality(self) -> int:
        return len(self.encoded_dict)
    
    @abstractmethod
    def to_file(self, filename: str):
        pass
    
    @classmethod
    @abstractmethod
    def from_file(self, *kwargs): 
        pass

    

class BloomEncodedData(EncodedData): 

    dataset_limit : int 
    bounds : List[int]
    inverted_index : Dict[int, Dict[str, set]]
    """reverse_index :   position_of_bit : entities {D0: x,y,z,...} and {D1: x2,y2,z2,...}"""

    def __init__(self, data: Dict[int, Dict[str, List[int]]] = None, length: int = 0):
        self.inverted_index = None
        self.skip_ground_truth = True
        if data: 
            self.length = length
            self.encoded_dict : Dict[int, Dict[str, List[int]]] = data
            if self.encoded_dict:
                self.bitarray_dict : Dict[int, Dict[str, bitarray]] = dict()
                for key, value in self.encoded_dict.items(): 
                    self.bitarray_dict[key] = dict()
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
        
    def to_entities(self, attribute: str = None) -> Dict[int, List[int]]:
        to_entities = dict() 
        inverted_index : Dict[int, Dict[int, set]]= dict()
        dataset = 0
        print(self.bounds)
        if attribute: 
            for idx, blooms in self.encoded_dict.items():
                if dataset < len(self.bounds) and idx >= self.bounds[dataset]: 
                    dataset += 1  
                to_entities[idx] = blooms[attribute]
                for bit in blooms[attribute]: 
                    if bit not in inverted_index: 
                        inverted_index[bit] = {key: set() for key in range(len(self.bounds))}
                    print(dataset)
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
                    ret.encoded_dict = {**ret.encoded_dict, 
                                **{key + offset :  value for key, value in obj.encoded_dict.items()} }
                    ret.bitarray_dict = {**ret.bitarray_dict, 
                                **{key + offset :  value for key, value in obj.bitarray_dict.items()} }
                    ret.bounds.append(offset + len(obj.encoded_dict)) 
                    
                    
                    # ret.dataset_indexes[dataset_num] = (offset, len(ret.encoded_dict)-1)

            dataset_num += 1
            
        ret.dataset_limit = offset
        ret.inverted_index = None

        
        # ret._create_inverted_index()
        
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
                    
    def get_inverted_index(self) -> Dict[int, Dict[str, set]]: 
        if not self.inverted_index: 
            self._create_inverted_index()
        
        return self.inverted_index
            
            
        

class HomomorphicEcnodedData(EncodedData): 
    def __init__(self, data: Dict[int, Dict[str, List[int]]]):
        self.encoded_dict : Dict[int, Dict[str, List[int]]] = data
        if self.encoded_dict:
            first_attr = next(iter(self.encoded_dict.values()), {})
            if first_attr:
                self.attributes = list(first_attr.keys())
    
    def __str__(self): 
        return f'{self.encoded_dict}'      


