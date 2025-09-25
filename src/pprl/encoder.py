from .datamodel import PPRLFeature, Data
from bitarray import bitarray
from typing import List, Literal, Union, Dict, Generator
from pybloom_live import BloomFilter as Bloom
import pandas as pd
import hashlib



# Differnece between errorate and multiple_hashes

class BloomFilter:

    def __init__(self,
        size : int = 1024,
        error_rate : float = 0.001,
        hashing_type : Literal["salted_string", "salted_qgrams"] = "salted_string", 
                            #    "seeded_qgrams"] 
        qgrams : int = 5,
        pad : bool = False,
        salt : str = "",
        # num_hashes = 20,
        # seeds : List[int] = None,
        attributes : List[str] = None,
    ):
        
        encoder_dict = { 
            "salted_string" : self._salted_string,
            "salted_qgrams" : self._salted_qgrams,
            # "seeded_qgrams" : self._seeded_qgrams
        }
        
        self.bf_a : Union[List[bitarray], List[Bloom]] = None
        self.bf_b : Union[List[bitarray], List[Bloom]] = None
        self.size : int  = size
        self.error_rate : float = error_rate
        self.hashing_type = hashing_type
        self.qgrams = qgrams
        self.pad = pad
        self.salt = salt
        # self.seeds = seeds
        self.fields = attributes
        # self.num_hashes = num_hashes
        # self.hash_seeds = seeds if seeds else range(num_hashes)
        self.attributes = attributes
        
        self._hash = encoder_dict[hashing_type] 
        
    @staticmethod
    def record_to_string(row):
        return " ".join(str(row))
    
    # def _qgrams(self, s : str):
    #     if self.pad: 
    #         s = '#' * (self.qgrams - 1) + s + '#' * (self.qgrams - 1)
    #     return (s[i : i + self.qgrams] for i in range(len(s) - self.qgrams + 1))
    
    # def _hashes(self, gram: str) -> List[int]:
    #     """Generate `num_hashes` indices from a q-gram."""
    #     return [
    #         int(hashlib.sha256(f"{gram}{seed}".encode()).hexdigest(), 16) % self.size
    #         for seed in self.hash_seeds
    #     ]

    def _create_sub_bloom_dict(self, row : pd.Series) -> Dict[str, Bloom]:
        bloom_dict = dict()
        if self.attributes: 
            for attribute in self.attributes: 
                bloom_dict[attribute] = self._hash(str(row[attribute]))
        else: 
            flatten_row : list = row.values.tolist()
            flatten_row = " ".join(str(x) for x in flatten_row) 
            bloom_dict['row'] = self._hash(flatten_row)        
        return bloom_dict
    
    
    def _encode(self) -> Dict[int, Dict[str, Bloom]]: 
        df_1 : pd.DataFrame = self.data.dataset_1[self.attributes] if self.attributes else self.data.dataset_1
        df_2 : pd.DataFrame = self.data.dataset_2[self.attributes] if self.attributes else self.data.dataset_2
        bloom_dict = dict()
        for id, (_, row) in enumerate(df_1.iterrows()): 
            bloom_dict[id] = self._create_sub_bloom_dict(row)
        for id, (_, row) in enumerate(df_2.iterrows(), start = self.data.dataset_limit): 
            bloom_dict[id] = self._create_sub_bloom_dict(row)
        return bloom_dict
    
    def _salted_string(self, str_to_enc: str) -> Bloom: 
        salted = (self.salt + str_to_enc).encode("utf-8")
        hashed = hashlib.sha256(salted).hexdigest()
        hashed_qgram = hashed.lower().strip()
        bf = Bloom(capacity=self.size, error_rate=self.error_rate)
        for i in range(len(hashed_qgram) - self.qgrams + 1):
            bf.add(hashed_qgram[i:i+self.qgrams])
        return bf
        
    def _salted_qgrams(self, str_to_enc: str) -> Bloom:
        str_to_enc = str_to_enc.encode("utf-8")
        bf = Bloom(capacity=self.size, error_rate=self.error_rate)
        for i in range(len(str_to_enc) - self.qgrams + 1):
            salted_gram = self.salt + str_to_enc[i:i+self.qgrams].encode('utf-8') 
            gram = hashlib.sha256(salted_gram).hexdigest()
            bf.add(gram)
        return bf
    
    # def _encode_seeded_qgrams(self, df: pd.DataFrame) -> List[bitarray]:
    #     bf_list : List[bitarray] = list()
    #     if self.attributes: 
    #         df = df[self.attributes]
        
    #     for _, row in df.iterrows():
    #         bits = bitarray(self.size) 
    #         bits.setall(False)
    #         salted = self.salt + row
    #         for gram in self._qgrams(salted): 
    #             print(f"this is a gram: {gram}")
    #             for idx in self._hashes(gram): 
    #                 bits[idx] = True
    #         bf_list.append(bits)
    #     return bf_list
    
    def encode(self, data : Data) -> Dict[int, Dict[str, Bloom]]:
        self.data = data
        columns_1 = list(data.dataset_1.columns)
        columns_2 = list(data.dataset_2.columns) 

        if self.attributes:
            for col in self.attributes: 
                if col not in columns_1 or col not in columns_2: 
                    raise ValueError("Attributes must be the same between the two dataframes")
        else: 
            upperset : set = set(columns_1) if len(set(columns_1)) > len(set(columns_2)) else set(columns_2)
            subset : set = set(columns_1) if len(set(columns_1)) <= len(set(columns_2)) else set(columns_2)
            self.attributes = list(subset) if subset.issubset(upperset) else None

        return self._encode()
        