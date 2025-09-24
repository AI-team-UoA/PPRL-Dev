from .datamodel import PPRLFeature, Data
from bitarray import bitarray
from typing import List, Literal, Union
from rbloom import Bloom
import pandas as pd
import hashlib


class BloomFilter:

    def __init__(self,
        size : int = 1024,
        error_rate : float = 0.001,
        hashing_type : Literal["salted_string", "salted_qgrams", "seeded_qgrams"] = "salted_string",
        qgrams : int = 5,
        pad : bool = True,
        salt : str = "",
        num_hashes = 20,
        seeds : List[int] = None,
        attributes : List[str] = None,
    ):
        self.bf_a : Union[List[bitarray], List[Bloom]] = None
        self.bf_b : Union[List[bitarray], List[Bloom]] = None
        self.size : int  = size
        self.error_rate : float = error_rate
        self.hashing_type = hashing_type
        self.qgrams = qgrams
        self.pad = pad
        self.salt = salt
        self.seeds = seeds
        self.fields = attributes
        self.num_hashes = num_hashes
        self.hash_seeds = seeds if seeds else range(num_hashes)
        
        self.attributes = attributes
        
    @staticmethod
    def record_to_string(row):
        return " ".join(str(row))
    
    def _qgrams(self, s : str):
        if self.pad: 
            s = '#' * (self.qgrams - 1) + s + '#' * (self.qgrams - 1)
        return (s[i : i + self.qgrams] for i in range(len(s) - self.qgrams + 1))
    
    def _hashes(self, gram: str) -> List[int]:
        """Generate `num_hashes` indices from a q-gram."""
        return [
            int(hashlib.sha256(f"{gram}{seed}".encode()).hexdigest(), 16) % self.size
            for seed in self.hash_seeds
        ]
    
    def _encode_salted_string(self, df : pd.DataFrame) -> List[Bloom]:
        bf_list : List[Bloom] = list()
        if self.attributes: 
            df: pd.DataFrame = df[self.attributes].apply(" ".join, axis=1)
        for _, row in df.iterrows(): 
            row_string = row.values.tolist()[1:]
            row_string = " ".join(row_string)
            salted = (self.salt + row_string).encode("utf-8")
            hashed = hashlib.sha256(salted).hexdigest()
            hashed_qgram = hashed.lower().strip()
            bf = Bloom(self.size, false_positive_rate=self.error_rate)
            for i in range(len(hashed_qgram) - self.qgrams + 1):
                bf.add(hashed_qgram[i:i+self.qgrams])
            bf_list.append(bf)
        return bf_list
    
    def _encode_salted_qgrams(self, df: pd.DataFrame) -> List[Bloom]:
        bf_list : List[Bloom] = list()
        if self.attributes: 
            df: pd.DataFrame = df[self.attributes].apply(" ".join, axis=1) 
    
        for _, row in df.iterrows(): 
            salted = (self.salt + row)
            bf = Bloom(self.size, false_positive_rate=self.error_rate)
            for gram in self._qgrams(salted): 
                bf.add(gram)
            bf_list.append(bf)        
        
        return bf_list
    
    def _encode_seeded_qgrams(self, df: pd.DataFrame) -> List[bitarray]:
        bf_list : List[bitarray] = list()
        if self.attributes: 
            df = df[self.attributes]
        
        for _, row in df.iterrows():
            bits = bitarray(self.size) 
            bits.setall(False)
            salted = self.salt + row
            for gram in self._qgrams(salted): 
                for idx in self._hashes(gram): 
                    bits[idx] = True
            bf_list.append(bits)
        return bf_list
    
    def encode(self, data : Data) -> None:
        encoder_dict = { 
            "salted_string" : self._encode_salted_string,
            "salted_qgrams" : self._encode_salted_qgrams,
            "seeded_qgrams" : self._encode_seeded_qgrams
        }
        encoder = encoder_dict[self.hashing_type]
        
        self.bf_a = encoder(data.dataset_1)
        self.bf_b = encoder(data.dataset_2)