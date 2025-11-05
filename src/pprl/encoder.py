from .datamodel import PPRLFeature, HomomorphicEcnodedData, BloomEncodedData
from bitarray import bitarray
from typing import List, Literal, Union, Dict, Generator, Set, Optional
# from pybloom_live import BloomFilter as Bloom
import pandas as pd
import hashlib
import re
from metaphone import doublemetaphone



# Differnece between errorate and multiple_hashes

class BloomFilter:
    VALID_HASHING_TYPES = [
        "salted_string","salted_qgrams",
        "salted_skipqgrams", "salted_metaphone",    
        "salted_tokens"
    ]        

    def __init__(self,
        size : int = 1024,
        num_hashes: int = 20,
        offset : int = 0,
        # error_rate : float = 0.001,
        hashing_type : Literal["salted_string", "salted_qgrams", 
                               "salted_skipqgrams", "salted_metaphone",
                               "salted_tokens"] = "salted_string", 
        qgrams : Optional[int] = 5,
        pad : bool = False,
        salt : str = "",
        attributes : List[str] = None,
    ):
        
        encoder_dict = { 
            "salted_string" : self._salted_string,
            "salted_qgrams" : self._salted_qgrams,
            "salted_skipqgrams" : self._salted_skipqgrams,
            "salted_metaphone" : self._salted_metaphone,    
            "salted_tokens" : self._salted_token,        
        }
        
        if hashing_type not in self.VALID_HASHING_TYPES:
            raise KeyError(
                f"Invalid hashing_type '{hashing_type}'. "
                f"Allowed values are: {', '.join(self.VALID_HASHING_TYPES)}"
            )
        
        self.size : int  = size
        self.num_hashes : int = num_hashes
        self.offset : int = offset 
        
        # self.error_rate : float = error_rate
        self.hashing_type = hashing_type
        self.qgrams = qgrams
        self.pad = pad
        self.salt = salt
        # self.seeds = seeds
        self.fields = attributes
        # self.num_hashes = num_hashes
        # self.hash_seeds = seeds if seeds else range(num_hashes)
        self.attributes = attributes
        self.print = False
        self._hash = encoder_dict[hashing_type] 
        
   

    def _create_bloom_dict_for_row(self, row : pd.Series) -> Dict[str, List[int]]:
        bloom_dict = dict()
        if self.attributes: 
            for attribute in self.attributes: 
                bloom_dict[attribute] = self._hash(str(row[attribute]))
        else: 
            flatten_row : list = row.values.tolist()
            flatten_row = " ".join(str(x) for x in flatten_row) 
            bloom_dict['row'] = self._hash(flatten_row)        
        return bloom_dict
    
    
    def _encode(self, df: pd.DataFrame) -> BloomEncodedData: 
        bloom_dict = dict()
        for id, (_, row) in enumerate(df.iterrows()): 
            bloom_dict[id] = self._create_bloom_dict_for_row(row)
        return BloomEncodedData(data = bloom_dict, length=self.size + self.offset, )  #leng
    
    def _hash_token(self, bf: Set[int],  salted_token: str, hash_token : bool = True): 
        # HASH STRING or HASH QGRAM
        
        # HASH STRING here

        if isinstance(bf, list): 
            bf = set(bf)

        if hash_token: 
            hashed = hashlib.sha256(salted_token).digest()
            for i in range(self.num_hashes):
                start_idx = i * 4 % (len(hashed) - 4)
                segment = hashed[start_idx:start_idx + 4]
                position = (int.from_bytes(segment, 'little') % self.size) + self.offset
                bf.add(position)
        else: 
            for i in range(len(salted_token) - self.qgrams + 1):
                qgram_token = salted_token[i:i+self.qgrams]
                if len(qgram_token) < self.qgrams and self.pad:
                    qgram_token = qgram_token + '#'*(self.qgrams - len(qgram_token))

                hashed = hashlib.sha256(qgram_token).digest()
                for i in range(self.num_hashes):
                    start_idx = i * 4 % (len(hashed) - 4)
                    segment = hashed[start_idx:start_idx + 4]
                    position = (int.from_bytes(segment, 'little') % self.size) + self.offset
                    bf.add(position)
                # position = (int.from_bytes(hashed, 'little') % self.size) + self.offset
                # bf.add(position)
        return [*bf]
    
    
    def _salted_string(self, str_to_enc: str) -> List[int]: 
        # bf = Bloom(capacity=self.size, error_rate=self.error_rate)
        
        bf = set()
        self.pad = False
        for i in range(self.num_hashes):
            salted_string = (self.salt + str(i) +  str_to_enc).encode("utf-8")
            bf = self._hash_token(bf, salted_string, True)

        return [*bf]
    
    def _salted_qgrams(self, str_to_enc: str) -> List[int]:
        tokens : List[str] = self._tokenize(str_to_enc)
        bf = set()
        for token in tokens: 
            for i in range(self.num_hashes):
            # salted_string = (self.salt + str(i) +  str_to_enc).encode("utf-8")
                salted_token : str = (self.salt + str(i) + token).encode("utf-8")
                bf = self._hash_token(bf, salted_token, False)                   # HASH QGRAM
        
        return [*bf]

    def _salted_skipqgrams(self, str_to_enc: str) -> List[int]:
        """" Gets two different strings one without the chars in the odd positions and ove without the even ones.
        If the string is shorter of qgrams, # is added for creating the qgrams """
    
        bf = set()
    
        tokens = self._tokenize(str_to_enc)
        for token in tokens: 
            for i in range(self.num_hashes):
                salted_token: str = (self.salt + str(i) +  token) 
                skipped_tokens : List[str] = [salted_token[1::2], salted_token[::2]]
                for skipped_token in skipped_tokens:
                    utf_8_token = skipped_token.encode('utf-8')
                    bf = self._hash_token(bf, utf_8_token, False)                   # HASH QGRAM
    
        return [*bf]

    def _salted_metaphone(self, str_to_enc: str) -> List[int]:
        """" Hashes metaphone if it exists else hashes token """
        bf = set()
        # bf = Bloom(capacity=self.size, error_rate=self.error_rate)
        tokens = self._tokenize(str_to_enc)
        for token in tokens: 
            for i in range(self.num_hashes):
                salted_token : str = (self.salt + str(i) + token) 
                double_metaphone = doublemetaphone(salted_token)
                
                if len(double_metaphone[0]) == 0 and len(double_metaphone[1])  == 0: 
                    utf_8_token = salted_token.encode('utf-8')
                    bf = self._hash_token(bf, utf_8_token, False)                   # HASH QGRAM
                    continue
                
                for metaphone in double_metaphone:
                    if '' == metaphone: 
                        continue
                    hashed = hashlib.sha256(metaphone.lower().encode('utf-8')).digest()
                    digest_as_int = (int.from_bytes(hashed, 'little') % self.size )  + self.offset
                    bf.add(digest_as_int)
        
        return [*bf]
    
    
    def _salted_token(self, str_to_enc: str) -> List[int]:
        """" Hashes tokens and extract from the hashed the qgrams """
        bf = set()
        tokens = self._tokenize(str_to_enc)
        for token in tokens: 
            for i in range(self.num_hashes):
                salted_token = (self.salt + str(i) + token).encode('utf-8') 
                bf = self._hash_token(bf, salted_token, True)

        return [*bf]


    def  _tokenize(self, str_to_enc : str) -> List:
        return list(set(filter(None, re.split('[\\W_]', str_to_enc.lower()))))
    
    
    
    def encode(self, df: pd.DataFrame) -> BloomEncodedData:
        columns = list(df.columns)

        if self.attributes:
            for col in self.attributes: 
                if col not in columns: 
                    raise ValueError("Attributes must be the same between the two dataframes")
        else: 
            self.attributes = columns
        print(f'Encrypting these attributes: {self.attributes}')

        df = df[self.attributes] if self.attributes else df
        return self._encode(df)

    
    
# class HomomorphicEncryption: 
#     def __init__(self,
#         qgrams : int = 2,
#         pad : bool = True,
#         salt : str = "",
#         attributes : List[str] = None,
#     ):
#         self.qgrams : int = qgrams
#         self.pad : bool = pad
#         self.salt : str = salt
#         self.attributes : List[str] = attributes
    
#     def _hash(self, str_to_enc: str) -> List[int]: 
#         tokens = list(set(filter(None, re.split('[\\W_]', str_to_enc.lower()))))
#         qgrams_set = set()
#         for token in tokens: 
#             salted_token = self.salt + token
#             # if len(salted_token) < self.qgrams and self.pad:
#             #     salted_token = salted_token + '#'*(self.qgrams - len(salted_token))
              
#             for i in range(len(salted_token) - self.qgrams + 1):
#                 qgram_token = str(salted_token[i: i+self.qgrams]) 
#                 if len(qgram_token) < self.qgrams and self.pad:
#                     qgram_token = qgram_token + '#'*(self.qgrams - len(qgram_token))
#                 qgrams_set.add(qgram_token)
#         vector_uint64 : List[int] = []
#         for token in list(qgrams_set): 
#             uint64 : int = 0
#             for idx, c in enumerate(reversed(token)):
#                 uint64 += (ord(c) & 0xff) << (8 * idx)
#             vector_uint64.append(uint64)
            
#         return vector_uint64 

#     def _create_vector_for_row(self, row : pd.Series) -> Dict[str, List[int]]:
#         homomorphic_vector = dict()
#         if self.attributes: 
#             for attribute in self.attributes:
#                 vector = self._hash(str(row[attribute]))
#                 if len(vector) > 0: 
#                     homomorphic_vector[attribute] = vector

#         else: 
#             flatten_row : list = row.values.tolist()
#             flatten_row = " ".join(str(x) for x in flatten_row) 
#             homomorphic_vector['row'] = self._hash(flatten_row)        
#         return homomorphic_vector
    
#     def _encode(self) -> HomomorphicEcnodedData:
#         df_1 : pd.DataFrame = self.data.dataset_1[self.attributes] if self.attributes else self.data.dataset_1
#         df_2 : pd.DataFrame = self.data.dataset_2[self.attributes] if self.attributes else self.data.dataset_2
#         homomorphic_vector = dict()
#         for id, (_, row) in enumerate(df_1.iterrows()): 
#             homomorphic_vector[id] = self._create_vector_for_row(row)
#         for id, (_, row) in enumerate(df_2.iterrows(), start = self.data.dataset_limit): 
#             homomorphic_vector[id] = self._create_vector_for_row(row)
        
#         return HomomorphicEcnodedData(data=homomorphic_vector)

#     def encode(self, data: Data) -> HomomorphicEcnodedData : 
#         self.data = data
#         columns_1 = list(data.dataset_1.columns)
#         columns_2 = list(data.dataset_2.columns) 

#         if self.attributes:
#             for col in self.attributes: 
#                 if col not in columns_1 or col not in columns_2: 
#                     raise ValueError("Attributes must be the same between the two dataframes")
#         else: 
#             upperset : set = set(columns_1) if len(set(columns_1)) > len(set(columns_2)) else set(columns_2)
#             subset : set = set(columns_1) if len(set(columns_1)) <= len(set(columns_2)) else set(columns_2)
#             self.attributes = list(subset) if subset.issubset(upperset) else None
#         print(f'Encrypting these attributes: {self.attributes}')
        
#         return self._encode() 
