from typing import Literal, List, Callable, Dict, overload, Tuple, Union
from src.pprl.utils import _dice, _scm
import src.pprl.openfhe_similarity
from src.pprl.datamodel import Data, Block, EncodedData, BloomEncodedData, HomomorphicEcnodedData
from pybloom_live import BloomFilter as Bloom
from networkx import Graph
from ordered_set import OrderedSet
import numpy as np
from openfhe import *

class Matcher: 
    
    def __init__(self,
        batch_size : int = 10_000,
        threshold : float = 0.6,
        metric : Literal["dice", "scm", 'jaccard'] = "dice", 
        attributes : List[str] = None,
    ) -> None:
        
        
        self.batch_size : int = batch_size 
        self.threshold : float = threshold
        self.metric : str = metric
        self.attributes : List[str] = attributes
        
        
    def _candidates(self, extract_candidates: Union[Dict[str, Block], EncodedData]) -> Tuple[set, OrderedSet]: 
        candidates = set()
        unique_ids = OrderedSet()

        if isinstance(extract_candidates, dict):
            for _, block in extract_candidates.items():
                for eid1 in block.entities_D1: 
                    for eid2 in block.entities_D2: 
                        candidates.add((eid1,eid2))
                        unique_ids.add(eid1)
                        unique_ids.add(eid2)
        else: 
            data_limit = self.data.dataset_limit
            end_of_dict = len(extract_candidates.encoded_dict) 
            for i in range(data_limit):
                for j in range(data_limit, end_of_dict):
                    candidates.add((i, j))
                    unique_ids.add(i)
                    unique_ids.add(j)
            
        return candidates, unique_ids
    
    @overload
    def predict(self,
            encoded_data: HomomorphicEcnodedData,
            blocks : Dict[str, Block] = None, 
            type_of_match: Literal['overlap', 'extension'] = 'overlap',
            homomorphism : Literal['bgvs'] = 'bgvs') -> Graph:
        """
        Similarity Computaation for Homomorphic Encrypted Data
        
        args:
        
            data : Data
            ecnoded_data: HomomorphicEncodedData
            blocks [optional] : Dictionary of Blocks 
            type_of_match [optional] : {'overlap', ...}
            homomorphism [optional] : {'bgvs', ...}
        """
        ...
       
    @overload        
    def predict(self, 
                encoded_data : BloomEncodedData,
                blocks: Dict[str, Block] = None)  -> Graph: 
        """
        Vectorized  similarity computation for Bloom filters.

        args:
        
            data : Data
            ecnoded_data: HomomorphicEncodedData
            blocks [optional] : Dictionary of Blocks 

        """
        
        ... 
        
        
    def predict(self,
                encoded_data: EncodedData, 
                blocks: Dict[str, Block] = None, 
                **kwargs) -> Graph: 

        self.candidates, self.unique_ids = self._candidates(blocks) if blocks else self._candidates(encoded_data)

        if not self.candidates:
            return Graph()

        if isinstance(encoded_data, BloomEncodedData): 
            return self._predict_from_bloom(encoded_data)
        elif isinstance(encoded_data, HomomorphicEcnodedData):
            return self._predict_from_vector(encoded_data, **kwargs) 
        else: 
            raise TypeError(f"encoded_data must be of type: BloomEncodedData or HomomorphicEcnodedData, but its {type(encoded_data)}")


    def _bgvs(self) -> None:
        parameters = CCParamsBGVRNS()
        sigma = 3.2                       # palisade: sigma
        security_level = SecurityLevel.HEStd_128_classic    

        parameters.SetStandardDeviation(sigma)   # set sigma
        parameters.SetSecurityLevel(security_level)
        parameters.SetPlaintextModulus(65537)  # plaintext_modulus
        parameters.SetMultiplicativeDepth(1)   

        self.cc : CryptoContext = GenCryptoContext(parameters)
        self.cc.Enable(PKESchemeFeature.PKE)  # palisade encryption
        self.cc.Enable(PKESchemeFeature.LEVELEDSHE) #palisade she
        self.cc.Enable(PKESchemeFeature.ADVANCEDSHE) #palisade she
        # Generate a public/private key pair
                                            # Palisade code
        self.key_pair : KeyPair = self.cc.KeyGen()  # LPKeyPair<DCRTPoly> keyPair;
                                            # keyPair = cc->KeyGen();

        # Generate the relinearization key
        self.cc.EvalSumKeyGen(self.key_pair.secretKey)      # cc->EvalSumKeyGen(keyPair.secretKey);
       
            
        

    def _predict_from_vector(self, encoded_data: HomomorphicEcnodedData,
            type_of_match: Literal['overlap', 'extension'] = 'overlap',
            homomorphism : Literal['bgvs'] = 'bgvs' ) -> Graph:
        
        if homomorphism == 'bgvs': 
            self._bgvs()
            
        matching_function = f'_{type_of_match}_jaccard'
        matching_function = getattr(src.pprl.openfhe_similarity, matching_function)

        attributes = self.attributes if self.attributes else encoded_data.attributes

        graph = Graph()
        
        homomorphic_dict = encoded_data.encoded_dict
        
        for id1, id2 in self.candidates: 
            similarity = 0.0
            encoded_1 = homomorphic_dict[id1]
            encoded_2 = homomorphic_dict[id2]
            for attr in attributes: 
                similarity += matching_function(self.cc, self.key_pair,
                                               encoded_1[attr], encoded_2[attr])
            avg_similarity = similarity / len(attributes)
            print(id1, id2, avg_similarity)
            if avg_similarity >= self.threshold: 
               graph.add_edge(id1, id2, weight=avg_similarity)           
        
        return graph
        
    def _predict_from_bloom(self, encoded_data: BloomEncodedData) -> Graph: 
        
        metrics_dict = { 
            "dice" : _dice, 
            "scm" : _scm
        }
       
        self._metric = metrics_dict['dice'] if self.metric not in metrics_dict else metrics_dict[self.metric]        
        
        candidates, unique_ids = self.candidates, self.unique_ids
        bloom_filters = encoded_data.encoded_dict
        attributes = self.attributes if self.attributes else encoded_data.attributes
        bf_length = encoded_data.bitarray_length
        n_attributes = len(attributes)

        id_to_idx = {eid: idx for idx, eid in enumerate(unique_ids)}
        idx_to_id = {idx: eid for idx, eid in enumerate(unique_ids)}
        
        
        # Create matrices: (n_entities, n_attributes, bf_length)
        bloom_matrix = np.zeros((len(unique_ids), n_attributes, bf_length), dtype=np.uint8)
        # Fill the bloom matrix
        for eid_idx, eid in enumerate(unique_ids):
            for attr_idx, attr in enumerate(attributes):
                bitarr = bloom_filters[eid][attr].bitarray
                bloom_matrix[eid_idx, attr_idx] = np.array(bitarr.tolist(), dtype=np.uint8)
        
        # Prepare candidate indices
        idx_pairs = np.array([(id_to_idx[eid1], id_to_idx[eid2]) 
                            for eid1, eid2 in candidates])
        n_candidates = len(idx_pairs)

        graph = Graph()
        cnt = 0
        for start in range(0, n_candidates, self.batch_size):
            cnt += 1
            end = min(start + self.batch_size, n_candidates)
            batch_pairs = idx_pairs[start:end]
            blooms_1 = bloom_matrix[batch_pairs[:,0]]
            blooms_2 = bloom_matrix[batch_pairs[:,1]]
            avg_similarities : np.array = self._metric(blooms_1, blooms_2)
            edges = [(idx_to_id[i], idx_to_id[j], float(sim))
                    for (i,j), sim in zip(batch_pairs, avg_similarities)
                    if sim >= self.threshold]
            graph.add_weighted_edges_from(edges)

        return graph