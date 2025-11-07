from typing import Literal, List, Callable, Dict, overload, Tuple, Union
from src.pprl.utils import _dice, _scm
import src.pprl.openfhe_similarity
from src.pprl.datamodel import  Block, EncodedData, BloomEncodedData, HomomorphicEcnodedData
from pybloom_live import BloomFilter as Bloom
from networkx import Graph
from ordered_set import OrderedSet
import numpy as np
from openfhe import *
from itertools import combinations, product
from .evaluation import Evaluation
from .datamodel import PPRLFeature
import time
class Matcher(PPRLFeature): 
    
    
    _method_name = 'Matcher'
    _method_short_name = 'Matcher'

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
        self.execution_time : float = 0.0
    
        
        
    def _candidates(self, extract_candidates: Union[Dict[str, Block], EncodedData]) -> Tuple[set, OrderedSet]: 
        candidates = set()
        unique_ids = OrderedSet()

        if isinstance(extract_candidates, dict):
            first_key, first_value = next(iter(extract_candidates.items()))
            if isinstance(first_value, Block):
                for _, block in extract_candidates.items():
                    for d1, d2 in combinations(block.entities.values(), 2):
                        for eid1, eid2 in product(d1, d2):
                            candidates.add((eid1,eid2))
                            unique_ids.add(eid1)
                            unique_ids.add(eid2)
            else: 
                for eid1, eid2s in extract_candidates.items():
                    for eid2 in eid2s:
                        candidates.add((eid1,eid2))
                        unique_ids.add(eid1)
                        unique_ids.add(eid2)
        else: 
            start = 0
            for i in range(0, len(extract_candidates.bounds)): 
                bound = extract_candidates.bounds[i]
                d1 = range(start, bound)
                for j in range(i+1, len(extract_candidates.bounds)): 
                    d2 = range(extract_candidates.bounds[j-1], extract_candidates.bounds[j])
                    for eid1, eid2 in product(d1, d2): 
                        candidates.add((eid1,eid2))
                        unique_ids.add(eid1)
                        unique_ids.add(eid2)
                start = bound
        
        return candidates, unique_ids
    
    @overload
    def predict(self,
            encoded_data: HomomorphicEcnodedData,
            blocks : Dict[str, Block] = None, 
            type_of_match: Literal['overlap', 'extension'] = 'overlap',
            homomorphism : Literal['bgvs'] = 'bgvs') -> Graph:
        """
        Similarity Computaation for Homomorphic Encrypted Data
        
        :param data: Data
        :param ecnoded_data: HomomorphicEncodedData
        :param blocks: Dictionary of Blocks 
        :type blocks: dict, optional
        :param type_of_match[optional]: {'overlap', ...}
        :param homomorphism[optional]: {'bgvs', ...}
        """
        ...
       
    @overload        
    def predict(self, 
                encoded_data : BloomEncodedData,
                blocks: Dict[str, Block] = None)  -> Graph: 
        """
        Vectorized  similarity computation for Bloom filters.

        :param BloomEncodedData ecnoded_data: Bloom Filters
        :param dict, optional blocks: Dictionary of Blocks 
        :return graph: Graph of similar pairs 
        :rtype: networkx.Graph
        """
        
        ... 
        
        
    def predict(self,
                encoded_data: EncodedData, 
                blocks: Dict[str, Block] = None, 
                **kwargs) -> Graph: 

        start_time = time.time()
        self.encoded_data = encoded_data
        self.skip_ground_truth = encoded_data.skip_ground_truth
        
        if not encoded_data.skip_ground_truth: 
            self.ground_truth = encoded_data.ground_truth
        
        self.candidates, self.unique_ids = self._candidates(blocks) if blocks else self._candidates(encoded_data)


        if not self.candidates:
            gragh = Graph()

        if isinstance(encoded_data, BloomEncodedData): 
            self._method_info = """Matching with Bloom Filters"""
            graph =  self._predict_from_bloom(encoded_data)
        elif isinstance(encoded_data, HomomorphicEcnodedData):
            self._method_info = """Matching with Homomorhpic Encoded Data """
            graph = self._predict_from_vector(encoded_data, **kwargs) 
        else: 
            raise TypeError(f"encoded_data must be of type: BloomEncodedData or HomomorphicEcnodedData, but its {type(encoded_data)}")

        self.execution_time = time.time() - start_time
        return graph

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
       
            
    def _configuration(self):
        return { "batch_size": self.batch_size,
            "threshold": self.threshold,
            "metric": self.metric,
            "attributes": self.attributes } 
        

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
        bloom_filters = encoded_data.bitarray_dict
        attributes = self.attributes if self.attributes else encoded_data.attributes
        bf_length = encoded_data.length
        n_attributes = len(attributes)

        id_to_idx = {eid: idx for idx, eid in enumerate(unique_ids)}
        idx_to_id = {idx: eid for idx, eid in enumerate(unique_ids)}
        
        
        # Create matrices: (n_entities, n_attributes, bf_length)
        bloom_matrix = np.zeros((len(unique_ids), n_attributes, bf_length), dtype=np.uint8)
        # Fill the bloom matrix
        for eid_idx, eid in enumerate(unique_ids):
            for attr_idx, attr in enumerate(attributes):
                bitarr = bloom_filters[eid][attr]
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
            i, j = batch_pairs[0]
            
            edges = [(idx_to_id[i], idx_to_id[j], float(sim))
                    for (i,j), sim in zip(batch_pairs, avg_similarities)
                    if sim >= self.threshold]
            graph.add_weighted_edges_from(edges)

        return graph
    
    def evaluate(self, 
                prediction : Graph,
                export_to_df: bool = False,
                export_to_dict: bool = False,
                with_classification_report: bool = False,
                verbose: bool = True) -> any:

        if self.encoded_data is None:
            raise AttributeError("Can not proceed to evaluation without data object.")

        if self.skip_ground_truth:
            raise AttributeError("Can not proceed to evaluation without a ground-truth file. " +
                    "Encoded Data object the ground-truth file has not been set.")

        eval_obj = Evaluation(self.encoded_data)
        true_positives = 0
        total_matching_pairs = prediction.number_of_edges()
        for _, (id1, id2) in self.encoded_data.ground_truth.iterrows():
            id2 = self.encoded_data.bounds[0] + id2
            if (id1 in prediction and id2 in prediction[id1]) or   \
                 (id2 in prediction and id1 in prediction[id2]):
                true_positives += 1

        eval_obj.calculate_scores(true_positives=true_positives,
                                  total_matching_pairs=total_matching_pairs)
        return eval_obj.report(self.method_configuration(),
                                export_to_df,
                                with_classification_report,
                                verbose)
        
    def stats(self) -> None:
        pass