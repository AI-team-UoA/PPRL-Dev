"""Matching methods of pprl"""

import time
from itertools import combinations, product
from typing import Literal, List, Dict, overload, Tuple, Union, Set
from dataclasses import dataclass

import numpy as np
from networkx import Graph
from ordered_set import OrderedSet
# from openfhe import CCParamsBGVRNS, SecurityLevel, \
#     CryptoContext, GenCryptoContext, KeyPair, PKESchemeFeature # pylint: disable=no-name-in-module
from tqdm.auto import tqdm
import pandas as pd

from pprl.utils import _dice, _scm
# import pprl.openfhe_similarity
from pprl.datamodel import  Block, EncodedData
from pprl.encoded_data import BloomEncodedData, HomomorphicEcnodedData
from pprl.evaluation import Evaluation
from pprl.datamodel import PPRLFeature


@dataclass
class MatcherConfig:
    """Matcher config class"""
    batch_size : int
    threshold : float
    metric : str
    attributes : List[str]
    candidates : Set
    unique_ids : OrderedSet
    concat_bf : bool = False

class Matcher(PPRLFeature):
    """Matcher class for pprl"""

    _method_name = 'Matcher'
    _method_short_name = 'Matcher'

    def __init__(self,
        batch_size : int = 10_000,
        threshold : float = 0.6,
        metric : Literal["dice", "scm", 'jaccard'] = "dice",
        attributes : List[str] = None,
        concat_bf : bool = False
    ) -> None:
        super().__init__()
        self.matcher_config = MatcherConfig(batch_size=batch_size,
            threshold=threshold, metric=metric,
            attributes=attributes, candidates=None, unique_ids=None, concat_bf=concat_bf)
        self.execution_time : float = 0.0
        self.encoded_data : EncodedData
        self.ground_truth : pd.DataFrame

        self.cc : CryptoContext
        self.key_pair : KeyPair



    def _candidates(self,
            extract_candidates: Union[Dict[str, Block], EncodedData]) -> Tuple[set, OrderedSet]:
        candidates = set()
        unique_ids = OrderedSet()

        if isinstance(extract_candidates, dict):
            _, first_value = next(iter(extract_candidates.items()))
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
            for i, bound in enumerate(extract_candidates.bounds):
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

        Args:

            ecnoded_data (HomomorphicEncodedData) : Homomorphic encoded data
            blocks (Dict[str, Block], optional): Dictionary of Blocks
            type_of_match (**'overlap','extension'**, optional): Type of match
            homomorphism (**'bgvs'**, optional): Homomorphism type

        Returns:
            Graph: The matching pairs

        """


    @overload
    def predict(self,
                encoded_data : BloomEncodedData,
                blocks: Dict[str, Block] = None)  -> Graph:
        """
        Vectorized  similarity computation for Bloom filters.

        Args:

            BloomEncodedData (ecnoded_data): Bloom Filters
            blocks (dict, optional): Dictionary of Blocks

        Returns:
            Graph: The matching pairs
        """



    def predict(self,
                encoded_data: EncodedData,
                blocks: Dict[str, Block] = None,
                **kwargs) -> Graph:
        """
        Vectorized  similarity computation for Bloom filters or HE.

        Args:

            BloomEncodedData (ecnoded_data): Bloom Filters
            blocks (dict, optional): Dictionary of Blocks
            **kwargs : Choose between he or bf

        Returns:
            Graph: The matching pairs
        """

        start_time = time.time()
        self.encoded_data = encoded_data

        if not encoded_data.skip_ground_truth:
            self.ground_truth = encoded_data.ground_truth

        self.matcher_config.candidates, self.matcher_config.unique_ids = \
            self._candidates(blocks) if blocks else self._candidates(encoded_data)


        if isinstance(encoded_data, BloomEncodedData):
            self._method_info = """Matching with Bloom Filters"""
            graph =  self._predict_from_bloom(encoded_data)
        elif isinstance(encoded_data, HomomorphicEcnodedData):
            self._method_info = """Matching with Homomorhpic Encoded Data """
            graph = self._predict_from_vector(encoded_data, **kwargs)
        else:
            raise TypeError(f"encoded_data must be of type: BloomEncodedData \
                or HomomorphicEcnodedData, but its {type(encoded_data)}")

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
        return { "batch_size": self.matcher_config.batch_size,
            "threshold": self.matcher_config.threshold,
            "metric": self.matcher_config.metric,
            "attributes": self.matcher_config.attributes }


    def _predict_from_vector(self, encoded_data: HomomorphicEcnodedData,
            type_of_match: Literal['overlap', 'extension'] = 'overlap',
            homomorphism : Literal['bgvs'] = 'bgvs' ) -> Graph:

        if homomorphism == 'bgvs':
            self._bgvs()

        matching_function = f'_{type_of_match}_jaccard'
        matching_function = getattr(src.pprl.openfhe_similarity, matching_function)

        attributes = self.matcher_config.attributes if self.matcher_config.attributes \
            else encoded_data.attributes

        graph = Graph()

        homomorphic_dict = encoded_data.encoded_dict

        for id1, id2 in self.matcher_config.candidates:
            similarity = 0.0
            encoded_1 = homomorphic_dict[id1]
            encoded_2 = homomorphic_dict[id2]
            for attr in attributes:
                similarity += matching_function(self.cc, self.key_pair,
                                               encoded_1[attr], encoded_2[attr])
            avg_similarity = similarity / len(attributes)
            if avg_similarity >= self.matcher_config.threshold:
                graph.add_edge(id1, id2, weight=avg_similarity)

        return graph

    def _predict_batches(self,
            batch_pairs : np.array,
            bloom_matrix: np.array,
            unique_ids_arr: np.array) -> List[Tuple[int, int, float]]:

        metrics_dict = {
            "dice" : _dice,
            "scm" : _scm
        }

        _metric = metrics_dict.get(self.matcher_config.metric, metrics_dict['dice'])

        avg_similarities : np.array = _metric(
                bloom_matrix[batch_pairs[:,0]],
                bloom_matrix[batch_pairs[:,1]],
                self.matcher_config.concat_bf)

        mask = avg_similarities >= self.matcher_config.threshold
        if not np.any(mask):
            return []

        valid_pairs = batch_pairs[mask]
        valid_scores = avg_similarities[mask]
        # edges = [(idx_to_id[i], idx_to_id[j], float(sim))
        #         for (i,j), sim in zip(batch_pairs, avg_similarities)
        #         if sim >= self.matcher_config.threshold]

        src_ids = unique_ids_arr[valid_pairs[:, 0]]
        tgt_ids = unique_ids_arr[valid_pairs[:, 1]]

        return list(zip(src_ids, tgt_ids, valid_scores.astype(float)))

    def _predict_create_bloom_matrix(self, attributes: list,
                            encoded_data: BloomEncodedData) -> np.array:
        n_ids = len(self.matcher_config.unique_ids)
        n_attrs = len(attributes)
        length = encoded_data.length

        # Pre-allocate matrix
        bloom_matrix = np.zeros((n_ids, n_attrs, length), dtype=np.uint8)
        # Create matrices: (n_entities, n_attributes, bf_length)
        # bloom_matrix = np.zeros((len(self.matcher_config.unique_ids),
                            # len(attributes), encoded_data.length), dtype=np.uint8)

        # Fill the bloom matrix
        for eid_idx, eid in enumerate(self.matcher_config.unique_ids):
            entity_attrs = encoded_data.bitarray_dict[eid]
            for attr_idx, attr in enumerate(attributes):
                bloom_matrix[eid_idx, attr_idx] = np.frombuffer(
                    entity_attrs[attr].unpack(), dtype=np.uint8
                )
                # bitarr = encoded_data.bitarray_dict[eid][attr]
                # bloom_matrix[eid_idx, attr_idx] = np.array(bitarr.tolist(), dtype=np.uint8)

        return bloom_matrix

    def _predict_from_bloom(self, encoded_data: BloomEncodedData) -> Graph:

        attributes = self.matcher_config.attributes if self.matcher_config.attributes \
            else encoded_data.attributes

        id_to_idx = {eid: idx \
            for idx, eid in enumerate(self.matcher_config.unique_ids)}
        idx_to_id = {idx: eid for eid, idx in id_to_idx.items()}
        unique_ids_arr = np.array(self.matcher_config.unique_ids)


        # Prepare candidate indices
        idx_pairs = np.array([(id_to_idx[eid1], id_to_idx[eid2])
                            for eid1, eid2 in self.matcher_config.candidates])
        n_candidates = len(idx_pairs)
        bloom_matrix = self._predict_create_bloom_matrix(attributes, encoded_data)

        graph = Graph()
        for start in tqdm(range(0, n_candidates, self.matcher_config.batch_size),
                    desc="Predicting batches", position=1, leave=False):
            end = min(start + self.matcher_config.batch_size, n_candidates)
            edges = self._predict_batches(idx_pairs[start:end],
                                bloom_matrix, unique_ids_arr)
            if edges:
                graph.add_weighted_edges_from(edges)

        return graph

    def evaluate(self,
                prediction : Graph,
                export_to_df: bool = False,
                with_classification_report: bool = False,
                verbose: bool = True) -> any:

        if self.encoded_data is None:
            raise AttributeError("Can not proceed to evaluation without data object.")

        if self.encoded_data.skip_ground_truth:
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
        """NOT IMPLEMENTED"""
