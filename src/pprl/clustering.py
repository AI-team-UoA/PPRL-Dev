"""
Docstring for pprl.clustering
"""
from __future__ import annotations
from queue import PriorityQueue
from collections import defaultdict, deque
from time import time
from typing import Dict, List, Tuple
from itertools import chain
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
# from rustworkx import PyGraph as Graph
import numpy as np
from pprl.encoded_data import BloomEncodedData
from pprl.base_clustering import AbstractClustering, ExtendedSimilarityEdge

RANDOM_SEED = 42

class ConnectedComponentsClustering(AbstractClustering):
    """Creates the connected components of the graph. \
        Applied to graph created from entity matching. \
        Input graph consists of the entity ids (nodes) and the similarity scores (edges).
    """


    _method_name: str = "Connected Components Clustering"
    _method_short_name: str = "CCC"
    _method_info: str = "Gets equivalence clusters from the " + \
                    "transitive closure of the similarity graph."

    def __init__(self) -> None:
        super().__init__()
        self.similarity_threshold: float


    def process(self, graph: tuple, encoded_data: BloomEncodedData,
            similarity_threshold: float = None) -> list:
        """NetworkX Connected Components Algorithm in the produced graph.

        Args:
            graph (Graph): Consists of the entity ids (nodes) and the similarity scores (edges).

        Returns:
            list: list of clusters
        """
        edges: np.array
        weights: np.array

        start_time = time()
        self.encoded_data = encoded_data
        self.similarity_threshold: float = similarity_threshold

        edges, weights = graph
        valid_array : np.ndarray
        if self.similarity_threshold is not None:
            all_edges = edges
            all_weights = weights
            mask = all_weights > self.similarity_threshold
            valid_array = all_edges[mask]
        else:
            valid_array = edges

        n_nodes = int(np.max(valid_array))

        resulting_clusters = self.scipi_connected_components(
            edges=valid_array,
            n_nodes=n_nodes + 1
        )

        self.execution_time = time() - start_time
        return resulting_clusters

    @staticmethod
    def scipi_connected_components(edges:np.array, n_nodes: int) -> list:
        """Scipy Connected Components Algorithm in the produced graph.

        Args:
            edges (np.array): Consists of the entity ids (nodes) and the similarity scores (edges).
        Returns:
            list: list of clusters
        """

        if len(edges) == 0:
            return []


        flat_nodes = edges.ravel()
        degrees = np.bincount(flat_nodes, minlength=n_nodes)
        deg_u = degrees[edges[:, 0]]
        deg_v = degrees[edges[:, 1]]
        mask = (deg_u == 1) & (deg_v == 1)

        isolated_edges = edges[mask]
        return isolated_edges.tolist()





class UniqueMappingClustering(AbstractClustering):
    """Prunes all edges with a weight lower than t, sorts the remaining ones in
        decreasing weight/similarity and iteratively forms a partition for
        the top-weighted pair as long as none of its entities has already
        been matched to some other.
    """
    _method_name: str = "Unique Mapping Clustering"
    _method_short_name: str = "UMC"
    _method_info: str = "Prunes all edges with a weight lower than t, \
            sorts the remaining ones in" + \
            "decreasing weight/similarity and iteratively forms a partition for" + \
            "the top-weighted pair as long as none of its entities has already" + \
            "been matched to some other."

    def __init__(self) -> None:
        """Unique Mapping Clustering Constructor

        Args:
            similarity_threshold (float, optional): Prunes all edges with a weight
                lower than this. Defaults to 0.1.
            data (Data): Dataset module.
        """
        super().__init__()
        self.similarity_threshold: float

    def process(self, graph: Tuple[np.array, np.array], encoded_data: BloomEncodedData,
            similarity_threshold: float = 0.1) -> list:
        """NetworkX Connected Components Algorithm in the produced graph.

        Args:
            graph (Graph): Consists of the entity ids (nodes) and the similarity scores (edges).

        Returns:
            list: list of clusters
        """
        self.similarity_threshold: float = similarity_threshold

        start_time = time()
        self.encoded_data = encoded_data

        edges, weights = graph

        mask = weights > self.similarity_threshold
        valid_edges = edges[mask]
        valid_weights = weights[mask]

        u = valid_edges[:,0]
        v = valid_edges[:,1]
        w = valid_weights

        sort_indices = np.lexsort((v, u, -w))
        candidates = valid_edges[sort_indices]

        candidates = candidates[:edges.shape[0]*2]

        seen = set()
        unique_edge_indices = []
        for i, (u,v) in enumerate(candidates):
            if u not in seen and v not in seen:
                seen.add(u)
                seen.add(v)
                unique_edge_indices.append(i)

        unique_edges = candidates[unique_edge_indices]

        clusters = ConnectedComponentsClustering.scipi_connected_components(
            edges=unique_edges,
            n_nodes=int(np.max(unique_edges)) + 1
        )
        self.execution_time = time() - start_time
        return clusters

class CenterClustering(AbstractClustering):
    """Implements the Center Clustering algorithm.
    Input comparisons (graph edges) are sorted in descending order of similarity.
    Pairs of entities connected by these edges form the basis of the updated graph.
    Entities are evaluated to determine if they will serve
    as a center of a future cluster or as its member.
    This evaluation is based on a comparison of their cumulative edge weights in the graph,
    normalized by the number of edges in which they are involved.
    Finally, the algorithm identifies connected components within the graph,
    using the previously defined centers as the focal points for forming clusters.
    """


    _method_name: str = "Center Clustering"
    _method_short_name: str = "CC"
    _method_info: str = "Implements the Center Clustering algorithm," + \
        "In essence, it keeps it defines if a node within an edge constitutes \
            a center or member of future clusters" + \
        " by normalized over the graph weight sum comparison"
    def __init__(self) -> None:
        super().__init__()
        self.similarity_threshold: float

    def _init_comparisons(self, graph: Graph):
        edges_weight = defaultdict(float)
        edges_attached = defaultdict(int)
        comparisons = PriorityQueue(maxsize = graph.number_of_edges()*2)

        for (v1, v2, data) in graph.edges(data=True):
            similarity_score = data.get('weight', 0)
            if similarity_score > self.similarity_threshold:
                comparisons.put((-similarity_score, v1, v2))
                edges_weight[v1] = edges_weight[v1] + similarity_score
                edges_weight[v2] = edges_weight[v2] + similarity_score

                edges_attached[v1] = edges_attached[v1] + 1
                edges_attached[v2] = edges_attached[v2] + 1

        return edges_weight, edges_attached, comparisons

    def _create_clusters(self, edges_weight: dict, edges_attached: dict,
                        comparisons: PriorityQueue):

        new_graph = Graph()
        cluster_centers = set()
        cluster_members = set()

        while not comparisons.empty():
            similarity_score, v1, v2 = comparisons.get()
            v1_is_center : bool = v1 in cluster_centers
            v2_is_center : bool = v2 in cluster_centers
            v1_is_member : bool = v1 in cluster_members
            v2_is_member : bool = v2 in cluster_members

            if not (v1_is_center or v2_is_center or v1_is_member or v2_is_member):
                w1 = edges_weight[v1] / edges_attached[v1]
                w2 = edges_weight[v2] / edges_attached[v2]

                cluster_centers.add(v1 if w1 > w2 else v2)
                cluster_members.add(v1 if w1 <= w2 else v2)
                new_graph.add_edge(v1, v2, weight=-similarity_score)
            elif ((v1_is_center and v2_is_center) or (v1_is_member and v2_is_member)):
                continue
            elif (v1_is_center and not v2_is_member):
                cluster_members.add(v2)
                new_graph.add_edge(v1, v2, weight=-similarity_score)
            elif (v2_is_center and not v1_is_member):
                cluster_members.add(v1)
                new_graph.add_edge(v1, v2, weight=-similarity_score)

        return list(connected_components(new_graph))


    def process(self, graph: Graph, encoded_data: BloomEncodedData,
            similarity_threshold: float = 0.5) -> list:
        """
        Docstring for process

        :param self: Description
        :param graph: Description
        :type graph: Graph
        :param encoded_data: Description
        :type encoded_data: BloomEncodedData
        :param similarity_threshold: Description
        :type similarity_threshold: float
        :return: Description
        :rtype: list
        """

        start_time = time()
        self.similarity_threshold : float = similarity_threshold
        self.encoded_data = encoded_data

        edges_weight, edges_attached, comparisons = self._init_comparisons(graph)


        clusters = self._create_clusters(edges_weight, edges_attached, comparisons)
        self.execution_time = time() - start_time
        return clusters

class KiralyMSMApproximateClustering(AbstractClustering):
    """Implements the Kiraly MSM Approximate Clustering algorithm.
    Implements the so-called "New Algorithm"
    by Zoltan Kiraly 2013, which is a 3/2-approximation
    to the Maximum Stable Marriage (MSM) problem.
    The pairs resulting from the approximation
    of the stable relationships are translated into a graph,
    whose connected components we retain.
    """

    _method_name: str = "Kiraly MSM Approximate Clustering"
    _method_short_name: str = "KMAC"
    _method_info: str = "Ιmplements the Kiraly MSM Approximate Clustering algorithm," + \
        "In essence, it is a 3/2-approximation to the Maximum Stable Marriage (MSM) problem."

    men_candidates : dict
    women_candidates : dict
    is_bachelor : list
    is_uncertain : list
    fiances : list
    current_matches: dict
    free_men : list
    men : set
    base_candidates : dict

    def __init__(self) -> None:
        super().__init__()
        self.similarity_threshold : float
        self.men_candidates = None
        self.women_candidates = None




    def process(self,
        graph: Tuple[np.array, np.array],
        encoded_data: BloomEncodedData,
        similarity_threshold: float = 0.1) -> list:
        """
        Docstring for process

        :param self: Description
        :param graph: Description
        :type graph: Graph
        :param encoded_data: Description
        :type encoded_data: BloomEncodedData
        :param similarity_threshold: Description
        :type similarity_threshold: float
        :return: Description
        :rtype: list
        """

        edges, weights = graph

        start_time = time()
        self.similarity_threshold : float = similarity_threshold
        self.encoded_data = encoded_data
        number_of_comparisons : int = edges.shape[0]
        clusters : list = []

        if number_of_comparisons == 0:
            return clusters

        men = np.unique(edges[:,0])




        valid_mask = weights > self.similarity_threshold
        valid_edges = edges[valid_mask]
        valid_weights = weights[valid_mask]
        sorted_idx = np.argsort(-valid_weights)

        key_men_candidates = valid_edges[sorted_idx, 0]
        key_women_candidates = valid_edges[sorted_idx, 1]
        men_candidates =  valid_edges[sorted_idx]
        similarities = valid_edges[sorted_idx]
        men_to_indices = defaultdict(list)


        for i, m in enumerate(key_men_candidates):
            men_to_indices[m].append(i)

        men_len = self.encoded_data.bounds[0]
        women_len = self.encoded_data.bounds[1] - self.encoded_data.bounds[0]
        men_active = np.full(men_candidates.shape[0], True, bool)
        edge_score = np.full((men_len, women_len), -np.inf)
        edge_index = np.full((men_len, women_len), -1, dtype=int)

        is_bachelor = np.full(men_len, False, bool)
        is_uncertain = np.full(men_len, False, bool)
        fiances = np.full(women_len, -1, np.int32)
        current_matches = np.full(men_len, -1, np.int32)


        free_men = deque(men.tolist())

        while free_men:
            man = free_men.popleft()
            indices = men_to_indices.get(man, [])
            if not indices:
                continue

            indices = np.asarray(indices)
            active_mask = men_active[indices]
            if not active_mask.any():
                woman=-1
            else:
                index = indices[active_mask][0]
                woman = key_women_candidates[index]

            if woman == -1:
                if not is_bachelor[man]:
                    is_bachelor[man] =True
                    men_active[indices] = True
            else:
                fiance = fiances[woman - men_len]
                man_score = similarities[index]
                if fiance == -1:
                    current_matches[man] = woman
                    edge_score[man, woman-men_len] = man_score
                    edge_index[man, woman-men_len] = index
                    fiances[woman - men_len] = man
                else:
                    fiance_score = edge_score[fiance, woman-men_len]
                    fiance_index = edge_index[fiance, woman-men_len]
                    if is_uncertain[fiance] or man_score > fiance_score:
                        current_matches[fiance] = -1
                        current_matches[man] = woman
                        edge_score[man, woman-men_len] = man_score
                        edge_index[man, woman-men_len] = index
                        fiances[woman - men_len] = man
                        if not is_uncertain[fiance]:
                            men_active[fiance_index] = False
                    else:
                        men_active[index] = False

        mask = current_matches != -1
        men_ids = np.flatnonzero(mask)
        women_ids = current_matches[mask]
        # edge_list = np.array([(m, w) for m,w in enumerate(current_matches) if w!=-1])
        edge_list = np.column_stack((men_ids, women_ids))

        n_nodes = np.max(edge_list) + 1
        matrix = csr_matrix(
            (np.ones(edge_list.shape[0]),
            (edge_list[:,0], edge_list[:,1])),
            shape=(n_nodes, n_nodes)
        )
        _, labels = connected_components(matrix, directed=False,
                                            return_labels=True)


        sorted_idx = np.argsort(labels)
        sorted_labels = labels[sorted_idx]

        # Find where the label changes to identify cluster boundaries
        # returns indices where the label shifts (e.g., from cluster 0 to cluster 1)
        split_idx = np.flatnonzero(np.diff(sorted_labels) > 0) + 1

        # Split the original node indices into arrays based on these boundaries
        clusters = np.split(sorted_idx, split_idx)
        # clusters = [[] for _ in range(n_components)]
        # for node_index, component_index in enumerate(labels):
        #     clusters[component_index].append(node_index)

        # clusters = list(connected_components(new_graph))
        self.execution_time = time() - start_time
        return clusters

    def is_male(self, entity: int) -> bool:
        """
        Docstring for is_male

        :param self: Description
        :param entity: Description
        :type entity: int
        :return: Description
        :rtype: bool
        """
        return entity < self.encoded_data.bounds[0]

    def get_entity_candidates(self, entity : int) -> PriorityQueue:
        """
        Docstring for get_entity_candidates

        :param self: Description
        :param entity: Description
        :type entity: int
        :return: Description
        :rtype: PriorityQueue
        """
        candidates = self.men_candidates if self.is_male(entity) else self.women_candidates
        return candidates[entity]

    def has_candidates(self, entity : int) -> bool:
        """
        Docstring for has_candidates

        :param self: Description
        :param entity: Description
        :type entity: int
        :return: Description
        :rtype: bool
        """
        return len(self.get_entity_candidates(entity=entity)) > 0

    def activate_candidates_of(self, entity : int) -> None:
        """
        Docstring for activate_candidates_of

        :param self: Description
        :param entity: Description
        :type entity: int
        """
        candidates : List[ExtendedSimilarityEdge] = self.get_entity_candidates(entity=entity)
        for candidate in candidates:
            candidate.set_active(active=True)

    def get_first_active_candidate(self, entity : int) -> int:
        """
        Docstring for get_first_active_candidate

        :param self: Description
        :param entity: Description
        :type entity: int
        :return: Description
        :rtype: int
        """
        candidates : List[ExtendedSimilarityEdge] = self.get_entity_candidates(entity=entity)
        for candidate in candidates:
            if candidate.is_active():
                return candidate.right_node
        return -1

    def add_match(self, man : int, woman : int, similarity : float) -> None:
        """
        Docstring for add_match

        :param self: Description
        :param man: Description
        :type man: int
        :param woman: Description
        :type woman: int
        :param similarity: Description
        :type similarity: float
        """
        if man not in self.current_matches:
            self.current_matches[man] = []
        self.current_matches[man].append(ExtendedSimilarityEdge(left_node=man,
                                                                right_node=woman,
                                                                similarity=similarity))
    def remove_match(self, man : int, woman : int) -> None:
        """
        Docstring for remove_match

        :param self: Description
        :param man: Description
        :type man: int
        :param woman: Description
        :type woman: int
        """
        self.current_matches[man] = [match for match in self.current_matches[man] \
                                    if (match.left_node != man or match.right_node != woman)]

    def get_woman_fiance(self, woman : int) -> int:
        """
        Docstring for get_woman_fiance

        :param self: Description
        :param woman: Description
        :type woman: int
        :return: Description
        :rtype: int
        """
        return self.fiances[woman - self.encoded_data.bounds[0]]

    def set_woman_fiance(self, woman : int, fiance : int) -> None:
        """
        Docstring for set_woman_fiance

        :param self: Description
        :param woman: Description
        :type woman: int
        :param fiance: Description
        :type fiance: int
        """
        self.fiances[woman - self.encoded_data.bounds[0]] = fiance

    def deactivate_candidate(self, entity : int, candidate : int) -> bool:
        """
        Docstring for deactivate_candidate

        :param self: Description
        :param entity: Description
        :type entity: int
        :param candidate: Description
        :type candidate: int
        :return: Description
        :rtype: bool
        """
        entity_candidates : List[ExtendedSimilarityEdge] = self.get_entity_candidates(entity=entity)
        for entity_candidate in entity_candidates:
            if entity_candidate.right_node == candidate:
                entity_candidate.set_active(active=False)
                return True
        return False

    def accepts_proposal(self, woman : int, man : int):
        """
        Docstring for accepts_proposal

        :param self: Description
        :param woman: Description
        :type woman: int
        :param man: Description
        :type man: int
        """
        current_fiance : int = self.get_woman_fiance(woman=woman)

        if current_fiance == -1:
            return True
        if self.is_uncertain[current_fiance]:
            return True

        man_score : float = 0.0
        current_fiance_score : float = 0.0

        woman_candidates : List[ExtendedSimilarityEdge] = self.get_entity_candidates(entity=woman)

        for comparison in woman_candidates:
            candidate : int = comparison.right_node
            if candidate == man:
                man_score = comparison.similarity
            elif candidate == current_fiance:
                current_fiance_score = comparison.similarity

        return man_score > current_fiance_score
