"""
Docstring for pprl.clustering
"""
from queue import PriorityQueue
from collections import defaultdict
from time import time
from typing import Dict, List
from rustworkx import PyGraph as Graph, connected_components
# from networkx import Graph, connected_components
from ordered_set import OrderedSet
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

    def process(self, graph: Graph, encoded_data: BloomEncodedData,
            similarity_threshold: float = None) -> list:
        """NetworkX Connected Components Algorithm in the produced graph.

        Args:
            graph (Graph): Consists of the entity ids (nodes) and the similarity scores (edges).

        Returns:
            list: list of clusters
        """
        start_time = time()
        self.encoded_data = encoded_data
        self.similarity_threshold: float = similarity_threshold
        graph_copy = graph.copy()
        if self.similarity_threshold is not None:
            for (u,v, weight) in graph.weighted_edge_list():
                if weight < self.similarity_threshold:
                    graph_copy.remove_edge(u,v)
        clusters = list(connected_components(graph_copy))
        resulting_clusters = list(filter(lambda x: len(x) == 2, clusters))
        self.execution_time = time() - start_time
        return resulting_clusters

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

    def process(self, graph: Graph, encoded_data: BloomEncodedData,
            similarity_threshold: float = 0.1) -> list:
        """NetworkX Connected Components Algorithm in the produced graph.

        Args:
            graph (Graph): Consists of the entity ids (nodes) and the similarity scores (edges).

        Returns:
            list: list of clusters
        """
        self.similarity_threshold: float = similarity_threshold

        start_time = time()
        matched_entities = OrderedSet()
        self.encoded_data = encoded_data
        new_graph = Graph()
        new_graph.add_nodes_from(range(self.encoded_data.bounds[1]))
        priority_queue = PriorityQueue(maxsize = graph.num_edges()*2)
        for (u, v, weight) in graph.weighted_edge_list():
            if weight > self.similarity_threshold:
                priority_queue.put_nowait((1 - weight, u, v))

        while not priority_queue.empty():
            sim, entity_1, entity_2 = priority_queue.get()
            if entity_1 in matched_entities or entity_2 in matched_entities:
                continue
            new_graph.add_edge(entity_1, entity_2, sim)
            matched_entities.add(entity_1)
            matched_entities.add(entity_2)

        clusters = ConnectedComponentsClustering().process(new_graph,
                        encoded_data, similarity_threshold=None)
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

    def __init__(self) -> None:
        super().__init__()
        self.similarity_threshold : float

    def process(self,
        graph: Graph,
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

        start_time = time()
        self.similarity_threshold : float = similarity_threshold
        self.encoded_data = encoded_data
        number_of_comparisons : int = len(graph.edge_list())
        clusters : list = []

        if number_of_comparisons == 0:
            return clusters

        new_graph : Graph = Graph()
        new_graph.add_nodes_from(range(self.encoded_data.bounds[1]))
        men : set = set()
        self.men_candidates : dict = defaultdict(list)
        self.women_candidates : dict = defaultdict(list)

        for (u, v, weight) in graph.weighted_edge_list():
            man, woman = self.sorted_indicators(u, v)
            similarity = weight
            if similarity > self.similarity_threshold:
                self.men_candidates[man].append(ExtendedSimilarityEdge(left_node=man,
                                                                right_node=woman,
                                                                similarity=similarity))
                self.women_candidates[woman].append(ExtendedSimilarityEdge(left_node=woman,
                                                                    right_node=man,
                                                                    similarity=similarity))
            men.add(man)

        for man, candidates in self.men_candidates.items():
            self.men_candidates[man] = sorted(candidates, reverse=True)
        for woman, candidates in self.women_candidates.items():
            self.women_candidates[woman] = sorted(candidates, reverse=True)

        self.is_bachelor : list = [False] * self.encoded_data.bounds[0]
        self.is_uncertain : list = [False] * self.encoded_data.bounds[0]
        self.fiances : list = [-1] * (self.encoded_data.bounds[1] - self.encoded_data.bounds[0])
        self.current_matches : Dict[any, List[ExtendedSimilarityEdge]] = {}
        self.free_men : list = list(men)

        while len(self.free_men) > 0:
            man = self.free_men.pop(0)
            woman = self.get_first_active_candidate(entity=man)

            if woman == -1:
                if not self.is_bachelor[man]:
                    self.is_bachelor[man] = True
                    if not self.has_candidates(entity=man):
                        self.free_men.append(man)
                    self.activate_candidates_of(entity=man)
                else:
                    continue
            else:
                fiance = self.get_woman_fiance(woman=woman)
                if fiance == -1:
                    self.add_match(man=man, woman=woman, similarity=0.0)
                    self.set_woman_fiance(woman=woman, fiance=man)
                else:
                    if self.accepts_proposal(woman=woman,
                                             man=man):
                        self.remove_match(man=fiance, woman=woman)
                        self.add_match(man=man, woman=woman, similarity=0.0)
                        self.set_woman_fiance(woman=woman, fiance=man)
                        if not self.is_uncertain[fiance]:
                            self.deactivate_candidate(entity=fiance, candidate=woman)
                    else:
                        self.deactivate_candidate(entity=man, candidate=woman)

        for _, edges in self.current_matches.items():
            for edge in edges:
                man, woman, similarity = edge.left_node, edge.right_node, edge.similarity
                new_graph.add_edge(man, woman, similarity)

        clusters = list(connected_components(new_graph))
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
