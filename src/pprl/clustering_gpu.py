"""
Docstring for pprl.clustering
"""
from __future__ import annotations
from queue import PriorityQueue
from collections import defaultdict, deque
from time import time
from typing import Dict, List, Tuple
from itertools import chain
import cudf
import cugraph
import cugraph
from cupyx.scipy.sparse import csr_matrix
from cupyx.scipy.sparse.csgraph import connected_components
# from rustworkx import PyGraph as Graph
import numpy as np
import cupy as cp
from pprl.encoded_data import BloomEncodedData
from pprl.base_clustering_gpu import AbstractClustering, ExtendedSimilarityEdge

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
        edges = cp.asarray(edges)
        weights = cp.asarray(weights)
        valid_array : np.ndarray
        if self.similarity_threshold is not None:
            all_edges = edges
            all_weights = weights
            mask = all_weights >= self.similarity_threshold
            valid_array = all_edges[mask]
        else:
            valid_array = edges

        n_nodes = int(cp.max(edges))

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

        edges_df = cudf.DataFrame(
            edges,
            columns=["src", "dst"],
            dtype="int32"
            )

        # Build graph from edge list (NO CSR!)
        G = cugraph.Graph(directed=False)
        G.from_cudf_edgelist(
            edges_df,
            source="src",
            destination="dst",
            renumber=True  # important since you already have integer IDs
        )

        # Run WCC
        labels_df = cugraph.connected_components(G)

        label_counts = labels_df['labels'].value_counts()

        # 3. Identify labels that strictly have size == 2
        # We get the indices (which are the labels) where the count is 2
        valid_labels = label_counts[label_counts == 2].index

        # 4. Filter the dataframe directly
        # We keep only rows where the label is in our list of valid_labels
        # merge is usually faster than isin for large GPU dataframes
        valid_labels_df = cudf.DataFrame({'labels': valid_labels})
        filtered_df = labels_df.merge(valid_labels_df, on='labels')

        # 5. Sort to align pairs
        # We sort by 'labels' so the two nodes belonging to the same component end up next to each other.
        # We break ties with 'vertex' just to be deterministic.
        filtered_df = filtered_df.sort_values(by=['labels', 'vertex'])

        # 6. Extract the actual Node IDs (vertex column)
        sorted_nodes = filtered_df['vertex'].values

        # 7. Reshape (The Magic Step)
        # This works because we filtered for size=2 and sorted by label
        result_matrix = sorted_nodes.reshape(-1, 2)

        # exit(10)
        # labels = labels_df['labels'].values


        # # matrix = csr_matrix(
        # #     (cp.ones(len(edges)),
        # #     (edges[:,0], edges[:,1])),
        # #     shape=(n_nodes, n_nodes)
        # # )
        # # _, labels = connected_components(matrix, directed=False,
        # #                                     return_labels=True)
        # unique_labels, counts = cp.unique(labels, return_counts=True)

        # # 3. Identify labels that strictly have size == 2
        # # valid_labels is a boolean mask of the unique_labels
        # valid_mask = (counts == 2)
        # valid_component_ids = unique_labels[valid_mask]

        # # 4. Filter the nodes
        # # 'np.isin' checks which nodes belong to the valid components.
        # # We essentially discard all nodes that are part of clusters != 2
        # node_mask = cp.isin(labels, valid_component_ids)

        # # We need the indices of the nodes, not just the boolean mask
        # # We sort them by label to ensure the pairs stay together
        # valid_node_indices = cp.where(node_mask)[0]
        # valid_node_labels = labels[valid_node_indices]

        # # Sort by label so that pairs are adjacent in the array
        # # e.g. [nodeA, nodeB, nodeC, nodeD] where A&B are comp 1, C&D are comp 5
        # sort_order = cp.argsort(valid_node_labels)
        # sorted_nodes = valid_node_indices[sort_order]

        # # 5. Reshape into pairs (The Magic Step)
        # # Since we filtered for ONLY size 2, the length of sorted_nodes
        # # must be even. We can reshape directly into a list of pairs.
        # # This is instantaneous compared to a loop.
        # result_matrix = sorted_nodes.reshape(-1, 2)
        # # sorted_idx = np.argsort(labels)
        # # sorted_labels = labels[sorted_idx]

        # # # Find where the label changes to identify cluster boundaries
        # # # returns indices where the label shifts (e.g., from cluster 0 to cluster 1)
        # # split_idx = np.flatnonzero(np.diff(sorted_labels) > 0) + 1

        # # # Split the original node indices into arrays based on these boundaries
        # # clusters = np.split(sorted_idx, split_idx)

        # # # 4. Filter for clusters of size 2
        # # # This list comprehension is now iterating over CLUSTERS, not NODES
        # # # (e.g., 500k iterations instead of 100M)
        # # resulting_clusters = [c.tolist() for c in clusters if len(c) == 2]

        # # clusters = [[] for _ in range(n_components)]
        # # for node_index, component_index in enumerate(labels):
        # #     clusters[component_index].append(node_index)

        # # resulting_clusters = [c for c in clusters if len(c) == 2]

        return result_matrix.tolist()



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
        edges = cp.asarray(edges)
        weights = cp.asarray(weights)
        mask = weights >= self.similarity_threshold
        valid_edges = edges[mask]
        valid_weights = weights[mask]

        u = valid_edges[:,0]
        v = valid_edges[:,1]
        w = valid_weights


        keys_stack = cp.stack([v, u, -w])
        sort_indices = cp.lexsort(keys_stack)
        candidates = valid_edges[sort_indices]

        candidates = candidates[:edges.shape[0]*2].get()
        max_node = candidates.max() + 1
        used = np.zeros(max_node, dtype=bool)

        unique_edge_indices = []
        for i, (u,v) in enumerate(candidates):
            if not used[u] and not used[v]:
                used[u] = True
                used[v] = True
                unique_edge_indices.append(i)

        unique_edges = cp.asarray(candidates[unique_edge_indices])

        clusters = ConnectedComponentsClustering.scipi_connected_components(
            edges=unique_edges,
            n_nodes=int(cp.max(edges)) + 1
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

        start_time = time()

        self.encoded_data = encoded_data
        self.similarity_threshold : float = similarity_threshold


        edges, weights = graph
        edges_np = edges


        if edges.shape[0] == 0:
            return []


        edges_cp = cp.asarray(edges)
        weights_cp = cp.asarray(weights)
        men = cp.unique(edges_cp[:,0]).get()
        valid_mask = weights_cp > self.similarity_threshold
        valid_edges = edges_cp[valid_mask]
        valid_weights = weights_cp[valid_mask]


        key_men_candidates = valid_edges[:, 0].get()
        men_candidates = valid_edges.get()

        key_women_candidates = valid_edges[:, 1].get()
        women_candidates = valid_edges.get()
        men_to_indices = defaultdict(list)





        #  sort

        sorted_weights_indices = cp.argsort((-valid_weights)).get()
        key_men_candidates = key_men_candidates[sorted_weights_indices]
        men_candidates = men_candidates[sorted_weights_indices]
        key_women_candidates = key_women_candidates[sorted_weights_indices]
        women_candidates = women_candidates[sorted_weights_indices]

        similarities = valid_weights[sorted_weights_indices]

        for i, m in enumerate(key_men_candidates):
            men_to_indices[m].append(i)

        men_len = self.encoded_data.bounds[0]
        women_len = self.encoded_data.bounds[1] - self.encoded_data.bounds[0]
        men_active = np.full(men_candidates.shape[0], True, bool)
        edge_score = np.full((men_len, women_len), -np.inf)
        edge_index = np.full((men_len, women_len), -1, dtype=int)

        is_bachelor = cp.asarray(np.full(men_len, False, bool))
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

        edge_list = [(m, w) for m,w in enumerate(current_matches) if w!=-1]

        # Create GPU DataFrame directly
        edges_df = cudf.DataFrame(
            edge_list,
            columns=["src", "dst"],
            dtype="int32"
        )
        # Build graph from edge list (NO CSR!)
        G = cugraph.Graph(directed=False)
        G.from_cudf_edgelist(
            edges_df,
            source="src",
            destination="dst",
            renumber=True
        )
        components_df = cugraph.connected_components(G).to_pandas()
        # print(components_df)
        # exit(10)

        clusters = components_df.groupby('labels')['vertex'].apply(set).tolist()
        # print(clusters)
        # clusters = [set(group_data['vertex'].values) for _, group_data in components_df.groupby('labels')]

        # for label, group_data in group_v:
        #     print(f"\nComponent Label: {label}")
        #     print(group_data)
        # # print(group_v)
        # exit(10)
        # Run WCC
        # components_df = components_df.sort_values(by='labels')
        # sorted_idx = cp.asarray(components_df['vertex'])
        # sorted_labels = cp.asarray(components_df['labels'])

        # # 5. Split into clusters (Logic remains exactly the same)
        # split_idx = cp.flatnonzero(cp.diff(sorted_labels) > 0) + 1
        # clusters = cp.split(sorted_idx, split_idx.get())
        # labels = cp.full(self.encoded_data.bounds[1], -1, dtype=cp.int32)

        # labels[labels_df['vertex'].values] = labels_df['labels'].values
        # # print(labels_df)
        # edges_np = np.array(edge_list, dtype=np.int32)
        # new_edges = cp.asarray(edges_np)

        # n_nodes = cp.max(new_edges) + 1

        # matrix = csr_matrix(
        #     (cp.ones(new_edges.shape[0]),
        #     (new_edges[:,0], new_edges[:,1])),
        #     shape=(n_nodes, n_nodes)
        # )
        # components, labels = connected_components(matrix, directed=False,
        #                                     return_labels=True)


        # print(components)
        # # # print(labels)

        # sorted_idx = cp.argsort(labels)
        # sorted_labels = labels[sorted_idx]

        # split_idx = cp.flatnonzero(cp.diff(sorted_labels) > 0) + 1
        # clusters = cp.split(sorted_idx, split_idx.get())

        # print(len(clusters))
        # exit(10)

        # labels_df = labels_df.sort_values(by='labels')
        # sorted_idx = labels_df['vertex'].values
        # sorted_labels = labels_df['labels'].values
        # split_idx = cp.flatnonzero(cp.diff(sorted_labels) > 0) + 1
        # clusters = cp.split(sorted_idx, split_idx.get())
        # vertices = labels_df['vertex'].values
        # components = labels_df['labels'].values
        # sorted_idx = cp.argsort(components)
        # sorted_components = components[sorted_idx]
        # sorted_vertices = vertices[sorted_idx]
        # split_idx = cp.flatnonzero(cp.diff(sorted_components) > 0) + 1
        # clusters = cp.split(sorted_idx, split_idx.get())
        self.execution_time = time() - start_time
        return clusters

    def _process(self,
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
        edges = cp.asarray(edges)
        weights = cp.asarray(weights)

        start_time = time()
        self.similarity_threshold : float = similarity_threshold
        self.encoded_data = encoded_data
        number_of_comparisons : int = edges.shape[0]
        clusters : list = []

        if number_of_comparisons == 0:
            return clusters

        # men : set = set()

        self.men_candidates : dict = defaultdict(list)


        self.women_candidates : dict = defaultdict(list)

        men = edges[:, 0]
        men = cp.unique(men).tolist()

        valid_indices = weights > self.similarity_threshold
        valid_edges = edges[valid_indices]
        valid_weights = weights[valid_indices]
        # 2. Transfer to CPU immediately
        # Moving raw arrays to CPU now is 100x faster than iterating over them on GPU later
        edges_np = valid_edges.get()        # Shape (N, 2)
        weights_np = valid_weights.get()    # Shape (N,)

        # --- Process Men Candidates (Group by Column 0) ---

        # Sort by Men (u)
        sort_order_u = np.argsort(edges_np[:, 0])
        edges_u_sorted = edges_np[sort_order_u]
        weights_u_sorted = weights_np[sort_order_u]

        # Get columns
        u_col = edges_u_sorted[:, 0]
        v_col = edges_u_sorted[:, 1]

        # Find split points
        unique_u, start_indices = np.unique(u_col, return_index=True)

        # Split into groups (v and w)
        # Note: We don't need to split 'u' because we know u is constant for the group
        v_groups = np.split(v_col, start_indices[1:])
        w_groups = np.split(weights_u_sorted, start_indices[1:])

        # Build Dictionary
        self.men_candidates = {
            int(u): [
                ExtendedSimilarityEdge(int(u), int(tgt), int(w))
                for tgt, w in zip(targets, group_weights)
            ]
            for u, targets, group_weights in zip(unique_u, v_groups, w_groups)
        }

        # --- Process Women Candidates (Group by Column 1) ---

        # Sort by Women (v) - CRITICAL FIX: You must re-sort by the second column!
        sort_order_v = np.argsort(edges_np[:, 1])
        edges_v_sorted = edges_np[sort_order_v]
        weights_v_sorted = weights_np[sort_order_v]

        # Get columns
        u_col_v = edges_v_sorted[:, 0]
        v_col_v = edges_v_sorted[:, 1]

        # Find split points
        unique_v, start_indices_v = np.unique(v_col_v, return_index=True)

        # Split into groups (u and w)
        u_groups = np.split(u_col_v, start_indices_v[1:])
        w_groups_v = np.split(weights_v_sorted, start_indices_v[1:])

        # Build Dictionary
        self.women_candidates = {
            int(v): [
                ExtendedSimilarityEdge(int(src), int(v), int(w))
                for src, w in zip(sources, group_weights)
            ]
            for v, sources, group_weights in zip(unique_v, u_groups, w_groups_v)
        }

        # for i, (u,v) in enumerate(edges):
        #     man, woman = self.sorted_indicators(u, v)
        #     man = int(man)
        #     woman = int(woman)

        #     similarity = weights[i]
        #     if similarity > self.similarity_threshold:
        #         self.men_candidates[man].append(ExtendedSimilarityEdge(left_node=man,
        #                                                         right_node=woman,
        #                                                         similarity=similarity))
        #         self.women_candidates[woman].append(ExtendedSimilarityEdge(left_node=woman,
        #                                                             right_node=man,
        #                                                             similarity=similarity))
        #     men.add(man)
        # self.men = men

        ##SOLVE

        # active_men : cp.array = cp.where((self.men_partners == -1) &
        #                       (self.men_next_proposal_idx < (self.row_ends - self.row_starts)))[0]
        # while active_men.size > 0:
        #     flat_indices = self.row_starts[active_men] + self.men_next_proposal_idx[active_men]
        #     proposed_women = self.men_preferences[flat_indices] # The ID of the woman
        #     proposal_scores = self.men_pref_scores[flat_indices]
        #     batch_sort = cp.lexsort((-proposal_scores, proposed_women))
        #     active_men_sorted = active_men[batch_sort]
        #     proposed_women_sorted = proposed_women[batch_sort]
        #     scores_sorted = proposal_scores[batch_sort]
        #     unique_women, unique_indices = cp.unique(proposed_women_sorted, return_index=True)
        #     best_suitors = active_men_sorted[unique_indices]
        #     best_scores = scores_sorted[unique_indices]
        #     current_fiances = self.women_partners[unique_women]
        #     if not hasattr(self, 'women_match_scores'):
        #         self.women_match_scores = cp.full(number_of_comparisons+1, -1.0, dtype=cp.float32)
        #     current_scores = self.women_match_scores[unique_women]
        #     new_wins = (current_fiances == -1) | (best_scores > current_scores)
        #     winning_women = unique_women[new_wins]
        #     winning_men = best_suitors[new_wins]
        #     winning_scores = best_scores[new_wins]
        #     old_men = self.women_partners[winning_women]
        #     mask_old_exists = old_men != -1
        #     if cp.any(mask_old_exists):
        #         self.men_partners[old_men[mask_old_exists]] = -1
        #     self.women_partners[winning_women] = winning_men
        #     self.women_match_scores[winning_women] = winning_scores
        #     self.men_partners[winning_men] = winning_women
        #     self.men_next_proposal_idx[active_men] += 1
        #     active_men : cp.array = cp.where((self.men_partners == -1) &
        #                       (self.men_next_proposal_idx < (self.row_ends - self.row_starts)))[0]


        self.is_bachelor : list = [False] * self.encoded_data.bounds[0]


        self.is_uncertain : list = [False] * self.encoded_data.bounds[0]
        self.fiances : list = [-1] * (self.encoded_data.bounds[1] - self.encoded_data.bounds[0])
        self.current_matches : Dict[any, List[ExtendedSimilarityEdge]] = {}
        # self.free_men : list = list(men)
        self.free_men = men

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

        ##CLUSTERING
        # matched_men = cp.where(self.men_partners != -1)[0]
        # matched_women = self.men_partners[matched_men]
        # all_u = cp.concatenate([matched_men, matched_women])
        # all_v = cp.concatenate([matched_women, matched_men])
        # data = cp.ones(all_u.shape[0], dtype=cp.float32)

        # matrix = csr_matrix((data, (all_u, all_v)), shape=(n_nodes, n_nodes))

        edge_list = [(e.left_node, e.right_node) for e in chain.from_iterable(self.current_matches.values())]
        edges_np = np.array(edge_list, dtype=np.int32)
        new_edges = cp.asarray(edges_np)

        # new_edges = cp.array([
        #     (e.left_node, e.right_node)
        #     for e in chain.from_iterable(self.current_matches.values())
        # ])

        n_nodes = cp.max(new_edges) + 1

        matrix = csr_matrix(
            (cp.ones(new_edges.shape[0]),
            (new_edges[:,0], new_edges[:,1])),
            shape=(n_nodes, n_nodes)
        )
        _, labels = connected_components(matrix, directed=False,
                                            return_labels=True)


        sorted_idx = cp.argsort(labels)
        sorted_labels = labels[sorted_idx]

        # Find where the label changes to identify cluster boundaries
        # returns indices where the label shifts (e.g., from cluster 0 to cluster 1)
        split_idx = cp.flatnonzero(cp.diff(sorted_labels) > 0) + 1

        # Split the original node indices into arrays based on these boundaries
        clusters = cp.split(sorted_idx, split_idx.get())
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
