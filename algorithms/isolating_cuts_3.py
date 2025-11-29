import numpy as np
from math import ceil, log2, floor
from algorithms.dinic_util import _build_dinic_with_super_nodes

def isolating_cut(graph_matrix: np.ndarray, R=None, trials=50) -> float:
    """
    Computes a minimum cut value using the randomised isolating cuts approach.
    Input:
        graph_matrix: (n, n) numpy array adjacency (weights >= 0). Undirected expected.
    Output:
        float: value of the found minimum cut (approx/exact according to flow).
    Note: This returns the min cut value found (not the cut set). Matches requested endpoint.
    """

    if graph_matrix is None:
        raise ValueError("graph_matrix is None")
    if graph_matrix.shape[0] == 0:
        return 0.0
    if graph_matrix.shape[0] != graph_matrix.shape[1]:
        raise ValueError("graph_matrix must be square")
    #Adding check to make sure the graph is undirected
    if not np.allclose(graph_matrix, graph_matrix.T):
        raise ValueError("graph_matrix must be undirected)")
    
    #If R is not provided, set it to all nodes as before
    if R is None:
        R = list(range(graph_matrix.shape[0]))

    #Making sure nothing went wrong when defining R previously
    for r in R:
        if r < 0 or r >= graph_matrix.shape[0]:
            raise ValueError(f"R contains invalid node index: {r}")

    n = graph_matrix.shape[0]

    total_weight = float(np.sum(np.triu(graph_matrix, k=1)))
    if total_weight <= 0:
        return 0.0

    if len(R) == 1:
        return 0.0

    best_cut = float('inf')
    probability_exp = 2
    sampling_probability = 1 / (max(1, np.log2(n)) ** probability_exp)
    super_node_capacity = total_weight + 1.0 # may be adjusted to a higher value

    trials = int((np.log2(n))**3)

    for _ in range(trials):
        # sample nodes from R with probability 1/polylog(n)
        R_sampled = [r for r in R if np.random.rand() < sampling_probability]
        if len(R_sampled) < 2:
            continue # trivial case, skip this trial

        # index map is required since R_sampled is not guaranteed to have contiguous indices
        index_map = {v: i for i, v in enumerate(R_sampled)}
        bits = ceil(log2(len(R_sampled)))
        sides = []
        for i in range(bits):
            A = [v for v in R_sampled if ((index_map[v] >> i) & 1) == 0]
            B = [v for v in R_sampled if ((index_map[v] >> i) & 1) == 1]

            if len(A) == 0 or len(B) == 0:
                sides.append([True] * n)
                continue

            din, s, t = _build_dinic_with_super_nodes(graph_matrix, A, B, super_node_capacity)
            _ = din.max_flow(s, t)
            reachable = din.mincut_source_side(s)
            sides.append(reachable[:n]) # sliced to n only, ignore super nodes
        
        for v in R_sampled:
            Uv_mask = np.ones(n, dtype=bool)
            for i in range(bits):
                side_mask = np.array(sides[i], dtype=bool)
                if side_mask[v]:
                    Uv_mask &= side_mask
                else:
                    Uv_mask &= ~side_mask
            Uv_mask[v] = True
            cut_value = np.sum(graph_matrix[Uv_mask][:, ~Uv_mask])
            best_cut = min(best_cut, cut_value)

    return best_cut