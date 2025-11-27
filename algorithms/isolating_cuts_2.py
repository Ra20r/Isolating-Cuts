import numpy as np
from math import ceil, log2
from dinic_util import _build_dinic_with_super_nodes, _build_dinic_from_adjmat
from expander_decomp import expander_decomposition
from sparsify_terminals import sparsify_terminals

def isolating_cut(graph_matrix: np.ndarray, R=None) -> float:
    """
    Computes a minimum cut value using the isolating cuts approach (Theorem 2.2).
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
    #May need to be larger than this???
    INF = total_weight + 1.0

    if len(R) == 1:
        return 0.0

    bits = max(1, ceil(log2(len(R))))

    sides = []
    for i in range(bits):
        A = [v for v in R if ((v >> i) & 1) == 0]
        B = [v for v in R if ((v >> i) & 1) == 1]

        if len(A) == 0 or len(B) == 0:
            sides.append([True] * (n + 2))
            continue

        din, s, t = _build_dinic_with_super_nodes(graph_matrix, A, B, INF)
        _ = din.max_flow(s, t)
        reachable = din.mincut_source_side(s)
        sides.append(reachable[:n])

    U_list = []
    for v in R:
        Uv_mask = np.ones(n, dtype=bool)
        for i in range(bits):
            side_mask = np.array(sides[i][:n], dtype=bool)
            if side_mask[v]:
                Uv_mask &= side_mask
            else:
                Uv_mask &= ~side_mask
        if not Uv_mask[v]:
            # at least ensure v included
            Uv_mask[v] = True
        U_list.append(Uv_mask)

    #Defining a list of cuts for each Uv to use later in recursion when finding min cut value
    cuts = []
    for Uv_mask in U_list:
        cut_other_side = ~Uv_mask
        cut= np.sum(graph_matrix[Uv_mask][:, cut_other_side])
        cuts.append(cut)

    #Finding the largest of the Uv sets to determine balanced vs unbalanced case
    Uv_sizes = [np.sum(Uv_mask) for Uv_mask in U_list]
    max_size = max(Uv_sizes)

    # Using a threshold of 0.5 to determine if we are in the balanced or unbalanced case
    if max_size <= 0.5 * n:  # 2-balanced case
        # terminal set R
        # sparsify R:
        #   - use expander decomposition (Matula) to get clusters
        #   - each cluster must not have any internal cuts with conductance < phi = 1/polylog(n)
        #   - i.e. if a cluster has conductance < phi, we split it further
        # - pick one representative from each cluster
        # - the representatives form R', which is smaller than R
        # repeat until |R'| < polylog(n) (variable: R_limit)
        # Set phi parameter (conductance threshold)
        C = 8 # this value needs to be tuned since large n can blow up phi
        n = graph_matrix.shape[0]
        # use natural log; ensure no division by zero
        phi = 1.0 / (max(2.0, np.log(n)) ** C)   # phi = 1/(log n)^C
        # stopping threshold for terminals (polylog(n))
        B = max(1, int(np.ceil((np.log(max(2.0, n))) ** 3)))

        # R is the current terminal set (list)
        R_curr = list(R)   # ensure list

        # iterative sparsification loop: repeat until terminals are polylog-sized
        while len(R_curr) > B:
            # decompose graph into phi-expanders (deterministic)
            clusters = expander_decomposition(graph_matrix, phi)

            # representative per cluster that intersects R_curr
            R_new = sparsify_terminals(clusters, R_curr, phi)

            # fallback to guarantee progress
            if len(R_new) >= len(R_curr):
                # deterministic halving (keep every 2nd element sorted by index)
                # since theorem guarantees at least halving
                R_sorted = sorted(R_curr)
                R_new = R_sorted[::2]
                if len(R_new) == 0:
                    R_new = R_sorted[:1]
            R_curr = R_new
        return isolating_cut(graph_matrix, R_curr)
    else:
        #Handle unbalanced case with recursion
        index_largest_set = np.argmax(Uv_sizes)
        largest_set_mask = U_list[index_largest_set]
        largest_set_cut_value = float(cuts[index_largest_set])

        rest_graph_mask = ~largest_set_mask
        rest_graph_indices = np.nonzero(rest_graph_mask)[0].tolist()

        #If the largest set is the whole graph, return its cut value
        if len(rest_graph_indices) == 0:
            return largest_set_cut_value

        # Build subgraph for the rest of the graph
        rest_adjacencies = graph_matrix[np.ix_(rest_graph_indices, rest_graph_indices)]

        #Determining and storing new set of terminals in the rest of the graph
        idx_map = {orig: new for new, orig in enumerate(rest_graph_indices)}
        R_new = []
        for v in R:
            if v in idx_map:
                R_new.append(idx_map[v])

        # If no terminals remain in subproblem, return cut
        if len(R_new) == 0:
            return largest_set_cut_value

        # Recursively solve inside complement
        rest_graph_min_cut = isolating_cut(rest_adjacencies, R_new)

        #return either the largest set cut value or the rest graph min cut, whichever is smaller
        return min(largest_set_cut_value, rest_graph_min_cut)
