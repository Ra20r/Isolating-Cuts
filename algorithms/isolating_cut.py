import numpy as np
from collections import deque
from math import ceil, log2


# Li-Panigrahi requires a maxflow algorithm; we implement Dinic/Dinitz's algorithm here
class Dinic:
    # define an Edge class inside Dinic for convenience; not to be used globally
    class Edge:
        __slots__ = ("v", "rev", "cap")

        def __init__(self, v, rev, cap):
            self.v = v
            self.rev = rev
            self.cap = cap

    def __init__(self, n):
        self.n = n
        self.g = [[] for _ in range(n)]

    def add_edge(self, u, v, cap):
        # forward edge index = len(g[u]), backward edge index = len(g[v])
        self.g[u].append(Dinic.Edge(v, len(self.g[v]), cap))
        self.g[v].append(Dinic.Edge(u, len(self.g[u]) - 1, 0.0))

    def bfs_level(self, s, t, level):
        for i in range(len(level)):
            level[i] = -1
        q = deque()
        level[s] = 0
        q.append(s)
        while q:
            u = q.popleft()
            for e in self.g[u]:
                if e.cap > 0 and level[e.v] < 0:
                    level[e.v] = level[u] + 1
                    if e.v == t:
                        return True
                    q.append(e.v)
        return level[t] >= 0

    def dfs_block(self, u, t, f, level, it):
        if u == t:
            return f
        for i in range(it[u], len(self.g[u])):
            e = self.g[u][i]
            if e.cap > 0 and level[e.v] == level[u] + 1:
                pushed = self.dfs_block(e.v, t, min(f, e.cap), level, it)
                if pushed > 0:
                    e.cap -= pushed
                    self.g[e.v][e.rev].cap += pushed
                    return pushed
            it[u] += 1
        return 0

    def max_flow(self, s, t):
        flow = 0.0
        level = [-1] * self.n
        # repeat BFS and DFS blocking flow
        while self.bfs_level(s, t, level):
            it = [0] * self.n
            pushed = self.dfs_block(s, t, float('inf'), level, it)
            while pushed and pushed > 0:
                flow += pushed
                pushed = self.dfs_block(s, t, float('inf'), level, it)
        return flow

    def mincut_source_side(self, s):
        # After running max_flow, find vertices reachable from s in residual graph
        seen = [False] * self.n
        q = deque([s])
        seen[s] = True
        while q:
            u = q.popleft()
            for e in self.g[u]:
                if e.cap > 1e-12 and not seen[e.v]:
                    seen[e.v] = True
                    q.append(e.v)
        return seen


def _build_dinic_from_adjmat(adj: np.ndarray, include_self_loops=False):
    """
    Build Dinic instance (directed) for an undirected weighted adjacency matrix.
    We'll add both directions for each undirected edge.
    """
    n = adj.shape[0]
    din = Dinic(n)
    for i in range(n):
        for j in range(n):
            if i == j and not include_self_loops:
                continue
            w = float(adj[i, j])
            if w > 0:
                din.add_edge(i, j, w)
    return din


def _build_dinic_with_super_nodes(adj: np.ndarray, Aset, Bset, INF):
    """
    Build graph with super-source (index = n) connected to Aset with INF,
    and super-sink (index = n+1) connected from Bset with INF.
    Return Dinic instance, s_index, t_index.
    """
    n = adj.shape[0]
    din = Dinic(n + 2)
    for i in range(n):
        for j in range(i+1, n):
            w = float(adj[i, j])
            if w > 0:
                # to simulate undirected capacity we add two directed edges
                din.add_edge(i, j, w)
                din.add_edge(j, i, w)
    s = n
    t = n + 1
    for u in Aset:
        din.add_edge(s, u, INF)
    for v in Bset:
        din.add_edge(v, t, INF)
    return din, s, t


def isolating_cut(graph_matrix: np.ndarray) -> float:
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

    n = graph_matrix.shape[0]

    total_weight = float(np.sum(np.triu(graph_matrix, k=1)))
    if total_weight <= 0:
        return 0.0
    INF = total_weight + 1.0

    # Terminals R = all vertices 0..n-1 at first
    R = list(range(n))
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

    min_cut_value = float('inf')
    for idx, v in enumerate(R):
        Uv_mask = U_list[idx]
        in_U_indices = [u for u in range(n) if Uv_mask[u]]
        if len(in_U_indices) == 0:
            continue
        id_map = {}
        for new_i, old in enumerate(in_U_indices):
            id_map[old] = new_i
        t_index = len(in_U_indices)  # contracted node index
        size = t_index + 1
        din = Dinic(size)

        for a in range(n):
            for b in range(a+1, n):
                w = float(graph_matrix[a, b])
                if w <= 0:
                    continue

                a_in = a in id_map
                b_in = b in id_map
                if a_in and b_in:
                    din.add_edge(id_map[a], id_map[b], w)
                    din.add_edge(id_map[b], id_map[a], w)
                elif a_in and not b_in:
                    din.add_edge(id_map[a], t_index, w)
                    din.add_edge(t_index, id_map[a], w)
                elif not a_in and b_in:
                    din.add_edge(id_map[b], t_index, w)
                    din.add_edge(t_index, id_map[b], w)
                else:
                    pass

        if v not in id_map:
            # if v got contracted (shouldn't happen because we ensured v in Uv)
            continue

        s_idx = id_map[v]
        t_idx = t_index

        val = din.max_flow(s_idx, t_idx)
        if val < min_cut_value:
            min_cut_value = val

    if min_cut_value == float('inf'):
        # fallback: try simple cut: smallest degree sum over singletons
        degs = np.sum(graph_matrix, axis=1)
        return float(np.min(degs))
    return float(min_cut_value)
