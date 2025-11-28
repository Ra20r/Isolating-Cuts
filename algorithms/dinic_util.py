import numpy as np
from algorithms.Dinic import Dinic

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
