import numpy as np


class _UnionFind:
    __slots__ = ['parent', 'rank', 'num_components']

    def __init__(self, n: int):
        self.parent = np.arange(n, dtype=int)
        self.rank = np.zeros(n, dtype=int)
        self.num_components = n

    def find(self, i: int) -> int:
        root = i
        while root != self.parent[root]:
            root = self.parent[root]

        curr = i
        while curr != root:
            nxt = self.parent[curr]
            self.parent[curr] = root
            curr = nxt
        return root

    def union(self, i: int, j: int) -> bool:
        root_i = self.find(i)
        root_j = self.find(j)

        if root_i != root_j:
            if self.rank[root_i] < self.rank[root_j]:
                self.parent[root_i] = root_j
            elif self.rank[root_i] > self.rank[root_j]:
                self.parent[root_j] = root_i
            else:
                self.parent[root_j] = root_i
                self.rank[root_i] += 1

            self.num_components -= 1
            return True
        return False


def _compress_graph(n: int, edges: np.ndarray, uf: _UnionFind) -> tuple[int, np.ndarray]:
    """
    Creates a new edge list with updated node indices and aggregated weights.
    """

    mapping = np.full(n, -1, dtype=int)
    new_n = 0

    for i in range(n):
        root = uf.find(i)
        if mapping[root] == -1:
            mapping[root] = new_n
            new_n += 1

    if new_n <= 1:
        return new_n, np.zeros((0, 3))

    adj_weights = {}

    for u, v, w in edges:
        root_u = uf.find(int(u))
        root_v = uf.find(int(v))

        if root_u != root_v:
            nu = mapping[root_u]
            nv = mapping[root_v]
            if nu > nv:
                nu, nv = nv, nu

            key = (nu, nv)
            adj_weights[key] = adj_weights.get(key, 0.0) + w

    if not adj_weights:
        return new_n, np.zeros((0, 3))

    new_edges = np.empty((len(adj_weights), 3))
    for idx, ((u, v), w) in enumerate(adj_weights.items()):
        new_edges[idx] = [u, v, w]

    return new_n, new_edges


def _contract(n: int, edges: np.ndarray, target_nodes: int) -> tuple[int, np.ndarray]:
    """
    Contracts graph down to `target_nodes` using Rejection Sampling.
    """
    if n <= target_nodes:
        return n, edges

    uf = _UnionFind(n)
    num_edges = edges.shape[0]

    weights = edges[:, 2].astype(float)
    total_weight = weights.sum()

    if total_weight <= 0:
        return n, edges

    probs = weights / total_weight

    batch_size = max(100, (n - target_nodes) * 2)

    while uf.num_components > target_nodes:
        rand_indices = np.random.choice(num_edges, size=batch_size, p=probs)

        for idx in rand_indices:
            u, v, _ = edges[idx]
            if uf.union(int(u), int(v)):
                if uf.num_components <= target_nodes:
                    break

    return _compress_graph(n, edges, uf)


def _karger_stein_recursive(n: int, edges: np.ndarray) -> float:
    """
    Recursive implementation of Karger-Stein logic.
    """
    if n <= 1:
        return 0.0

    if n <= 6:
        final_n, final_edges = _contract(n, edges, 2)
        return float(final_edges[:, 2].sum())

    t = int(np.ceil(n / np.sqrt(2) + 1))

    n1, edges1 = _contract(n, edges, t)
    cut1 = _karger_stein_recursive(n1, edges1)

    n2, edges2 = _contract(n, edges, t)
    cut2 = _karger_stein_recursive(n2, edges2)

    return min(cut1, cut2)


def karger_stein_wrapper(graph_matrix: np.ndarray, repetitions: int = None) -> float:
    """
    Public wrapper. graph_matrix is an (n x n) symmetric adjacency matrix.
    """
    n = graph_matrix.shape[0]
    if n <= 1:
        return 0.0

    rows, cols = np.where(np.triu(graph_matrix) > 0)
    if rows.size == 0:
        return 0.0

    weights = graph_matrix[rows, cols]
    edges = np.stack([rows.astype(int), cols.astype(int),
                     weights.astype(float)], axis=1)

    if repetitions is None:
        repetitions = max(1, int(np.ceil(np.log(max(2, n))**2)))

    min_cut = float('inf')

    for _ in range(repetitions):
        cut = _karger_stein_recursive(n, edges)
        min_cut = min(min_cut, cut)

    return min_cut
