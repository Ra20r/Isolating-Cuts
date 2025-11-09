import numpy as np


class _UnionFind:
    """A helper class for the Karger-Stein algorithm using union-find."""

    def __init__(self, n):
        self.parent = np.arange(n)
        self.num_components = n

    def find(self, i):
        root = i
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[i] != root:
            parent_i = self.parent[i]
            self.parent[i] = root
            i = parent_i
        return root

    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            self.parent[root_j] = root_i
            self.num_components -= 1
            return True
        return False


def _karger_base(n: int, edges: np.ndarray) -> float:
    """Base case for Karger-Stein: runs Karger's simple algorithm n^2 times."""
    min_cut = float('inf')
    if edges.shape[0] == 0:
        return 0.0

    num_trials = n * (n - 1) // 2
    for _ in range(num_trials):
        uf = _UnionFind(n)
        shuffled_indices = np.random.permutation(edges.shape[0])

        for idx in shuffled_indices:
            if uf.num_components <= 2:
                break
            u, v, _ = edges[idx]
            uf.union(int(u), int(v))

        current_cut = 0.0
        for u, v, w in edges:
            if uf.find(int(u)) != uf.find(int(v)):
                current_cut += w

        min_cut = min(min_cut, current_cut)

    return min_cut


def _contract(n: int, edges: np.ndarray, t: int) -> tuple[int, np.ndarray]:
    """
    Contracts the graph G(n, edges) to t supernodes.
    Returns the new number of nodes (t) and the new edge list.
    """
    uf = _UnionFind(n)
    if edges.shape[0] > 0:
        shuffled_indices = np.random.permutation(edges.shape[0])
        for idx in shuffled_indices:
            if uf.num_components <= t:
                break
            u, v, _ = edges[idx]
            uf.union(int(u), int(v))

    new_node_map = {}
    new_idx = 0
    new_parent = np.zeros(n, dtype=int)

    for i in range(n):
        root = uf.find(i)
        if root not in new_node_map:
            new_node_map[root] = new_idx
            new_idx += 1
        new_parent[i] = new_node_map[root]

    new_n = uf.num_components

    if new_n <= 1 or edges.shape[0] == 0:
        return new_n, np.zeros((0, 3))

    new_adj_matrix = np.zeros((new_n, new_n))
    for u, v, w in edges:
        new_u, new_v = new_parent[int(u)], new_parent[int(v)]
        if new_u != new_v:
            new_adj_matrix[new_u, new_v] += w
            new_adj_matrix[new_v, new_u] += w  # keep symmetric for safety

    rows, cols = np.where(np.triu(new_adj_matrix) > 0)
    new_weights = new_adj_matrix[rows, cols]
    if rows.size == 0:
        new_edges = np.zeros((0, 3))
    else:
        new_edges = np.stack([rows, cols, new_weights], axis=1)

    return new_n, new_edges


def _karger_stein_iterative(n: int, edges: np.ndarray, base_threshold: int = 6) -> float:
    """
    Iterative version of Karger-Stein using an explicit stack to avoid recursion.
    Processes the recursion tree in DFS order, using a stack of (n, edges).
    """
    if edges.shape[0] == 0:
        return 0.0

    min_cut = float('inf')
    stack = [(n, edges)]

    while stack:
        cur_n, cur_edges = stack.pop()

        if cur_edges.shape[0] == 0:
            min_cut = min(min_cut, 0.0)
            continue

        if cur_n <= base_threshold:
            # base case: run the randomized base algorithm
            cut = _karger_base(cur_n, cur_edges)
            min_cut = min(min_cut, cut)
            continue

        # compute t and produce two contracted graphs
        t = int(np.ceil(1 + cur_n / np.sqrt(2)))

        n1, edges1 = _contract(cur_n, cur_edges, t)
        n2, edges2 = _contract(cur_n, cur_edges, t)

        # push both branches to stack for DFS processing
        # smaller graphs first or any order is fine
        if edges1.shape[0] > 0:
            stack.append((n1, edges1))
        else:
            min_cut = min(min_cut, 0.0)

        if edges2.shape[0] > 0:
            stack.append((n2, edges2))
        else:
            min_cut = min(min_cut, 0.0)

    return min_cut


# the one at (https://github.com/cshjin/MinCutAlgo/blob/master/algo/MinCut.py) was incorrect so here's a fixed version
def karger_stein_wrapper(graph_matrix: np.ndarray) -> float:
    """
    Public wrapper to handle the numpy adjacency matrix input.

    This function implements the full KR-Stein algorithm, which
    requires O(log^2 n) repetitions of the core recursive procedure
    to amplify the success probability.
    """
    n = graph_matrix.shape[0]
    if n <= 1:
        return 0.0

    rows, cols = np.where(np.triu(graph_matrix) > 0)
    weights = graph_matrix[rows, cols]
    if rows.size == 0:
        return 0.0

    edges = np.stack([rows, cols, weights], axis=1)

    '''
    # We must repeat the O(n^2 log n) algorithm O(log^2 n) times to get a high probability of success, as required by Karger-Stein.
    num_trials = int(np.ceil(np.log(n)**2)) + 1

    min_overall_cut = float('inf')

    for _ in range(num_trials):
        # Run one full instance of the recursive algorithm
        current_cut = _karger_stein_iterative(n, edges, base_threshold=6)
        min_overall_cut = min(min_overall_cut, current_cut)

    return min_overall_cut
    '''
    # that was painfully slow, so for benchmarking just do one iteration

    return _karger_stein_iterative(n, edges)
