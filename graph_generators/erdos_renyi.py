import numpy as np


def generate_er(n: int, p: float) -> np.ndarray:
    """
    Generates an Erdős-Rényi (G(n, p)) random graph.

    Returns:
        np.ndarray: An (n, n) adjacency matrix with 0/1 integer weights.
    """
    matrix = np.zeros((n, n), dtype=int)

    # indices for the upper triangle (k=1 excludes the diagonal)
    rows, cols = np.triu_indices(n, k=1)

    edges = np.random.rand(rows.size) < p
    matrix[rows[edges], cols[edges]] = 1

    # mirror the matrix to make it symmetric (undirected)
    matrix[cols[edges], rows[edges]] = 1

    return matrix
