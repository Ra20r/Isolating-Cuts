import numpy as np


def generate_ba(n: int, m: int) -> np.ndarray:
    """
    Generates a Barabási-Albert (BA) random graph using preferential attachment.

    Args:
        n (int): Total number of nodes.
        m (int): Number of edges to attach from a new node to existing nodes.
                 (m <= m0, where m0 is the initial number of nodes)

    Returns:
        np.ndarray: An (n, n) adjacency matrix.
    """
    m0 = m  # initial number of nodes, must be >= m
    if n < m0:
        raise ValueError("n must be >= m")

    matrix = np.zeros((n, n), dtype=int)

    rows, cols = np.triu_indices(m0, k=1)
    matrix[rows, cols] = 1
    matrix[cols, rows] = 1

    degrees = np.sum(matrix, axis=1)

    for i in range(m0, n):
        current_degrees = degrees[:i]
        total_degree = np.sum(current_degrees)

        if total_degree == 0:
            # if disconnected, connect randomly
            targets = np.random.choice(i, size=m, replace=False)
        else:
            probabilities = current_degrees / total_degree
            targets = np.random.choice(
                i, size=m, replace=False, p=probabilities)

        matrix[i, targets] = 1
        matrix[targets, i] = 1

        degrees[i] = m
        degrees[targets] += 1

    return matrix
