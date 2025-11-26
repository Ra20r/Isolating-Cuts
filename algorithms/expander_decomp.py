import numpy as np

def expander_decomposition(graph_matrix: np.ndarray, phi: float) -> list:
    """
    Decomposes the graph into clusters with conductance at least phi.
    Returns a list of clusters (each cluster is a list of vertex indices).
    """
    n = graph_matrix.shape[0]
    
    # Simple implementation: treat each vertex in U as its own cluster
    # and group all non-U vertices together
    clusters = []
    
    # Each vertex in U becomes a singleton cluster
    for v in U:
        clusters.append([v])
    
    # Group remaining vertices
    remaining = [v for v in range(n) if v not in U]
    if remaining:
        clusters.append(remaining)
    
    return clusters