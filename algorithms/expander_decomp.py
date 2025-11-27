import numpy as np

def expander_decomposition(graph_matrix: np.ndarray, phi: float, U: list) -> list:
    """
    Decomposes the graph into clusters with conductance at least phi.
    Returns a list of clusters (each cluster is a list of vertex indices).
    """
    # lambda_val = from a linear time solver, MATULA
    n = graph_matrix.shape[0]
    
    d_vec = np.zeros(n)
    if U is not None:
        for v in U:
            d_vec[v] = lambda_val
    
    # all in one cluster initially
    clusters = [list(range(n))]
    
    i = 0
    while i < len(clusters):
        C = clusters[i]
        if len(C) <= 1:
            i += 1
            continue
        
        
            
    return clusters