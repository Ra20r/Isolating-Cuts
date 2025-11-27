import numpy as np
from scipy.sparse.linalg import eigsh

def conductance(A, S):
    """Compute conductance for set S in adjacency matrix A"""
    S = np.array(S, dtype=int)
    T = np.setdiff1d(np.arange(A.shape[0]), S)

    cut = A[S][:, T].sum()
    volS = A[S].sum()
    volT = A[T].sum()
    if min(volS, volT) == 0:
        return 1.0
    return cut / min(volS, volT)

def fiedler_cut(A, nodes):
    """Return a bipartition of 'nodes' using the Fiedler vector."""
    subA = A[np.ix_(nodes, nodes)]
    # compute 2 smallest eigenvalues of Laplacian
    deg = subA.sum(axis=1)
    L = np.diag(deg) - subA
    try:
        _, vecs = eigsh(L, k=2, which='SM')
        f = vecs[:, 1]
    except:
        # fallback if eigsh fails
        f = np.random.randn(len(nodes))

    order = np.argsort(f)
    # best sweep cut
    best_phi = 1.0
    best_S = None
    for k in range(1, len(nodes)):
        S = nodes[order[:k]]
        phi = conductance(A, S)
        if phi < best_phi:
            best_phi = phi
            best_S = S
    return best_phi, best_S

def expander_decomposition(A, phi):
    """
        Return list of clusters with conductance >= phi. Very simplified version of expander decomposition.
        Implemented just for the sake of completeness and not optimized for performance.
        The implementation follows the high-level idea of recursive spectral partitioning.
    """
    clusters = [np.arange(A.shape[0])]
    i = 0

    while i < len(clusters):
        C = clusters[i]
        if len(C) <= 2:
            i += 1
            continue

        cut_phi, S = fiedler_cut(A, C)
        if cut_phi >= phi or S is None or len(S) == 0 or len(S) == len(C):
            # already an expander
            i += 1
            continue

        T = np.setdiff1d(C, S)
        clusters[i] = S
        clusters.append(T)

    return [list(c) for c in clusters]