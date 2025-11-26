def sparsify_terminals(clusters, R_curr):
    """
    Selects representatives from each cluster to form a new terminal set R'.
    
    Input:
        clusters: List of clusters from expander decomposition
        R_curr: Current set of terminals
    
    Output:
        R_new: New sparsified terminal set
    """
    Rset = set(R_curr)
    Rprime = []
    for C in clusters:
        inter = [v for v in C if v in Rset]
        if inter:
            rep = min(inter)
            Rprime.append(rep)

    Rprime = sorted(list(dict.fromkeys(Rprime)))
    return Rprime
