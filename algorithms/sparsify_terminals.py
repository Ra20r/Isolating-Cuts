import math

def sparsify_terminals(clusters: list, R_curr: list, phi: float) -> list:
    """
    Selects representatives from each cluster to form a new terminal set R'.
    
    Input:
        clusters: List of clusters from expander decomposition
        R_curr: Current set of terminals
        phi: Conductance threshold used in decomposition
    
    Output:
        R_new: New sparsified terminal set
    """
    Rset = set(R_curr)
    Rnew = []
    for V in clusters:
        Ui = [v for v in V if v in Rset]
        k = len(Ui)
        if k == 0: # trivial cluster: do nothing
            continue
        elif 1 <= k <= 1/phi**2: # small cluster: add one arbitrary terminal
            Rnew.append(Ui[0])
        else: # large cluster: add 1+1/phi terminals
            to_take = int(1 + math.ceil(1/phi))
            Rnew.extend(Ui[:to_take])

    return Rnew
