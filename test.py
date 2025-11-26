import networkx as nx
import numpy as np
import random
import time

def test():
    a = time.perf_counter()
    G_er = nx.erdos_renyi_graph(100, 0.01)
    G_ba = nx.barabasi_albert_graph(100, 5)

    for (u,v) in G_er.edges():
        G_er[u][v]['weight'] = random.uniform(1, 10)

    Adj_er = nx.to_numpy_array(G_er, weight='weight')

    for (u,v) in G_ba.edges():
        G_ba[u][v]['weight'] = random.uniform(1, 10)

    Adj_ba = nx.to_numpy_array(G_ba, weight='weight')
    
    # try isolating cuts with dinic
    # try isolating cuts with GR

    b = time.perf_counter()
    print(f"Graph generation took {b - a:.6f} seconds")


if __name__ == "__main__": 
    test()