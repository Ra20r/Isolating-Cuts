import networkx as nx
import numpy as np
import random
import time
import os

random.seed(42)
# test data info (need to cite)
#
# the above are chosen randomly from different categories

def test():
    ch = input("Run default test? (y/n): ")

    if ch.lower() != 'y':
        ch = int(input("0: for ER, 1: for BA: "))
        models = [nx.erdos_renyi_graph, nx.barabasi_albert_graph]
        model_params = [(100, 0.01), (100, 5)]

        G = models[ch](*model_params[ch])

        for (u,v) in G.edges():
            G[u][v]['weight'] = random.uniform(1, 10)

        if ch == 0:
            # it does not guarantee connected graph, so take the largest component
            largest_cc_nodes = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest_cc_nodes)
        Adj = nx.to_numpy_array(G, weight='weight')
        a = time.perf_counter()
        value, _ = nx.stoer_wagner(G, weight='weight')
        b = time.perf_counter()
        print(f"Networkx Min Cut computation took {b - a:.6f} seconds")
        print(f"Min Cut Value: {value}")
        # add call to our min cut implementation here
        # add relative error calculation here        
        print("------------------------------------------------------")

    else:
        # load test graphs from files
        data_dir = "test_data"
        graphs = {}

        for filename in os.listdir(data_dir):
            if filename.endswith(".edges"):
                path = os.path.join(data_dir, filename)

                G = nx.read_edgelist(path, delimiter=' ', create_using=nx.Graph())

                # adding  random weights since dataset is unweighted
                for (u, v) in G.edges():
                    G[u][v]['weight'] = random.uniform(1, 10)

                graphs[filename] = G
                print(f"Loaded {filename}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

                # dataset has multiple components, focus on largest component
                largest_cc_nodes = max(nx.connected_components(G), key=len)
                H = G.subgraph(largest_cc_nodes)
                
                a = time.perf_counter()
                value, _ = nx.stoer_wagner(H, weight='weight')
                b = time.perf_counter()
                print(f"Networkx Min Cut computation took {b - a:.6f} seconds")
                print(f"Largest Component: {H.number_of_nodes()} nodes, {H.number_of_edges()} edges")
                print(f"Min Cut of Largest Component: {value}")
                # add call to our min cut implementation here
                # add relative error calculation here
                print("------------------------------------------------------")



if __name__ == "__main__": 
    test()