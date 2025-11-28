import networkx as nx
import numpy as np
import argparse
import time
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

from algorithms.isolating_cuts_3 import isolating_cut
from algorithms.karger_stein import karger_stein_wrapper

# the graphs have random weights assigned to edges in range 1-10 integers 
# the true value is computed using networkx stoer_wagner implementation, this might take time for large graphs

np.random.seed(42)
# test data info (need to cite)
# 1. https://networkrepository.com/bn-fly-drosophila-medulla-1.php
# 2. https://networkrepository.com/bn-cat-mixed-species-brain-1.php
# the above are chosen randomly from different categories

def trials(graph, iterations, only_isolating_cuts=False):
    a = time.perf_counter()
    true_value, _ = nx.stoer_wagner(nx.from_numpy_array(graph), weight='weight')
    b = time.perf_counter()
    print(f"Networkx Min Cut computation took {b - a:.6f} seconds")
    print(f"True Min Cut Value (NetworkX Stoer-Wagner): {true_value}")
    # run multiple iterations to average time and error for both isolating cuts and karger cuts
    # each element in time_measurements and error_measurements corresponds to one iteration and is a tuple with two values, one for our implementation and one for karger implementation
    time_measurements = []
    error_measurements = []
    for _ in tqdm(range(iterations), desc="Running trials..."):
        isolating_trial_error = None
        isolating_trial_time = None
        karger_trial_error = None
        karger_trial_time = None

        a = time.perf_counter()
        isolating_cut_value = isolating_cut(graph)
        b = time.perf_counter()
        # print(f"Isolating Cuts Min Cut Value: {value}")
        isolating_trial_error = abs(isolating_cut_value - true_value) / true_value
        isolating_trial_time = b - a

        if not only_isolating_cuts:
            a = time.perf_counter()
            karger_value = karger_stein_wrapper(graph)
            b = time.perf_counter()
            karger_trial_error = abs(karger_value - true_value) / true_value
            karger_trial_time = b - a

        # add results to time_measurements and error_measurements lists
        time_measurements.append((isolating_trial_time, karger_trial_time))
        error_measurements.append((isolating_trial_error, karger_trial_error))
    
    if not only_isolating_cuts:
        karger_avg_time = sum(t[1] for t in time_measurements) / iterations
        karger_avg_error = sum(e[1] for e in error_measurements) / iterations
        print(f"Average Time (Karger): {karger_avg_time:.6f} seconds")
        print(f"Average Relative Error (Karger): {karger_avg_error:.6f}")
    isolating_cuts_time = sum(t[0] for t in time_measurements) / iterations
    isolating_cuts_avg_error = sum(e[0] for e in error_measurements) / iterations
    print(f"Average Time (Isolating Cuts): {isolating_cuts_time:.6f} seconds")
    print(f"Average Relative Error (Isolating Cuts): {isolating_cuts_avg_error:.6f}")
    # plot charts for errors here if needed
    # plot charts for time measurements here if needed


def test(args):
    ch = args.random.lower()
    iterations = args.iterations
    if ch.lower() == 'y':
        if args.model.upper() == "ER":
            ch = 0
        else:
            ch = 1
        models = [nx.erdos_renyi_graph, nx.barabasi_albert_graph]
        model_params = [(100, 0.01), (100, 5)]

        G = models[ch](*model_params[ch])

        for (u,v) in G.edges():
            G[u][v]['weight'] = np.random.randint(1, 10)

        if ch == 0:
            # ER does not guarantee connected graph, so take the largest component to test
            largest_cc_nodes = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest_cc_nodes)
            G = nx.convert_node_labels_to_integers(G, ordering="sorted")

        Adj = nx.to_numpy_array(G, weight='weight')

        print(f"Generated Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

        # plt.figure(figsize=(8, 6))
        # pos = nx.spring_layout(G)     # nice force-directed layout

        # # draw nodes + edges
        # nx.draw_networkx_nodes(G, pos, node_size=500, node_color="lightgreen")
        # nx.draw_networkx_edges(G, pos, alpha=0.6)

        # # draw edge labels (weights)
        # edge_labels = { (u,v): f"{Adj[u,v]:.2f}" for u,v in G.edges() }
        # nx.draw_networkx_labels(G, pos)
        # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

        # plt.show()

        trials(Adj, iterations, only_isolating_cuts=(args.only_isolating_cuts.lower() == 'y'))

        print("------------------------------------------------------")

    else:
        # load test graphs from files
        data_dir = args.data_dir
        graphs = {}

        for filename in os.listdir(data_dir):
            if filename.endswith(".edges"):
                path = os.path.join(data_dir, filename)

                G = nx.read_edgelist(path, delimiter=' ', create_using=nx.Graph())

                # adding  random weights since dataset is unweighted
                for (u, v) in G.edges():
                    G[u][v]['weight'] = np.random.randint(1, 10)

                graphs[filename] = G
                print(f"Loaded {filename}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

                # dataset has multiple components, focus on largest component
                largest_cc_nodes = max(nx.connected_components(G), key=len)
                H = G.subgraph(largest_cc_nodes)
                H = nx.convert_node_labels_to_integers(H, ordering="sorted")
                Adj = nx.to_numpy_array(H, weight='weight')
                trials(Adj, iterations, only_isolating_cuts=(args.only_isolating_cuts.lower() == 'y'))
                print("------------------------------------------------------")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Min Cut Benchmark")

    parser.add_argument("--random", type=str, default="n",
                        help="Run random graph test? (y/n)")

    parser.add_argument("--iterations", type=int, required=True,
                        help="Number of iterations per test")

    parser.add_argument("--model", type=str, default="ER",
                        choices=["ER", "BA"],
                        help="Model to use when random='n': ER or BA")

    parser.add_argument("--data_dir", type=str, default="test_data",
                        help="Directory containing .edges dataset files")
    
    parser.add_argument("--only_isolating_cuts", type=str, default="n",
                        help="Run only isolating cuts algorithm? (y/n)")

    args = parser.parse_args()

    test(args)