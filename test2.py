import networkx as nx
import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import tqdm

from algorithms.isolating_cuts_3 import isolating_cut
from algorithms.karger_stein import karger_stein_wrapper

np.random.seed(42)


ER_SAMPLES = 20
ITERATIONS = 3
P = 0.1
START_N = 20
STEP_N = 20

nodes_list = []
edges_list = []
polylog_n_calls_list = []
true_cut_list = []

time_iso = []
time_ks = []

err_iso = []


for i in tqdm(range(ER_SAMPLES), desc="Graph scaling experiments"):

    n = START_N + i * STEP_N

    G = nx.erdos_renyi_graph(n, P)

    if not nx.is_connected(G):
        largest_cc_nodes = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc_nodes).copy()

    nodes_list.append(G.number_of_nodes())

    for (u, v) in G.edges():
        G[u][v]['weight'] = np.random.randint(1, 10)

    Adj = nx.to_numpy_array(G, weight='weight')

    edges_list.append(G.number_of_edges())
    polylog_n_calls_list.append((np.log2(G.number_of_nodes()))**6*G.number_of_edges())

    t0 = time.perf_counter()
    true_value, _ = nx.stoer_wagner(G, weight='weight')
    t1 = time.perf_counter()

    true_cut_list.append(true_value)

    iso_times = []
    ks_times = []
    iso_errs = []

    for _ in range(ITERATIONS):

        t0 = time.perf_counter()
        val_iso = isolating_cut(Adj)
        t1 = time.perf_counter()
        iso_times.append(t1 - t0)
        iso_errs.append(abs(val_iso - true_value) / true_value)

        #t0 = time.perf_counter()
        #val_ks = karger_stein_wrapper(Adj)
        #t1 = time.perf_counter()
        #ks_times.append(t1 - t0)

    time_iso.append(np.mean(iso_times))
    #time_ks.append(np.mean(ks_times))
    err_iso.append(np.mean(iso_errs))

plt.figure()
plt.plot(polylog_n_calls_list, time_iso, label="Isolating Cuts")
#plt.scatter(nodes_list, time_ks, label="Karger–Stein")
plt.xlabel("Polylog(n) * m")
plt.ylabel("Runtime (seconds)")
plt.title("Runtime vs Polylog(n) * m")
plt.legend()
plt.grid(True)
plt.show()
'''
plt.figure()
plt.scatter(nodes_list, time_iso, label="Isolating Cuts")
#plt.scatter(nodes_list, time_ks, label="Karger–Stein")
plt.xlabel("Number of Nodes")
plt.ylabel("Runtime (seconds)")
plt.title("Runtime vs Number of Nodes")
plt.legend()
plt.grid(True)
plt.show()


plt.figure()
plt.scatter(edges_list, time_iso, label="Isolating Cuts")
#plt.scatter(edges_list, time_ks, label="Karger–Stein")
plt.xlabel("Number of Edges")
plt.ylabel("Runtime (seconds)")
plt.title("Runtime vs Number of Edges")
plt.legend()
plt.grid(True)
plt.show()


plt.figure()
plt.scatter(nodes_list, err_iso)
plt.xlabel("Number of Nodes")
plt.ylabel("Relative Error")
plt.title("Isolating Cuts Error vs Graph Size")
plt.grid(True)
plt.show()


plt.figure()
plt.scatter(true_cut_list, time_iso, label="Isolating Cuts")
#plt.scatter(true_cut_list, time_ks, label="Karger–Stein")
plt.xlabel("True Min Cut Value")
plt.ylabel("Runtime (seconds)")
plt.title("Min Cut Value vs Runtime")
plt.legend()
plt.grid(True)
plt.show()
'''
print("\nEXPERIMENT COMPLETE\n")
