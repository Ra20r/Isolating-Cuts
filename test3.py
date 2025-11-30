import random
import time
import math
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm

from algorithms.isolating_cuts_3 import isolating_cut
from algorithms.karger_stein import karger_stein_wrapper


RNG_SEED = 42
ER_SAMPLES = 20
ITERATIONS = 5
WARMUP_RUNS = 1
P = 0.1
START_N = 20
STEP_N = 20

np.random.seed(RNG_SEED)
random.seed(RNG_SEED)

nodes_list = []
edges_list = []
true_cut_list = []

predictor_iso_list = []
predictor_ks_list = []

time_iso_means = []
time_iso_stds = []
time_ks_means = []
time_ks_stds = []

err_iso = []
err_ks = []

# ------------------------------------------------------------
# Main experiment loop
# ------------------------------------------------------------
for i in tqdm(range(ER_SAMPLES), desc="Graph experiments"):
    n = START_N + i * STEP_N
    graph_seed = RNG_SEED + i
    G = nx.erdos_renyi_graph(n, P, seed=graph_seed)

    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()

    n_actual = G.number_of_nodes()
    m_actual = G.number_of_edges()

    rng_w = np.random.RandomState(RNG_SEED + i + 123)
    for (u, v) in G.edges():
        G[u][v]['weight'] = int(rng_w.randint(1, 10))

    Adj = nx.to_numpy_array(G, weight='weight')

    nodes_list.append(n_actual)
    edges_list.append(m_actual)

    # Theoretical predictors
    predictor_iso = (np.log2(n_actual)**6) * m_actual
    predictor_ks = (n_actual**2) * np.log2(n_actual)

    predictor_iso_list.append(predictor_iso)
    predictor_ks_list.append(predictor_ks)

    true_val, _ = nx.stoer_wagner(G, weight='weight')
    true_cut_list.append(true_val)

    # warm-up
    for _ in range(WARMUP_RUNS):
        isolating_cut(Adj)
        karger_stein_wrapper(Adj)

    iso_times = []
    ks_times = []
    iso_errs = []
    ks_errs = []

    for _ in range(ITERATIONS):
        t0 = time.perf_counter()
        v_iso = isolating_cut(Adj)
        t1 = time.perf_counter()
        iso_times.append(t1 - t0)
        iso_errs.append(abs(v_iso - true_val) / true_val)

        t0 = time.perf_counter()
        v_ks = karger_stein_wrapper(Adj)
        t1 = time.perf_counter()
        ks_times.append(t1 - t0)
        ks_errs.append(abs(v_ks - true_val) / true_val)

    time_iso_means.append(np.mean(iso_times))
    time_iso_stds.append(np.std(iso_times))
    time_ks_means.append(np.mean(ks_times))
    time_ks_stds.append(np.std(ks_times))

    err_iso.append(np.mean(iso_errs))
    err_ks.append(np.mean(ks_errs))


baseline_iso = time_iso_means[0]
baseline_ks = time_ks_means[0]

norm_time_iso = [t / baseline_iso for t in time_iso_means]
norm_time_ks = [t / baseline_ks for t in time_ks_means]

norm_pred_iso = [p / predictor_iso_list[0] for p in predictor_iso_list]
norm_pred_ks = [p / predictor_ks_list[0] for p in predictor_ks_list]


def loglog_slope(x_list, y_list):
    x = np.array(x_list)
    y = np.array(y_list)
    mask = (x > 0) & (y > 0)
    lx = np.log(x[mask])
    ly = np.log(y[mask])
    a, b = np.polyfit(lx, ly, 1)
    return a, b


slope_iso, intercept_iso = loglog_slope(predictor_iso_list, time_iso_means)
slope_ks,  intercept_ks = loglog_slope(predictor_ks_list,  time_ks_means)

print("\nLoglog slopes (runtime vs predictor):")
print(f"  Isolating Cuts: slope={slope_iso:.3f} (expect ~1)")
print(
    f"  Karger-Stein:   slope={slope_ks:.3f}  (expect ~1 for n^2 log n predictor)\n")


plt.figure(figsize=(7, 5))
plt.plot(norm_pred_iso, norm_time_iso, 'o-', label="Isolating Cuts")
plt.plot(norm_pred_ks,  norm_time_ks,  'o-', label="Karger-Stein")
plt.xlabel("Predictor (normalized)")
plt.ylabel("Runtime (normalized)")
plt.title("Normalized Runtime vs Normalized Predictor")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("normalized_runtime_test3.png", dpi=150)

plt.figure(figsize=(7, 5))
plt.scatter(predictor_iso_list, time_iso_means, label="Iso (data)")
plt.scatter(predictor_ks_list,  time_ks_means,  label="KS (data)")

xs = np.logspace(
    math.log10(min(predictor_iso_list + predictor_ks_list)),
    math.log10(max(predictor_iso_list + predictor_ks_list)),
    200
)

ys_iso = np.exp(intercept_iso) * xs**slope_iso
ys_ks = np.exp(intercept_ks) * xs**slope_ks

plt.plot(xs, ys_iso, '--', label=f"Iso fit slope={slope_iso:.2f}")
plt.plot(xs, ys_ks,  '--', label=f"KS fit slope={slope_ks:.2f}")

plt.xscale('log')
plt.yscale('log')
plt.xlabel("Predictor")
plt.ylabel("Runtime")
plt.title("Runtime vs Predictor (log-log)")
plt.grid(True, which='both')
plt.legend()
plt.tight_layout()
plt.savefig("raw_runtime_loglog_test3.png", dpi=150)

plt.figure(figsize=(7, 5))
plt.plot(nodes_list, err_iso, 'o-', label="Iso err")
plt.plot(nodes_list, err_ks,  'o-', label="KS err")
plt.xlabel("n")
plt.ylabel("Relative error")
plt.title("Relative Error vs n")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("error_vs_n_test3.png", dpi=150)

print("Experiment complete. Files saved: normalized_runtime_test3.png, raw_runtime_loglog_test3.png, error_vs_n_test3.png")
