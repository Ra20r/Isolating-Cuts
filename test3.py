import pandas as pd
import math
import matplotlib.pyplot as plt
import os
import random
import time
import numpy as np
import networkx as nx
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
    # for log^2 n runs, use n^2 log^3 n
    predictor_ks = (n_actual**2) * (np.log2(n_actual)**3)

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


output_dir = "test3_results_single_py_file"
os.makedirs(output_dir, exist_ok=True)
print(f"\nProcessing results... saving to {output_dir}/")

csv_path = os.path.join(output_dir, "experiment_data.csv")

df = pd.DataFrame({
    "Nodes_N": nodes_list,
    "Edges_M": edges_list,
    "True_MinCut": true_cut_list,

    "Iso_Time_Mean": time_iso_means,
    "Iso_Time_Std": time_iso_stds,
    "Iso_Error_Rel": err_iso,
    "Iso_Predictor": predictor_iso_list,

    "KS_Time_Mean": time_ks_means,
    "KS_Time_Std": time_ks_stds,
    "KS_Error_Rel": err_ks,
    "KS_Predictor": predictor_ks_list
})

df.to_csv(csv_path, index_label="Run_Index")


baseline_iso = time_iso_means[0]
baseline_ks = time_ks_means[0]

norm_time_iso = [t / baseline_iso for t in time_iso_means]
norm_time_ks = [t / baseline_ks for t in time_ks_means]

norm_pred_iso = [p / predictor_iso_list[0] for p in predictor_iso_list]
norm_pred_ks = [p / predictor_ks_list[0] for p in predictor_ks_list]


def loglog_slope(x_list, y_list):
    """Calculates slope of log-log data to determine empirical exponent."""
    x = np.array(x_list)
    y = np.array(y_list)
    mask = (x > 0) & (y > 0)
    lx = np.log(x[mask])
    ly = np.log(y[mask])
    a, b = np.polyfit(lx, ly, 1)
    return a, b


slope_iso, intercept_iso = loglog_slope(predictor_iso_list, time_iso_means)
slope_ks,  intercept_ks = loglog_slope(predictor_ks_list,  time_ks_means)

print("-" * 40)
print(f"Empirical Complexity Validation (Slope of Log-Log):")
print(f"  Isolating Cuts: {slope_iso:.3f} (Ideal: ~1.0 vs predictor)")
print(f"  Karger-Stein:   {slope_ks:.3f}  (Ideal: ~1.0 vs predictor)")
print("-" * 40)


def save_chart(fig, filename):
    path_png = os.path.join(output_dir, f"{filename}.png")
    path_pdf = os.path.join(output_dir, f"{filename}.pdf")
    fig.savefig(path_png, dpi=150, bbox_inches='tight')
    fig.savefig(path_pdf, bbox_inches='tight')
    print(f"Saved: {filename} (.png & .pdf)")
    plt.close(fig)


plt.style.use('seaborn-v0_8-whitegrid')

# Individual validation (does the algo match its own theory)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Iso
ax1.plot(norm_pred_iso, norm_time_iso, 'o-',
         color='tab:blue', label="Observed")
ax1.plot(norm_pred_iso, norm_pred_iso, '--', color='gray',
         alpha=0.5, label="Perfect Linear Fit")
ax1.set_xlabel("Theoretical Complexity Increase")
ax1.set_ylabel("Actual Runtime Increase")
ax1.set_title(f"Isolating Cuts Validation\n(Slope: {slope_iso:.2f})")
ax1.legend()
ax1.grid(True)

# KS
ax2.plot(norm_pred_ks, norm_time_ks, 'o-',
         color='tab:orange', label="Observed")
ax2.plot(norm_pred_ks, norm_pred_ks, '--', color='gray',
         alpha=0.5, label="Perfect Linear Fit")
ax2.set_xlabel("Theoretical Complexity Increase")
ax2.set_ylabel("Actual Runtime Increase")
ax2.set_title(f"Karger-Stein Validation\n(Slope: {slope_ks:.2f})")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
save_chart(fig, "1_validation_individual")

# combined normalized
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(norm_pred_iso, norm_time_iso, 'o-', label="Isolating Cuts")
ax.plot(norm_pred_ks,  norm_time_ks,  's-', label="Karger-Stein")
ax.plot([1, max(max(norm_pred_iso), max(norm_pred_ks))],
        [1, max(max(norm_pred_iso), max(norm_pred_ks))],
        'k--', alpha=0.3, label="Ideal Scaling")

ax.set_xlabel("Normalized Predictor Growth")
ax.set_ylabel("Normalized Runtime Growth")
ax.set_title("Scaling Comparison (Normalized to N=20)")
ax.legend()
save_chart(fig, "2_validation_combined")

# raw "runtime vs n"
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(nodes_list, time_iso_means, 'o-', label="Isolating Cuts")
ax.plot(nodes_list, time_ks_means,  's-', label="Karger-Stein")

ax.set_xlabel("Number of Nodes (N)")
ax.set_ylabel("Time (seconds)")
ax.set_yscale('log')  # Log scale since KS is much slower
ax.set_title("Runtime vs Input Size (Log Scale)")
ax.legend()
save_chart(fig, "3_raw_runtime_vs_n")

# log-log analysis
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(predictor_iso_list, time_iso_means,
           color='tab:blue', label="Iso Data")
ax.scatter(predictor_ks_list,  time_ks_means,
           color='tab:orange', label="KS Data")

xs_range = np.logspace(
    math.log10(min(min(predictor_iso_list), min(predictor_ks_list))),
    math.log10(max(max(predictor_iso_list), max(predictor_ks_list))),
    100
)
ys_iso_fit = np.exp(intercept_iso) * xs_range**slope_iso
ys_ks_fit = np.exp(intercept_ks) * xs_range**slope_ks

ax.plot(xs_range, ys_iso_fit, '--', color='tab:blue',
        label=f"Iso Fit (m ~ {slope_iso:.2f})")
ax.plot(xs_range, ys_ks_fit,  '--', color='tab:orange',
        label=f"KS Fit (m ~ {slope_ks:.2f})")

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel("Theoretical Operations (Predictor)")
ax.set_ylabel("Actual Runtime (Seconds)")
ax.set_title("Log-Log Complexity Analysis")
ax.legend()
save_chart(fig, "4_loglog_analysis")

# accuracy / error
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(nodes_list, err_iso, 'o-', label="Isolating Cuts")
ax.plot(nodes_list, err_ks,  's-', label="Karger-Stein")
ax.axhline(0, color='black', linewidth=1, linestyle='--')

ax.set_xlabel("Number of Nodes (N)")
ax.set_ylabel("Relative Error (|approx - true| / true)")
ax.set_title("Approximation Error vs Graph Size")
ax.legend()
save_chart(fig, "5_accuracy_error")

print("\nAll plots generated in 'test3_results/'.")
