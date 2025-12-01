import pandas as pd
import math
import matplotlib.pyplot as plt
import os
import random
import time
import numpy as np
import networkx as nx
from tqdm import tqdm
from scipy.stats import linregress


from algorithms.isolating_cuts_3 import isolating_cut
from algorithms.karger_stein import karger_stein_wrapper


RNG_SEED = 42
OUTPUT_DIR = "test4_results"
CSV_FILENAME = "raw_results.csv"

# How many different graphs of EACH type to generate
SAMPLES_PER_TYPE = 20
ITERATIONS_PER_GRAPH = 5
WARMUP_RUNS = 1

START_N = 20
STEP_N = 20

GRAPH_TYPES = ['erdos_renyi', 'barabasi_albert',
               'stochastic_block', 'watts_strogatz', 'powerlaw_cluster']

np.random.seed(RNG_SEED)
random.seed(RNG_SEED)


def generate_graph(g_type, n, seed):
    """
    Generates different graph topologies to test algorithm robustness.
    """
    rng_g = np.random.default_rng(seed)

    if g_type == 'erdos_renyi':
        p = float(np.clip(2.5 * np.log(max(2, n)) / n, 0.05, 0.5))
        G = nx.erdos_renyi_graph(n, p, seed=int(seed))

    elif g_type == 'barabasi_albert':
        target_avg_deg = int(np.clip(2 * np.log(max(2, n)), 2, 12))
        m = max(1, target_avg_deg // 2)
        G = nx.barabasi_albert_graph(n, m, seed=int(seed))

    elif g_type == 'stochastic_block':
        max_comms = min(5, max(2, n // 3))
        k = int(rng_g.integers(2, max_comms + 1))
        base = np.ones(k, dtype=int)
        if n - k > 0:
            extra = rng_g.multinomial(n - k, [1.0 / k] * k)
            sizes = (base + extra).tolist()
        else:
            sizes = base.tolist()

        intra = float(rng_g.uniform(0.35, 0.8))
        inter = float(rng_g.uniform(0.01, 0.12))
        probs = [[intra if i == j else max(0.0, inter + rng_g.normal(0, 0.02))
                  for j in range(k)] for i in range(k)]
        G = nx.stochastic_block_model(sizes, probs, seed=int(seed))

    elif g_type == 'watts_strogatz':
        k_neighbors = int(np.clip(2 * int(np.log(max(2, n))), 2, n - 1))
        p_rewire = float(np.clip(rng_g.uniform(0.05, 0.3), 0.0, 0.9))
        if k_neighbors % 2 == 1:
            k_neighbors += 1
            if k_neighbors >= n:
                k_neighbors = max(2, n - 1 if (n - 1) % 2 == 0 else n - 2)
        G = nx.watts_strogatz_graph(n, k_neighbors, p_rewire, seed=int(seed))

    elif g_type == 'powerlaw_cluster':
        target_avg_deg = int(np.clip(2 * np.log(max(2, n)), 2, 12))
        m = max(1, target_avg_deg // 2)
        tri_prob = float(rng_g.uniform(0.0, 0.4))
        G = nx.powerlaw_cluster_graph(n, m, tri_prob, seed=int(seed))

    else:
        raise ValueError(f"Unknown graph type: {g_type}")

    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()

    rng_w = np.random.default_rng(seed + 999)
    for (u, v) in G.edges():
        G[u][v]['weight'] = int(rng_w.integers(1, 10))

    return G


def run_experiment():
    results_data = []

    total_runs = len(GRAPH_TYPES) * SAMPLES_PER_TYPE
    pbar = tqdm(total=total_runs, desc="Benchmarking Protocols")

    for g_type in GRAPH_TYPES:
        for i in range(SAMPLES_PER_TYPE):
            n_target = START_N + i * STEP_N
            graph_seed = RNG_SEED + (i * 1000)

            G = generate_graph(g_type, n_target, graph_seed)
            Adj = nx.to_numpy_array(G, weight='weight')

            n_actual = G.number_of_nodes()
            m_actual = G.number_of_edges()

            # SW can be slow on very large dense graphs
            true_val, _ = nx.stoer_wagner(G, weight='weight')

            pred_iso = (m_actual) * (np.log2(n_actual)**6)
            pred_ks = (n_actual**2) * (np.log2(n_actual)**3)

            if i == 0:
                for _ in range(WARMUP_RUNS):
                    isolating_cut(Adj)
                    karger_stein_wrapper(Adj)

            times_iso, times_ks = [], []
            errs_iso, errs_ks = [], []
            found_exact_iso, found_exact_ks = 0, 0

            for _ in range(ITERATIONS_PER_GRAPH):
                t0 = time.perf_counter()
                v_iso = isolating_cut(Adj)
                t1 = time.perf_counter()
                times_iso.append(t1 - t0)

                if true_val > 0:
                    errs_iso.append(abs(v_iso - true_val) / true_val)
                    if abs(v_iso - true_val) < 1e-9:
                        found_exact_iso += 1
                else:
                    errs_iso.append(0.0)

                t0 = time.perf_counter()
                v_ks = karger_stein_wrapper(Adj)
                t1 = time.perf_counter()
                times_ks.append(t1 - t0)

                if true_val > 0:
                    errs_ks.append(abs(v_ks - true_val) / true_val)
                    if abs(v_ks - true_val) < 1e-9:
                        found_exact_ks += 1
                else:
                    errs_ks.append(0.0)

            record = {
                "Graph_Type": g_type,
                "Nodes": n_actual,
                "Edges": m_actual,
                "Density": (2 * m_actual) / (n_actual * (n_actual - 1)),
                "True_MinCut": true_val,

                "Pred_Iso": pred_iso,
                "Pred_KS": pred_ks,

                "Time_Iso_Mean": np.mean(times_iso),
                "Time_Iso_Std": np.std(times_iso),
                "Err_Iso_Mean": np.mean(errs_iso),
                "Success_Rate_Iso": found_exact_iso / ITERATIONS_PER_GRAPH,

                "Time_KS_Mean": np.mean(times_ks),
                "Time_KS_Std": np.std(times_ks),
                "Err_KS_Mean": np.mean(errs_ks),
                "Success_Rate_KS": found_exact_ks / ITERATIONS_PER_GRAPH
            }
            results_data.append(record)
            pbar.update(1)

    pbar.close()
    return pd.DataFrame(results_data)


def setup_plot_style():
    """Sets a professional, scientific style."""
    plt.rcParams.update({
        'figure.figsize': (10, 6),
        'figure.dpi': 150,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'lines.linewidth': 2,
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 13,
        'axes.prop_cycle': plt.cycler(color=['#008080', "#FF1869", '#4169E1', "#ECCD1D"])
    })


def plot_results(df, output_dir):
    setup_plot_style()

    fig, ax = plt.subplots()

    ax.scatter(df["Pred_Iso"], df["Time_Iso_Mean"],
               alpha=0.6, label="Isolating Cuts", color='#008080', marker='o')

    ax.scatter(df["Pred_KS"], df["Time_KS_Mean"],
               alpha=0.6, label="Karger-Stein", color='#FF7F50', marker='s')

    for name, x_col, y_col, color in [("Iso", "Pred_Iso", "Time_Iso_Mean", "#008080"),
                                      ("KS", "Pred_KS", "Time_KS_Mean", "#FF7F50")]:
        slope, intercept, r_val, _, _ = linregress(
            np.log(df[x_col]), np.log(df[y_col]))
        x_fit = np.linspace(df[x_col].min(), df[x_col].max(), 100)
        y_fit = np.exp(intercept) * x_fit**slope
        ax.plot(x_fit, y_fit, '--', color=color, alpha=0.8,
                label=f"{name} Fit (Slope $\\approx$ {slope:.2f})")

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("Theoretical Complexity (Log Scale)")
    ax.set_ylabel("Measured Runtime (s) (Log Scale)")
    ax.set_title(
        "Complexity Verification: Slope Analysis\n(Slope of 1.0 = Perfect Match with Theory)")
    ax.legend()
    fig.savefig(os.path.join(
        output_dir, "1_complexity_verification.png"), bbox_inches='tight')
    plt.close(fig)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    markers = {'erdos_renyi': 'o',
               'barabasi_albert': '^', 'stochastic_block': 's', 'watts_strogatz': 'D', 'powerlaw_cluster': 'v'}

    for g_type in df['Graph_Type'].unique():
        sub = df[df['Graph_Type'] == g_type]
        m = markers.get(g_type, 'o')

        ax1.plot(sub["Nodes"], sub["Time_Iso_Mean"], marker=m, linestyle='-',
                 alpha=0.7, label=f"{g_type}")
        ax1.fill_between(sub["Nodes"],
                         sub["Time_Iso_Mean"] - sub["Time_Iso_Std"],
                         sub["Time_Iso_Mean"] + sub["Time_Iso_Std"], alpha=0.1)

        ax2.plot(sub["Nodes"], sub["Time_KS_Mean"], marker=m, linestyle='-',
                 alpha=0.7, label=f"{g_type}")
        ax2.fill_between(sub["Nodes"],
                         sub["Time_KS_Mean"] - sub["Time_KS_Std"],
                         sub["Time_KS_Mean"] + sub["Time_KS_Std"], alpha=0.1)

    ax1.set_title("Isolating Cuts Performance")
    ax1.set_xlabel("Nodes (N)")
    ax1.set_ylabel("Time (s)")
    ax1.legend()

    ax2.set_title("Karger-Stein Performance")
    ax2.set_xlabel("Nodes (N)")
    ax2.set_yscale('log')
    ax2.legend()

    fig.suptitle("Runtime vs Graph Size by Topology")
    fig.savefig(os.path.join(
        output_dir, "2_runtime_by_topology.png"), bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots()

    df['Node_Bin'] = pd.cut(df['Nodes'], bins=5)
    grouped = df.groupby('Node_Bin', observed=True)[
        ['Err_Iso_Mean', 'Err_KS_Mean']].mean()

    x = np.arange(len(grouped))
    width = 0.35

    ax.bar(x - width/2, grouped['Err_Iso_Mean'], width,
           label='Isolating Cuts Error', color='#008080')
    ax.bar(x + width/2, grouped['Err_KS_Mean'], width,
           label='Karger-Stein Error', color='#FF1869')

    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in grouped.index], rotation=15)
    ax.set_ylabel("Mean Relative Error")
    ax.set_title("Algorithm Accuracy vs Graph Size (Binned)")
    ax.legend()

    fig.savefig(os.path.join(output_dir, "3_accuracy_analysis.png"),
                bbox_inches='tight')
    plt.close(fig)

    print(f"Charts saved to {output_dir}/")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIR, CSV_FILENAME)

    print("Starting Comprehensive Benchmark...")
    print(f"Graph Types: {GRAPH_TYPES}")

    # comment out this if you don't want to rerun
    df_results = run_experiment()
    df_results.to_csv(csv_path, index=False)
    print(f"\nRaw data saved to {csv_path}")

    df_load = pd.read_csv(csv_path)
    plot_results(df_load, OUTPUT_DIR)

    print("\nBenchmark Complete.")
