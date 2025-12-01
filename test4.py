import matplotlib.colors as mcolors
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
SAMPLES_PER_TYPE = 8
ITERATIONS_PER_GRAPH = 3
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
        # probs = [[intra if i == j else max(0.0, inter + rng_g.normal(0, 0.02))
        #           for j in range(k)] for i in range(k)]

        probs = np.zeros((k, k))

        for i in range(k):
            for j in range(i, k):
                if i == j:
                    probs[i][j] = intra
                else:
                    # Generate noise once for the pair
                    val = max(0.0, inter + rng_g.normal(0, 0.02))
                    probs[i][j] = val
                    probs[j][i] = val  # Enforce symmetry

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
    pbar = tqdm(total=total_runs, desc="Benchmarking Graphs")

    for g_type in GRAPH_TYPES:
        for i in range(SAMPLES_PER_TYPE):
            n_target = 2**i
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

    # scatter
    fig, ax = plt.subplots(figsize=(8, 8))

    min_val = min(df["Time_Iso_Mean"].min(), df["Time_KS_Mean"].min())
    max_val = max(df["Time_Iso_Mean"].max(), df["Time_KS_Mean"].max())

    ax.plot([min_val, max_val], [min_val, max_val],
            'k--', alpha=0.5, label="Equal Performance")

    colors = plt.colormaps['tab10'].resampled(len(GRAPH_TYPES))

    for idx, g_type in enumerate(GRAPH_TYPES):
        sub = df[df['Graph_Type'] == g_type]
        ax.scatter(sub["Time_Iso_Mean"], sub["Time_KS_Mean"],
                   s=60, alpha=0.8, edgecolors='w',
                   label=g_type, color=colors(idx))

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("Isolating Cuts Time (s)")
    ax.set_ylabel("Karger-Stein Time (s)")
    ax.set_title(
        "Head-to-Head: Runtime Comparison\n(Points above line = Isolating Cuts is Faster)")
    ax.legend(title="Topology")

    fig.savefig(os.path.join(output_dir, "1_head_to_head.png"),
                bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, "1_head_to_head.pdf"),
                bbox_inches='tight')
    plt.close(fig)

    # subplots per topology
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharey=False)
    axes = axes.flatten()

    for idx, g_type in enumerate(GRAPH_TYPES):
        ax = axes[idx]
        sub = df[df['Graph_Type'] == g_type].sort_values("Nodes")

        ax.plot(sub["Nodes"], sub["Time_Iso_Mean"], 'o-', color='#008080',
                label='Isolating Cuts', linewidth=2)
        ax.plot(sub["Nodes"], sub["Time_KS_Mean"], 's--', color='#FF7F50',
                label='Karger-Stein', linewidth=2)

        ax.set_title(f"Topology: {g_type}", fontsize=11, fontweight='bold')
        ax.set_xlabel("Nodes (N)")
        ax.set_ylabel("Time (s)")
        ax.set_yscale('log')
        ax.grid(True, which="both", ls="-", alpha=0.2)
        if idx == 0:
            ax.legend()

    if len(GRAPH_TYPES) < 6:
        fig.delaxes(axes[5])

    fig.suptitle("Runtime Scaling per Graph Topology", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(os.path.join(
        output_dir, "2_topology_breakdown.png"), bbox_inches='tight')
    fig.savefig(os.path.join(
        output_dir, "2_topology_breakdown.pdf"), bbox_inches='tight')
    plt.close(fig)

    # speedup factor
    df['Speedup'] = df['Time_KS_Mean'] / df['Time_Iso_Mean']

    fig, ax = plt.subplots(figsize=(12, 6))

    for idx, g_type in enumerate(GRAPH_TYPES):
        sub = df[df['Graph_Type'] == g_type].sort_values("Nodes")
        ax.plot(sub["Nodes"], sub["Speedup"], marker='o',
                linewidth=2, label=g_type, color=colors(idx))

    ax.axhline(1, color='k', linestyle='--', alpha=0.5, label="Parity (1.0)")
    ax.set_xlabel("Nodes (N)")
    ax.set_ylabel("Speedup Factor (Time KS / Time Iso)")
    ax.set_title("Relative Performance: How much faster is Isolating Cuts?")
    ax.set_yscale('log')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    max_speedup = df['Speedup'].max()
    ax.annotate(f'Max Speedup: {max_speedup:.1f}x',
                xy=(df.iloc[df['Speedup'].argmax()]['Nodes'], max_speedup),
                xytext=(10, -20), textcoords='offset points',
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))

    fig.savefig(os.path.join(output_dir, "3_speedup_factor.png"),
                bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, "3_speedup_factor.pdf"),
                bbox_inches='tight')
    plt.close(fig)

    # success rate
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    df['Node_Bin'] = pd.cut(df['Nodes'], bins=5)

    grouped_success = df.groupby('Node_Bin', observed=True)[
        ['Success_Rate_Iso', 'Success_Rate_KS']].mean()

    x = np.arange(len(grouped_success))
    width = 0.35

    rects1 = ax1.bar(x - width/2, grouped_success['Success_Rate_Iso'] * 100, width,
                     label='Isolating Cuts', color='#008080', alpha=0.8)
    rects2 = ax1.bar(x + width/2, grouped_success['Success_Rate_KS'] * 100, width,
                     label='Karger-Stein', color='#FF7F50', alpha=0.8)

    ax1.set_ylabel('Success Rate (%)')
    ax1.set_title('Algorithm Reliability by Graph Size')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"~{int(i.mid)}" for i in grouped_success.index])
    ax1.legend()
    ax1.set_ylim(0, 110)

    mask_iso = df['Err_Iso_Mean'] > 0
    mask_ks = df['Err_KS_Mean'] > 0

    data_iso = df.loc[mask_iso, 'Err_Iso_Mean']
    data_ks = df.loc[mask_ks, 'Err_KS_Mean']

    if len(data_iso) > 0 or len(data_ks) > 0:
        bplot = ax2.boxplot([data_iso, data_ks], labels=['Isolating Cuts', 'Karger-Stein'],
                            patch_artist=True)

        colors_box = ['#008080', '#FF7F50']
        for patch, color in zip(bplot['boxes'], colors_box):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax2.set_yscale('log')
        ax2.set_ylabel('Relative Error (Log Scale)')
        ax2.set_title('Error Magnitude (When exact cut not found)')
    else:
        ax2.text(0.5, 0.5, "Perfect Accuracy Achieved\n(No errors to plot)",
                 ha='center', va='center')

    fig.savefig(os.path.join(
        output_dir, "4_reliability_analysis.png"), bbox_inches='tight')
    fig.savefig(os.path.join(
        output_dir, "4_reliability_analysis.pdf"), bbox_inches='tight')
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
