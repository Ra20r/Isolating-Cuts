import pandas as pd
import matplotlib.pyplot as plt
import os
import random
import time
import numpy as np
import networkx as nx
from tqdm import tqdm
from scipy.stats import wilcoxon, ttest_rel, linregress


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

    df = df.copy()
    df['Time_Ratio_KS_to_Iso'] = df['Time_KS_Mean'] / \
        (df['Time_Iso_Mean'] + 1e-12)
    df['Rel_Error_Diff'] = df['Err_KS_Mean'] - df['Err_Iso_Mean']
    df['Density_Bin'] = pd.qcut(df['Density'], q=4, duplicates='drop')

    # 1) Complexity verification: scatter with density coloring and fitted curves (log-log)
    fig, ax = plt.subplots()
    sc = ax.scatter(df["Pred_Iso"], df["Time_Iso_Mean"], c=df['Density'],
                    cmap='viridis', s=40, alpha=0.8, marker='o', label="Isolating Cuts")
    sc2 = ax.scatter(df["Pred_KS"], df["Time_KS_Mean"], c=df['Density'],
                     cmap='plasma', s=40, alpha=0.7, marker='s', label="Karger-Stein")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label('Graph Density')

    for name, x_col, y_col, style in [
        ("Iso", "Pred_Iso", "Time_Iso_Mean", "-"),
        ("KS", "Pred_KS", "Time_KS_Mean", "--")
    ]:
        mask = (df[x_col] > 0) & (df[y_col] > 0)
        if mask.sum() > 2:
            slope, intercept, r_val, _, _ = linregress(np.log(df.loc[mask, x_col]),
                                                       np.log(df.loc[mask, y_col]))
            x_fit = np.logspace(np.log10(df.loc[mask, x_col].min()),
                                np.log10(df.loc[mask, x_col].max()), 100)
            y_fit = np.exp(intercept) * x_fit**slope
            ax.plot(x_fit, y_fit, style, alpha=0.9,
                    label=f"{name} Fit (slope={slope:.2f}, $R^2$={r_val**2:.2f})")

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("Theoretical Complexity (log)")
    ax.set_ylabel("Measured Runtime (s) (log)")
    ax.set_title("Complexity Verification with Density Overlay")
    ax.legend()
    fig.savefig(os.path.join(
        output_dir, "1_complexity_verification.png"), bbox_inches='tight')
    fig.savefig(os.path.join(
        output_dir, "1_complexity_verification.pdf"), bbox_inches='tight')
    plt.close(fig)

    # 2) Paired runtime comparison per topology: boxplots and scatter of paired samples
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    types = df['Graph_Type'].unique()
    iso_times = [df[df['Graph_Type'] == t]
                 ['Time_Iso_Mean'].values for t in types]
    ks_times = [df[df['Graph_Type'] == t]
                ['Time_KS_Mean'].values for t in types]

    ax1.boxplot(iso_times, positions=np.arange(len(types)) - 0.15, widths=0.25, patch_artist=True,
                boxprops=dict(facecolor='#008080', alpha=0.6), labels=types)
    ax1.boxplot(ks_times, positions=np.arange(len(types)) + 0.15, widths=0.25, patch_artist=True,
                boxprops=dict(facecolor='#FF1869', alpha=0.6))
    ax1.set_ylabel("Runtime (s)")
    ax1.set_title("Runtime Distribution by Topology (Isolating vs KS)")
    ax1.set_yscale('log')

    # paired scatter with connecting lines to show per-sample change
    for i, t in enumerate(types):
        sub = df[df['Graph_Type'] == t]
        ax2.scatter(np.full(len(sub), i) - 0.05, sub['Time_Iso_Mean'],
                    color='#008080', alpha=0.8, s=30)
        ax2.scatter(np.full(len(sub), i) + 0.05, sub['Time_KS_Mean'],
                    color='#FF1869', alpha=0.8, s=30)
        for _, row in sub.iterrows():
            ax2.plot([i - 0.05, i + 0.05], [row['Time_Iso_Mean'], row['Time_KS_Mean']],
                     color='gray', alpha=0.3)
    ax2.set_xticks(range(len(types)))
    ax2.set_xticklabels(types, rotation=25)
    ax2.set_yscale('log')
    ax2.set_title("Paired Runtime Changes by Graph Sample")

    fig.suptitle("Paired Runtime Comparison Across Topologies")
    fig.savefig(os.path.join(
        output_dir, "2_paired_runtime_by_topology.png"), bbox_inches='tight')
    fig.savefig(os.path.join(
        output_dir, "2_paired_runtime_by_topology.pdf"), bbox_inches='tight')
    plt.close(fig)

    # 3) Success rate and accuracy comparison (grouped bars)
    summary = df.groupby('Graph_Type', observed=True).agg({
        'Success_Rate_Iso': 'mean',
        'Success_Rate_KS': 'mean',
        'Err_Iso_Mean': 'mean',
        'Err_KS_Mean': 'mean',
        'Time_Iso_Mean': 'mean',
        'Time_KS_Mean': 'mean'
    }).reset_index()
    x = np.arange(len(summary))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width/2, summary['Success_Rate_Iso'],
           width, label='Iso Success', color='#008080')
    ax.bar(x + width/2, summary['Success_Rate_KS'],
           width, label='KS Success', color='#FF1869')
    ax.set_xticks(x)
    ax.set_xticklabels(summary['Graph_Type'], rotation=25)
    ax.set_ylabel("Success Rate")
    ax.set_title("Success Rate by Topology")
    ax.legend()
    fig.savefig(os.path.join(
        output_dir, "3_success_rate_by_topology.png"), bbox_inches='tight')
    fig.savefig(os.path.join(
        output_dir, "3_success_rate_by_topology.pdf"), bbox_inches='tight')
    plt.close(fig)

    # 4) Runtime ratio vs density with regression and binned heatmap
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.scatter(df['Density'], df['Time_Ratio_KS_to_Iso'], alpha=0.75, s=30)
    ax1.set_xscale('linear')
    ax1.set_yscale('log')
    ax1.set_xlabel('Density')
    ax1.set_ylabel('Time Ratio (KS / Iso)')
    ax1.set_title('Runtime Ratio vs Graph Density')
    if len(df) > 2:
        slope, intercept, r_val, _, _ = linregress(
            np.log(df['Density'] + 1e-12), np.log(df['Time_Ratio_KS_to_Iso'] + 1e-12))
        x_vals = np.linspace(df['Density'].min(), df['Density'].max(), 100)
        ax1.plot(x_vals, np.exp(intercept) * (x_vals**slope), '--', color='black',
                 label=f'Fit slope={slope:.2f}, $R^2$={r_val**2:.2f}')
        ax1.legend()

    # heatmap: average time ratio per Nodes bin vs Density bin
    df['Node_Bin'] = pd.qcut(df['Nodes'], q=4, duplicates='drop')
    pivot = df.pivot_table(index='Node_Bin', columns='Density_Bin',
                           values='Time_Ratio_KS_to_Iso', aggfunc='mean')
    im = ax2.imshow(pivot.fillna(0).values, aspect='auto',
                    interpolation='nearest', cmap='RdYlBu_r')
    ax2.set_yticks(range(len(pivot.index)))
    ax2.set_yticklabels([str(i) for i in pivot.index])
    ax2.set_xticks(range(len(pivot.columns)))
    ax2.set_xticklabels([str(i) for i in pivot.columns], rotation=45)
    ax2.set_title('Mean Time Ratio (KS/Iso) — Nodes Bin x Density Bin')
    fig.colorbar(im, ax=ax2, label='KS / Iso Time')
    fig.tight_layout()
    fig.savefig(os.path.join(
        output_dir, "4_ratio_vs_density_and_heatmap.png"), bbox_inches='tight')
    fig.savefig(os.path.join(
        output_dir, "4_ratio_vs_density_and_heatmap.pdf"), bbox_inches='tight')
    plt.close(fig)

    # 5) Statistical tests per topology (paired) and export a summary CSV
    stats_rows = []
    for t in df['Graph_Type'].unique():
        sub = df[df['Graph_Type'] == t]
        if len(sub) >= 2:
            try:
                stat, p_w = wilcoxon(sub['Time_Iso_Mean'], sub['Time_KS_Mean'])
            except Exception:
                stat, p_w = (np.nan, np.nan)
            t_stat, p_t = ttest_rel(sub['Time_Iso_Mean'], sub['Time_KS_Mean'])
            mean_ratio = sub['Time_Ratio_KS_to_Iso'].mean()
            median_ratio = sub['Time_Ratio_KS_to_Iso'].median()
            stats_rows.append({
                'Graph_Type': t,
                'N': len(sub),
                'Mean_Time_Iso': sub['Time_Iso_Mean'].mean(),
                'Mean_Time_KS': sub['Time_KS_Mean'].mean(),
                'Mean_Ratio_KS_to_Iso': mean_ratio,
                'Median_Ratio_KS_to_Iso': median_ratio,
                'Wilcoxon_stat': stat,
                'Wilcoxon_p': p_w,
                'PairedT_stat': t_stat,
                'PairedT_p': p_t,
                'Mean_Error_Iso': sub['Err_Iso_Mean'].mean(),
                'Mean_Error_KS': sub['Err_KS_Mean'].mean()
            })

    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(os.path.join(
        output_dir, "summary_by_topology_statistics.csv"), index=False)

    # final combined figure: accuracy vs runtime ratio scatter (helps choose trade-offs)
    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(df['Time_Ratio_KS_to_Iso'], df['Rel_Error_Diff'],
                    c=df['Nodes'], cmap='viridis', alpha=0.85, s=40)
    cbar = fig.colorbar(sc)
    cbar.set_label('Nodes')
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax.set_xscale('log')
    ax.set_xlabel('Time Ratio (KS / Iso) [log scale]')
    ax.set_ylabel('Error Diff (KS - Iso)')
    ax.set_title(
        'Trade-off: Runtime Ratio vs Accuracy Difference (positive => KS worse)')
    fig.savefig(os.path.join(
        output_dir, "5_tradeoff_runtime_vs_accuracy.png"), bbox_inches='tight')
    fig.savefig(os.path.join(
        output_dir, "5_tradeoff_runtime_vs_accuracy.pdf"), bbox_inches='tight')
    plt.close(fig)

    df.to_csv(os.path.join(output_dir, "raw_results.csv"), index=False)
    print(f"Charts and statistical summary saved to {output_dir}/")


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
