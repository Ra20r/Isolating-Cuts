import pandas as pd
import os
import random
import time
import numpy as np
import networkx as nx
from tqdm import tqdm
import concurrent.futures
import multiprocessing

from algorithms.isolating_cuts_3 import isolating_cut
from algorithms.karger_stein import karger_stein_wrapper

RNG_SEED = 42
OUTPUT_DIR = "test4_results_concurrent"
CSV_FILENAME = "raw_results.csv"

SAMPLES_PER_TYPE = 20
ITERATIONS_PER_GRAPH = 5
START_N = 20
STEP_N = 20

GRAPH_TYPES = ['erdos_renyi', 'barabasi_albert',
               'stochastic_block', 'watts_strogatz', 'powerlaw_cluster']

np.random.seed(RNG_SEED)
random.seed(RNG_SEED)


def generate_graph(g_type, n, seed):
    """
    Generates different graph topologies.
    (Code identical to your original function)
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

        probs = np.zeros((k, k))
        for i in range(k):
            for j in range(i, k):
                if i == j:
                    probs[i][j] = intra
                else:
                    val = max(0.0, inter + rng_g.normal(0, 0.02))
                    probs[i][j] = val
                    probs[j][i] = val

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


# worker
def process_single_graph(task_args):
    """
    Worker function to handle ONE specific graph configuration.
    It generates the graph and runs the iterations locally.
    """
    g_type, n_target, graph_seed = task_args

    G = generate_graph(g_type, n_target, graph_seed)
    Adj = nx.to_numpy_array(G, weight='weight')

    n_actual = G.number_of_nodes()
    m_actual = G.number_of_edges()

    true_val, _ = nx.stoer_wagner(G, weight='weight')

    pred_iso = (m_actual) * (np.log2(n_actual)**6)
    pred_ks = (n_actual**2) * (np.log2(n_actual)**3)

    times_iso, times_ks = [], []
    errs_iso, errs_ks = [], []
    found_exact_iso, found_exact_ks = 0, 0

    # do NOT split these iterations (of the same graph) across processes to preserve cache locality
    for _ in range(ITERATIONS_PER_GRAPH):
        # Isolating Cuts
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

        # Karger-Stein
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

    return {
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


def run_experiment_concurrently():
    results_data = []

    # the task is a tuple of (graph_type, n_target)
    # so there are samples per type times number of types total tasks
    tasks = []
    for g_type in GRAPH_TYPES:
        for i in range(SAMPLES_PER_TYPE):
            n_target = START_N + i * STEP_N
            graph_seed = RNG_SEED + (i * 1000)
            tasks.append((g_type, n_target, graph_seed))

    total_jobs = len(tasks)

    # determine logical cores and leave 1 core free for OS
    # does not differentiate between performance and efficiency cores. not sure if this would impact the benchmarking
    # since each process is sent to an arbitrary core and it's possible that performance core might run it faster
    max_workers = max(1, multiprocessing.cpu_count() - 1)

    print(
        f"Spinning up pool with {max_workers} workers for {total_jobs} jobs...")

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_graph, t): t for t in tasks}

        with tqdm(total=total_jobs, desc="Benchmarking (Concurrent)") as pbar:
            for future in concurrent.futures.as_completed(futures):
                try:
                    data = future.result()
                    results_data.append(data)
                except Exception as exc:
                    print(f"Job generated an exception: {exc}")
                finally:
                    pbar.update(1)

    return pd.DataFrame(results_data)


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIR, CSV_FILENAME)

    print("Starting Concurrent Benchmark...")

    df_results = run_experiment_concurrently()

    df_results = df_results.sort_values(by=['Graph_Type', 'Nodes'])

    df_results.to_csv(csv_path, index=False)
    print(f"\nRaw data saved to {csv_path}")

    # will figure out the plots later. csv is more important
