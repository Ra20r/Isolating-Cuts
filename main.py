import pandas as pd
from benchmarking import BenchmarkRunner

from graph_generators.erdos_renyi import generate_er
from graph_generators.barabasi_albert import generate_ba

from algorithms.karger_stein import karger_stein_wrapper
from algorithms.isolating_cut import isolating_cut


def main():
    algorithms_to_test = {
        'karger_stein': karger_stein_wrapper,
        'isolating_cut': isolating_cut,
    }

    graph_generators = {
        'ER': generate_er,
        'BA': generate_ba,
    }

    models = ['ER', 'BA']

    # Graph sizes (n)
    # Karger-Stein is O(n^2 log n), but the base case is slow. Using small n for now.
    n_values = [20, 40, 60, 80, 100]

    # this is for each (model, n) pair. >= 30 for decent stats.
    R_TRIALS = 100

    model_params = {
        'ER': {'p': 0.1},  # G(n, p) with p=0.1
        'BA': {'m': 3}     # G(n, m) with m=3 new edges per node
    }

    runner = BenchmarkRunner(algorithms_to_test, graph_generators)
    results_df = runner.run(
        models=models,
        n_values=n_values,
        trials=R_TRIALS,
        model_params=model_params
    )

    pd.set_option('display.width', 1000)
    pd.set_option('display.max_rows', None)

    print("\nBenchmark Results:")
    print(results_df)

    results_df.to_csv("benchmark_results.csv", index=False)
    print("\nResults saved to benchmark_results.csv")


if __name__ == "__main__":
    main()
