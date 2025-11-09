import numpy as np
import pandas as pd
import time
from typing import List, Dict, Callable, Any


class BenchmarkRunner:
    """
    Handles running benchmarks for different graph models and algorithms.
    """

    def __init__(self, algorithms: Dict[str, Callable],
                 generators: Dict[str, Callable]):
        """
        Args:
            algorithms (Dict[str, Callable]):
                Dict of {'algo_name': algorithm_function}
                Each function must accept one arg: an (n, n) numpy matrix.

            generators (Dict[str, Callable]):
                Dict of {'model_name': generator_function}
                Each function must accept n and **kwargs.
        """
        self.algorithms = algorithms
        self.generators = generators

    def run(self,
            models: List[str],
            n_values: List[int],
            trials: int,
            model_params: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        Runs the full benchmark.

        Args:
            models (List[str]): List of model names (e.g., ['ER', 'BA']).
            n_values (List[int]): List of graph sizes (n).
            trials (int): Number of trials to run for each (model, n) pair.
            model_params (Dict): Parameters for each model generator.
                                 e.g., {'ER': {'p': 0.1}, 'BA': {'m': 3}}

        Returns:
            pd.DataFrame: A DataFrame with all results.
        """
        all_results = []

        for model_name in models:
            if model_name not in self.generators:
                print(
                    f"Warning: Generator '{model_name}' not found. Skipping.")
                continue
            gen_func = self.generators[model_name]
            params = model_params.get(model_name, {})

            for n in n_values:
                print(
                    f"--- Running: Model={model_name}, n={n}, Trials={trials} ---")

                # raw results for this (model, n) block
                trial_results = {name: {'times': [], 'cuts': []}
                                 for name in self.algorithms}

                for i in range(trials):
                    # ensures all algos are benchmarked on the same graph
                    graph = gen_func(n=n, **params)

                    for algo_name, algo_func in self.algorithms.items():
                        # pass a copy in case the algorithm mutates it
                        graph_copy = np.copy(graph)

                        start_time = time.perf_counter()
                        cut_val = algo_func(graph_copy)
                        end_time = time.perf_counter()

                        trial_results[algo_name]['times'].append(
                            end_time - start_time)
                        trial_results[algo_name]['cuts'].append(cut_val)

                for algo_name, data in trial_results.items():
                    all_results.append({
                        'model': model_name,
                        'n': n,
                        'algorithm': algo_name,
                        'trials': trials,
                        'mean_time_s': np.mean(data['times']),
                        'std_time_s': np.std(data['times']),
                        'mean_cut': np.mean(data['cuts']),
                        'std_cut': np.std(data['cuts']),
                        'min_found_cut': np.min(data['cuts']),
                        'max_found_cut': np.max(data['cuts']),
                    })

        print("--- Benchmark Complete ---")
        return pd.DataFrame(all_results)
