# Comparative Study of Randomized Global Min-Cut Algorithms

This repository contains an implementation and empirical evaluation of two randomized algorithms for the **global minimum cut problem** on weighted undirected graphs:

- **Karger–Stein** (randomized contraction algorithm)  
  [Karger & Stein, JACM 1996](https://doi.org/10.1145/234533.234534)

- **Isolating Cuts** (Li & Panigrahi, 2022)  
  [arXiv:2111.02008](https://arxiv.org/abs/2111.02008)

The project focuses on **practical performance**, **empirical correctness**, and **consistency with theoretical runtime guarantees**, using controlled benchmarking across multiple random graph models.

---

## Problem Overview

Given a connected, weighted, undirected graph $G = (V, E)$, the **global minimum cut** problem asks for a partition of vertices into two non-empty sets that minimizes the total weight of edges crossing the cut.

When the terminal set $T = V$, the Steiner minimum cut problem reduces to the global min-cut problem. This project studies two randomized algorithms for this setting and compares their empirical behavior.

---

## Algorithms Implemented

### 1. Karger–Stein Algorithm

- Random edge contraction with recursion
- Uses Union–Find for efficient contractions
- Edges sampled with probability proportional to their weights
- Graph is repeatedly contracted to $\lceil n / \sqrt{2} + 1 \rceil$ vertices
- Repeated $\log^2 n$ times to amplify success probability

**Theoretical runtime:** $O(n^2 \log^3 n)$


---

### 2. Isolating Cuts Algorithm (Li–Panigrahi)

- Randomly samples a subset of terminals
- Uses bitwise set-splitting to isolate candidate vertices
- Reduces the global min-cut problem to polylogarithmically many
  minimum $s$–$t$ cut computations
- Max-flow subroutine implemented using **Dinic’s algorithm**

**Theoretical runtime:** $O(\log^4 n \cdot T_{\text{max-flow}})$

---

## Graph Models Evaluated

Each algorithm is benchmarked on the following random graph families:

- **Erdős–Rényi (ER)**
- **Barabási–Albert (BA)**
- **Watts–Strogatz (WS)**
- **Power-Law Cluster (PLC)**
- **Stochastic Block Model (SBM)**

### Graph Generation Details

- Number of nodes: 20 → 500 (step size 20)
- Edge weights: integers uniformly sampled from $[1, 9]$
- Average degree: $O(\log n)$ for most graph families
- For disconnected graphs, only the largest connected component is used
- One graph instance per size per model

---

## Experimental Setup

- **Language:** Python 3.14  
- **Machine:** Mac Studio (M4 Max)  
- **Concurrency:** 15 parallel workers  
- **Random seed:** 42  
- **Trials:** 3 per graph after one warm-up run  

### Evaluation Metrics

- Runtime (mean and standard deviation)
- Correctness against deterministic ground truth
- Empirical scaling vs theoretical complexity
- Graph statistics (edge count, density)

Ground-truth min-cuts are computed using the **Stoer–Wagner** deterministic algorithm.

### Repository Tree

```python
.
├── algorithms/
│   ├── Dinic.py                 # Max-flow (Dinic) used by Isolating Cuts
│   ├── karger_stein.py          # Karger–Stein randomized contraction algorithm
│   ├── isolating_cut.py         # Isolating Cuts implementation
│   ├── isolating_cuts_*.py      # Variants and experimental refinements
│   ├── expander_decomp.py       # Supporting routines for the deterministic implementation (partial)
│   └── sparsify_terminals.py    # Supporting routines for the deterministic implementation (partial)
│
├── graph_generators/            # Scripts here are not used in actual test scripts
│   ├── erdos_renyi.py           # Erdős–Rényi graph generator
│   └── barabasi_albert.py       # Barabási–Albert graph generator
│
├── benchmarking.py              # Runtime benchmarking and trial orchestration
├── data_analysis.py             # Scaling analysis and regression on runtime data
├── main.py                      # Entry point for running full experiments
│
├── test_data/                   # Real-world graph edge lists (for sanity checks)
├── test*.py                     # Unit tests, stress tests, and concurrency tests
├── requirements.txt             # Python dependencies
└── README.md
```

---

## Results Summary

### Empirical Correctness

- All runs produce the correct minimum cut value except for **one SBM instance**
- The deviation is fractional (< 0.1) despite integer edge weights
- Attributed to floating-point behavior rather than algorithmic error
- Overall correctness aligns with theoretical success guarantees

---

### Runtime Scaling

- Empirical runtime matches theoretical complexity closely
- Log–log regression yields:
  - $R^2 \approx 0.99$ across all graph types
  - Regression slopes within $1.0 \pm 0.1$
- Confirms faithful implementation of both algorithms

---

### Performance Comparison

- **Isolating Cuts** is significantly faster than **Karger–Stein** on sparse graphs
- **Karger–Stein** exhibits large constant factors due to repeated graph reconstruction
- **SBM graphs** slow down Isolating Cuts due to higher density and max-flow costs
- Dense graphs expose the limitations of Dinic’s algorithm in Python

---

### Key Takeaways

- Theoretical improvements in min-cut algorithms translate clearly into practice
- Isolating Cuts dramatically reduces runtime on sparse graphs
- Dense graphs highlight max-flow bottlenecks
- Python is suitable for experimental algorithmics, but constant factors matter

---

### Authors

- Matthew Castro  
- Sagnik Chakraborty
- Zimo Luo  
- Shuo Zhang  