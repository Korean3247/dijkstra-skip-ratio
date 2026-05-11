# Dijkstra Skip-Ratio Characterization

Companion code for the paper:

> **Empirical Characterization of Skip-Ratio in Lazy-Deletion Dijkstra on Random Graphs**  
> G. W. Park, 2026.

## Overview

This repository provides all experiment scripts, theory probes, and CSV result data to reproduce the figures and tables in the paper. The core finding is that the *skip-ratio* (fraction of heap-pops that are stale/skipped) in lazy-deletion Dijkstra is well-approximated by:

```
skip_ratio ≈ 1 - 0.884 / k^0.262   (R² = 0.984, k ∈ [4, 512])
```

where `k` is the mean degree of the random **undirected** graph, under U[1,100] edge weights.

## Repository Structure

```
.
├── paper.tex / paper.pdf           # Main manuscript
├── requirements.txt                # Python dependencies
│
├── theory_probe_v4.py              # §IV theory probe (skip-ratio vs k, distribution)
├── theory_probe_v3.py              # earlier version
├── theory_probe_v2.py              # §III calibration (V=3,000 baseline)
│
├── dijkstra_experiment7.py/cpp     # §V Python / C++ timing experiments
├── dijkstra_experiment9.py/cpp     # §V extended timing
├── dijkstra_experimentA.cpp        # §VI ca-GrQc real-graph experiment
├── dijkstra_experimentA_plot.py
├── dijkstra_experimentB.cpp        # §VI additional real-graph
├── dijkstra_experimentB_plot.py
├── dijkstra_experimentC.py         # §VI Python real-graph
│
├── dijkstra_heap_compare.py        # §VII Python: indexed/d-ary heap comparison
├── dijkstra_heap_compare.cpp       # §VII C++:  indexed/d-ary heap comparison
├── source_sensitivity.py           # §VII source-vertex sensitivity experiment
├── bootstrap_ci.py                 # §III bootstrap 95% CI for α and c
│
├── resultsTheory/                  # CSV outputs of theory_probe_v4.py
│   ├── alpha_fits_*.csv            # Power-law α fits for 3 distributions × 4 k-ranges
│   ├── local_slope_*.csv           # Local slopes and H_k ratios
│   ├── skip_ratio_by_dist_*.csv
│   └── bootstrap_ci_*.csv          # Bootstrap 95% CI for U[1,100], k∈[4,64]
│
├── resultsHeapCompare/             # CSV outputs of heap comparison experiments
│   ├── heap_compare_*.csv          # Python results (lazy/indexed/4-ary/8-ary)
│   └── heap_compare_cpp_*.csv      # C++ results (same variants)
│
├── resultsSourceSensitivity/       # CSV outputs of source_sensitivity.py
│   └── source_sensitivity_*.csv
│
└── run_experiment.sh               # convenience wrapper
```

### Heap Comparison (§VII)

```bash
# Python (lazy binary vs indexed vs 4-ary vs 8-ary)
python dijkstra_heap_compare.py

# C++ (same variants, -O2 optimized)
g++ -O2 -std=c++17 -o heap_compare dijkstra_heap_compare.cpp
./heap_compare
```

**Key finding:** In Python, lazy heapq beats indexed heap by 3–4× at low k. In C++, the pattern reverses: indexed heap is 1.4–2.2× faster than lazy binary for k≥8, and the 4-ary lazy heap is 1.4–1.6× faster than binary heapq across all k.

### Source Sensitivity (§VII)

```bash
python source_sensitivity.py   # ~2 min; outputs to resultsSourceSensitivity/
```

Confirms skip_ratio is source-vertex independent: CV ≤ 2.5% across 20 random sources.

### Bootstrap CI (§III)

```bash
python bootstrap_ci.py   # requires resultsTheory/skip_ratio_by_dist_*.csv
```

Reports 95% bootstrap CI for α and c over B=10,000 resamples.

## Reproducing the Results

### Requirements

```bash
pip install -r requirements.txt
```

Python 3.11+ (tested on 3.14.2), NumPy ≥2.4 (tested on 2.4.4), SciPy, Matplotlib.  
C++ experiments require g++ with `-O2` and OpenMP.

### Theory Probe (§III–IV)

```bash
python theory_probe_v4.py
# Outputs CSVs + plots to resultsTheory/
```

### Timing Experiments (§V)

```bash
# Python
python dijkstra_experiment7.py

# C++
g++ -O2 -std=c++17 -o exp7 dijkstra_experiment7.cpp
./exp7
```

### Real-Graph Experiments (§VI)

```bash
g++ -O2 -std=c++17 -fopenmp -o expA dijkstra_experimentA.cpp
./expA
python dijkstra_experimentA_plot.py
```

The ca-GrQc network is downloaded automatically from SNAP (Stanford) on first run.

## Key Results

| Setting | Geometric mean speedup |
|---|---|
| C++ lazy vs. indexed heap (Dijkstra timing, §V) | **2.9×** |
| C++ peak speedup with Dial's bucket queue | **4.3×** |
| Python lazy vs. indexed heap (§V) | **1.5×** |
| C++ indexed heap vs. lazy binary (heap compare, §VII) | **1.4–2.2× faster** |
| C++ 4-ary heap vs. lazy binary (heap compare, §VII) | **1.4–1.6× faster** |
| Source-vertex CV (skip_ratio stability) | **≤ 2.5%** |

See `resultsTheory/alpha_fits_*.csv` for fitted power-law exponents and R² values across distributions.

## Citation

If you use this code, please cite:

```bibtex
@article{park2026dijkstra,
  title={Empirical Characterization of Skip-Ratio in Lazy-Deletion Dijkstra on Random Graphs},
  author={Park, G. W.},
  year={2026}
}
```

## License

MIT
