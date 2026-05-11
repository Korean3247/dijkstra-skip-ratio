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
│   (all paper-referenced CSV results are committed to Git)
├── results2/experiment2_*.csv      # §III V-independence
├── results3/experiment3_*.csv      # §III extended V-independence
├── results5/experiment5_*.csv      # §V  threshold heuristic (Experiment 5)
├── results6/experiment6_*.csv      # §V  Python Fibonacci vs heapq (Exp. 6)
├── results7/experiment7_*.csv      # §V  Python timing (Exp. 7)
├── results8/experiment8_*.csv      # §V  Dial's natural pruning (Exp. 8)
├── results9/experiment9_*.csv      # §V  float-weight comparison (Exp. 9, Python+C++)
├── results10/experiment10_*.csv    # §VI SNAP real-world validation (Exp. 10)
├── resultsA/experimentA_*.csv      # §IV large-V experiment
├── resultsB/experimentB_*.csv      # §VI Dial's W-sensitivity
├── resultsC/experimentC_*.csv      # §VI Python real-graph
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

### Quick Start (all experiments)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
make reproduce-all   # runs all experiments in dependency order
make pdf             # recompile paper.tex → paper.pdf
```

Individual targets:

```
make reproduce-theory      # §III–IV power-law fits         (~10–20 min)
make reproduce-bootstrap   # §III    bootstrap 95% CI       (requires theory output)
make reproduce-sensitivity # §VII    source-vertex CV       (~2 min)
make reproduce-heap-py     # §VII    Python heap comparison (~5–10 min)
make reproduce-heap-cpp    # §VII    C++ heap comparison    (~2 min)
make reproduce-tables      # §V–VI   timing experiments
make help                  # show all targets
```

### Requirements

Python 3.11+ (tested on 3.14.2). All required packages are listed in
`requirements.txt`. The code uses only stable NumPy APIs; `numpy>=1.24`
is the minimum, but NumPy 2.4.4 was used for the paper's results.  
C++ experiments require g++ with `-O2` (tested with Apple clang 17); OpenMP
is needed only for `dijkstra_experimentA.cpp`. On Apple clang, install
`libomp` first (`brew install libomp`) or use GCC (`make GPP=g++-14 reproduce-real`).

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

| Setting | Result |
|---|---|
| C++ Dial's bucket queue vs. heapq (§V, geometric mean) | **2.9× faster** |
| C++ Dial's bucket queue peak (large k, §V) | **4.3× faster** |
| Python Dial's vs. heapq geometric mean (§V) | **1.5× faster** |
| C++ indexed heap vs. lazy binary (supplemental §VII, Apple M3, directed ER) | **1.4–2.2× faster** at k≥8 |
| C++ 4-ary lazy heap vs. binary heapq (supplemental §VII, Apple M3, directed ER) | **1.4–1.6× faster** |
| Source-vertex CV (skip_ratio stability, §VII) | **≤ 2.5%** across 20 sources |

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
