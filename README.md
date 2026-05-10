# Dijkstra Skip-Ratio Characterization

Companion code for the paper:

> **Empirical Characterization of Skip-Ratio in Lazy-Deletion Dijkstra on Random Graphs**  
> G. W. Park, 2026.

## Overview

This repository provides all experiment scripts, theory probes, and CSV result data to reproduce the figures and tables in the paper. The core finding is that the *skip-ratio* (fraction of heap-pops that are stale/skipped) in lazy-deletion Dijkstra is well-approximated by:

```
skip_ratio ≈ 1 - 0.884 / k^0.262   (R² = 0.984, k ∈ [4, 64])
```

where `k` is the mean out-degree of the random directed graph, under U[1,100] edge weights.

## Repository Structure

```
.
├── paper.tex / paper.pdf       # Main manuscript
├── requirements.txt            # Python dependencies
│
├── theory_probe_v4.py          # §IV theory probe (skip-ratio vs k, distribution)
├── theory_probe_v3.py          # earlier version
├── theory_probe_v2.py          # §III calibration (V=3,000 baseline)
│
├── dijkstra_experiment7.py/cpp # §V Python / C++ timing experiments
├── dijkstra_experiment9.py/cpp # §V extended timing
├── dijkstra_experimentA.cpp    # §VI ca-GrQc real-graph experiment
├── dijkstra_experimentA_plot.py
├── dijkstra_experimentB.cpp    # §VI additional real-graph
├── dijkstra_experimentB_plot.py
├── dijkstra_experimentC.py     # §VI Python real-graph
│
├── resultsTheory/              # CSV outputs of theory_probe_v4.py
│   ├── alpha_fits_*.csv        # Power-law α fits for 3 distributions × 4 k-ranges
│   ├── local_slope_*.csv       # Local slopes and H_k ratios
│   └── skip_ratio_by_dist_*.csv
│
└── run_experiment.sh           # convenience wrapper
```

## Reproducing the Results

### Requirements

```bash
pip install -r requirements.txt
```

Python 3.11+, NumPy, SciPy, Matplotlib.  
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
| C++ lazy vs. indexed heap | **2.9×** |
| C++ peak (large k) | **4.3×** |
| Python lazy vs. indexed heap | **1.5×** |

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
