"""
source_sensitivity.py
=====================
Tests whether skip_ratio depends on the choice of source vertex.
Runs Dijkstra from 20 randomly chosen source vertices for each (V, k) pair
and reports mean ± std of skip_ratio across sources.

Key question: is the single-source-vertex design (node 0) in the main
experiments a methodological concern?
"""

import heapq
import random
import csv
import math
from datetime import datetime

def make_graph(V, k, seed=None):
    rng = random.Random(seed)
    graph = [[] for _ in range(V)]
    m = V * k
    for _ in range(m):
        u = rng.randint(0, V-1)
        v = rng.randint(0, V-1)
        w = rng.randint(1, 100)
        graph[u].append((w, v))
    # spanning path for connectivity
    for i in range(V-1):
        w = rng.randint(1, 100)
        graph[i].append((w, i+1))
    return graph

def dijkstra_skip(graph, src, V):
    dist = [float('inf')] * V
    dist[src] = 0
    heap = [(0, src)]
    push_cnt = 1
    pop_cnt = 0
    skip_cnt = 0
    while heap:
        d, u = heapq.heappop(heap)
        pop_cnt += 1
        if d > dist[u]:
            skip_cnt += 1
            continue
        for w, v in graph[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
                push_cnt += 1
    return skip_cnt / pop_cnt if pop_cnt > 0 else 0.0

# Parameters
V_LIST   = [1000, 3000, 10000]
K_LIST   = [4, 8, 16, 32, 64, 128]
N_GRAPHS = 5    # different graph instances per (V, k)
N_SOURCES = 20  # source vertices per graph

print(f"Source sensitivity: {N_GRAPHS} graphs × {N_SOURCES} sources per (V,k)")
print(f"V={V_LIST}, k={K_LIST}\n")

records = []
for V in V_LIST:
    for k in K_LIST:
        all_skip = []
        for g_seed in range(N_GRAPHS):
            graph = make_graph(V, k, seed=g_seed*9999 + V + k)
            # sample N_SOURCES distinct source vertices
            rng = random.Random(g_seed)
            sources = rng.sample(range(V), min(N_SOURCES, V))
            for src in sources:
                sr = dijkstra_skip(graph, src, V)
                all_skip.append(sr)
        mean_sr = sum(all_skip) / len(all_skip)
        std_sr  = math.sqrt(sum((x-mean_sr)**2 for x in all_skip) / len(all_skip))
        min_sr  = min(all_skip)
        max_sr  = max(all_skip)
        cv      = std_sr / mean_sr if mean_sr > 0 else 0  # coefficient of variation
        records.append({
            'V': V, 'k': k,
            'mean': round(mean_sr, 4),
            'std':  round(std_sr,  4),
            'min':  round(min_sr,  4),
            'max':  round(max_sr,  4),
            'cv_pct': round(cv*100, 2),
            'n_obs': len(all_skip),
        })
        print(f"V={V:>6} k={k:>3}  mean={mean_sr:.4f}  std={std_sr:.4f}  "
              f"[{min_sr:.4f},{max_sr:.4f}]  CV={cv*100:.1f}%")

# Save CSV
import os
out_dir = "/Users/gwpark/Downloads/Algorithm Research/resultsSourceSensitivity"
os.makedirs(out_dir, exist_ok=True)
ts = datetime.now().strftime('%Y%m%d_%H%M%S')
csv_path = f"{out_dir}/source_sensitivity_{ts}.csv"
with open(csv_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['V','k','mean','std','min','max','cv_pct','n_obs'])
    w.writeheader()
    w.writerows(records)
print(f"\nCSV: {csv_path}")

# Summary
print("\n=== SUMMARY (max CV across all conditions) ===")
max_cv = max(r['cv_pct'] for r in records)
max_std = max(r['std'] for r in records)
print(f"Max CV:  {max_cv:.1f}%")
print(f"Max std: {max_std:.4f}")
print(f"Conclusion: skip_ratio {'IS' if max_cv < 5 else 'IS NOT'} stable across source vertices (CV < 5%)")
