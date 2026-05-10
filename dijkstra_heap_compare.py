"""
dijkstra_heap_compare.py
========================
Comparison of Dijkstra implementations across heap variants:
  1. Lazy-deletion binary heap (heapq)   — baseline in the paper
  2. Indexed binary heap                 — O(log V) decrease-key, zero stale pops
  3. Lazy-deletion 4-ary heap            — branching factor d=4, fewer comparisons
  4. Lazy-deletion 8-ary heap            — branching factor d=8

Metrics:
  - push_count / V  (heap operations per vertex)
  - skip_ratio      (stale pops / total pops, 0 for indexed heap)
  - wall-clock time (seconds, mean over reps)
  - speedup vs indexed heap

Graph model: directed Erdős–Rényi G(V, m=k*V), U[1,100] integer weights.
"""

import heapq
import random
import time
import csv
import math
from datetime import datetime

# ──────────────────────────────────────────────
# Graph generation
# ──────────────────────────────────────────────

def make_graph(V, k, weight_range=(1, 100), seed=None):
    """Return adjacency list: graph[u] = [(w, v), ...]"""
    rng = random.Random(seed)
    graph = [[] for _ in range(V)]
    m = V * k
    for _ in range(m):
        u = rng.randint(0, V - 1)
        v = rng.randint(0, V - 1)
        w = rng.randint(*weight_range)
        graph[u].append((w, v))
    return graph


# ──────────────────────────────────────────────
# 1. Lazy-deletion binary heap (standard heapq)
# ──────────────────────────────────────────────

def dijkstra_lazy(graph, src, V):
    dist = [float('inf')] * V
    dist[src] = 0
    heap = [(0, src)]
    push_count = 0
    pop_count = 0
    skip_count = 0

    while heap:
        d, u = heapq.heappop(heap)
        pop_count += 1
        if d > dist[u]:
            skip_count += 1
            continue
        for w, v in graph[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
                push_count += 1

    skip_ratio = skip_count / pop_count if pop_count > 0 else 0.0
    return dist, push_count, skip_ratio


# ──────────────────────────────────────────────
# 2. Indexed binary heap (O(log V) decrease-key)
# ──────────────────────────────────────────────

class IndexedHeap:
    """
    Min-heap with O(log n) decrease-key via position array.
    Stores (key, index) pairs; position[i] = heap position of item i.
    """
    __slots__ = ('heap', 'pos', 'size')

    def __init__(self, n):
        self.heap = [(float('inf'), i) for i in range(n)]  # (key, id)
        self.pos  = list(range(n))                          # pos[id] = heap index
        self.size = n

    def _swap(self, i, j):
        h, p = self.heap, self.pos
        h[i], h[j] = h[j], h[i]
        p[h[i][1]] = i
        p[h[j][1]] = j

    def _sift_up(self, i):
        while i > 0:
            parent = (i - 1) >> 1
            if self.heap[parent][0] > self.heap[i][0]:
                self._swap(parent, i)
                i = parent
            else:
                break

    def _sift_down(self, i):
        n = self.size
        while True:
            left  = 2 * i + 1
            right = 2 * i + 2
            smallest = i
            if left  < n and self.heap[left][0]  < self.heap[smallest][0]:
                smallest = left
            if right < n and self.heap[right][0] < self.heap[smallest][0]:
                smallest = right
            if smallest == i:
                break
            self._swap(i, smallest)
            i = smallest

    def decrease_key(self, idx, new_key):
        """Update key of item `idx` if new_key is smaller."""
        i = self.pos[idx]
        if new_key < self.heap[i][0]:
            self.heap[i] = (new_key, idx)
            self._sift_up(i)

    def pop_min(self):
        """Remove and return (key, id) of minimum."""
        if self.size == 0:
            raise IndexError("pop from empty heap")
        # swap root with last element
        self._swap(0, self.size - 1)
        self.size -= 1
        self._sift_down(0)
        return self.heap[self.size]

    def empty(self):
        return self.size == 0


def dijkstra_indexed(graph, src, V):
    ih = IndexedHeap(V)
    ih.decrease_key(src, 0)
    dist = [float('inf')] * V
    dist[src] = 0
    pop_count = 0
    decrease_count = 0   # = push_count analogue

    while not ih.empty():
        d, u = ih.pop_min()
        pop_count += 1
        if d == float('inf'):
            break
        for w, v in graph[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                ih.decrease_key(v, nd)
                decrease_count += 1

    # indexed heap: exactly V pops, 0 stale pops
    return dist, decrease_count, 0.0  # skip_ratio always 0


# ──────────────────────────────────────────────
# 3 & 4. Lazy-deletion d-ary heap
# ──────────────────────────────────────────────

class DaryHeap:
    """Simple d-ary min-heap using a Python list."""
    __slots__ = ('d', 'data')

    def __init__(self, d=4):
        self.d = d
        self.data = []

    def push(self, item):
        self.data.append(item)
        self._sift_up(len(self.data) - 1)

    def pop(self):
        d = self.data
        d[0], d[-1] = d[-1], d[0]
        val = d.pop()
        if d:
            self._sift_down(0)
        return val

    def _sift_up(self, i):
        d = self.d
        data = self.data
        while i > 0:
            parent = (i - 1) // d
            if data[parent][0] > data[i][0]:
                data[parent], data[i] = data[i], data[parent]
                i = parent
            else:
                break

    def _sift_down(self, i):
        d = self.d
        data = self.data
        n = len(data)
        while True:
            first_child = d * i + 1
            if first_child >= n:
                break
            last_child = min(first_child + d, n)
            # find smallest child
            best = first_child
            for c in range(first_child + 1, last_child):
                if data[c][0] < data[best][0]:
                    best = c
            if data[best][0] < data[i][0]:
                data[i], data[best] = data[best], data[i]
                i = best
            else:
                break

    def __len__(self):
        return len(self.data)


def dijkstra_dary(graph, src, V, d=4):
    dist = [float('inf')] * V
    dist[src] = 0
    heap = DaryHeap(d)
    heap.push((0, src))
    push_count = 0
    pop_count = 0
    skip_count = 0

    while len(heap) > 0:
        dd, u = heap.pop()
        pop_count += 1
        if dd > dist[u]:
            skip_count += 1
            continue
        for w, v in graph[u]:
            nd = dd + w
            if nd < dist[v]:
                dist[v] = nd
                heap.push((nd, v))
                push_count += 1

    skip_ratio = skip_count / pop_count if pop_count > 0 else 0.0
    return dist, push_count, skip_ratio


# ──────────────────────────────────────────────
# Experiment runner
# ──────────────────────────────────────────────

VARIANTS = [
    ('lazy_binary',   lambda g, s, V: dijkstra_lazy(g, s, V)),
    ('indexed_binary', lambda g, s, V: dijkstra_indexed(g, s, V)),
    ('lazy_4ary',     lambda g, s, V: dijkstra_dary(g, s, V, d=4)),
    ('lazy_8ary',     lambda g, s, V: dijkstra_dary(g, s, V, d=8)),
]

def run_experiment(V_list, k_list, reps=5, src=0):
    records = []
    total = len(V_list) * len(k_list) * reps
    done = 0

    for V in V_list:
        for k in k_list:
            timings = {name: [] for name, _ in VARIANTS}
            pushes  = {name: [] for name, _ in VARIANTS}
            skips   = {name: [] for name, _ in VARIANTS}
            ref_dist = None   # indexed heap is reference

            for rep in range(reps):
                g = make_graph(V, k, seed=rep * 10000 + V + k)

                for name, fn in VARIANTS:
                    t0 = time.perf_counter()
                    dist, push_cnt, skip_ratio = fn(g, src, V)
                    t1 = time.perf_counter()
                    timings[name].append(t1 - t0)
                    pushes[name].append(push_cnt / V)
                    skips[name].append(skip_ratio)

                    if name == 'indexed_binary' and rep == 0:
                        ref_dist = dist

                # correctness check on first rep
                if rep == 0:
                    lazy_dist, _, _ = dijkstra_lazy(g, src, V)
                    assert lazy_dist == ref_dist, f"Correctness mismatch V={V} k={k}"

                done += 1
                if done % 20 == 0:
                    print(f"  [{done}/{total}] V={V} k={k} rep={rep}")

            # compute means
            for name, _ in VARIANTS:
                mean_t   = sum(timings[name]) / reps
                mean_push = sum(pushes[name]) / reps
                mean_skip = sum(skips[name]) / reps

                indexed_t = sum(timings['indexed_binary']) / reps
                speedup_vs_indexed = indexed_t / mean_t if mean_t > 0 else float('nan')

                records.append({
                    'V': V,
                    'k': k,
                    'variant': name,
                    'push_per_V': round(mean_push, 4),
                    'skip_ratio': round(mean_skip, 4),
                    'time_mean_s': round(mean_t, 6),
                    'speedup_vs_indexed': round(speedup_vs_indexed, 4),
                    'reps': reps,
                })

    return records


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

if __name__ == '__main__':
    V_LIST = [500, 1000, 3000, 5000, 10000]
    K_LIST = [4, 8, 16, 32, 64, 128]
    REPS   = 5

    print(f"Heap comparison experiment: V={V_LIST}, k={K_LIST}, reps={REPS}")
    print("Running...\n")

    records = run_experiment(V_LIST, K_LIST, reps=REPS)

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = "/Users/gwpark/Downloads/Algorithm Research/resultsHeapCompare"
    import os
    os.makedirs(out_dir, exist_ok=True)

    # ── CSV output ──
    csv_path = f"{out_dir}/heap_compare_{ts}.csv"
    fieldnames = ['V', 'k', 'variant', 'push_per_V', 'skip_ratio',
                  'time_mean_s', 'speedup_vs_indexed', 'reps']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"\nCSV saved: {csv_path}")

    # ── Summary table ──
    print("\n" + "="*80)
    print(f"{'V':>6}  {'k':>4}  {'variant':<18}  {'push/V':>7}  {'skip%':>6}  {'time(s)':>9}  {'speed/idx':>9}")
    print("-"*80)
    for r in records:
        print(f"{r['V']:>6}  {r['k']:>4}  {r['variant']:<18}  "
              f"{r['push_per_V']:>7.3f}  {100*r['skip_ratio']:>5.1f}%  "
              f"{r['time_mean_s']:>9.4f}  {r['speedup_vs_indexed']:>9.3f}x")

    # ── Skip-ratio comparison table (V=3000) ──
    print("\n" + "="*80)
    print("Skip-ratio by k (V=3000, all variants):")
    print(f"{'k':>4}  {'lazy_bin':>9}  {'indexed':>9}  {'lazy_4ary':>10}  {'lazy_8ary':>10}")
    print("-"*50)
    for k in K_LIST:
        row = {r['variant']: r for r in records if r['V'] == 3000 and r['k'] == k}
        if not row:
            continue
        print(f"{k:>4}  "
              f"{row.get('lazy_binary',{}).get('skip_ratio', float('nan')):>9.4f}  "
              f"{row.get('indexed_binary',{}).get('skip_ratio', float('nan')):>9.4f}  "
              f"{row.get('lazy_4ary',{}).get('skip_ratio', float('nan')):>10.4f}  "
              f"{row.get('lazy_8ary',{}).get('skip_ratio', float('nan')):>10.4f}")

    print(f"\nDone. Results in {out_dir}/")
