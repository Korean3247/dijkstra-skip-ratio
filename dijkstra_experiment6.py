"""
Dijkstra 6차 실험: 세 자료구조 정면 비교
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
비교 대상:
  A. heapq (lazy deletion)   — 기존 baseline
  B. Fibonacci heap          — O(1) decrease-key, 이론적 최적
  C. Dial's bucket queue     — 정수 가중치 O(V+E+W)

조건:
  k = 8 (sparse) / 32 (dense-mid) / 64 (dense-high)
  V = 1,000 / 10,000 / 100,000  (dense는 V≤10,000)
  가중치: 정수 uniform [1, 100]
  반복: 10회 평균

측정:
  - wall-clock time (ms)
  - heapq: push/pop/skip_count, skip_ratio
  - Fibonacci: insert/decrease_key/extract_min count
  - Dial's:   bucket_push/stale_skip count
  - accuracy: max |dist_algo - dist_heapq| (정답 비교)
"""

import heapq
import math
import random
import time
from datetime import datetime
from pathlib import Path
from collections import deque

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────

RESULTS_DIR = Path("results6")
RESULTS_DIR.mkdir(exist_ok=True)

REPEATS   = 10
SEED      = 42
W_MAX     = 100          # 정수 가중치 상한

# (V, k, label) 실험 조합
CONFIGS = [
    # sparse
    (1_000,    8,  "sparse"),
    (10_000,   8,  "sparse"),
    (100_000,  8,  "sparse"),
    # dense-mid
    (1_000,   32,  "dense-mid"),
    (10_000,  32,  "dense-mid"),
    # dense-high
    (1_000,   64,  "dense-high"),
    (10_000,  64,  "dense-high"),
]

K_COLORS = {8: "#2196F3", 32: "#FF9800", 64: "#F44336"}
DS_COLORS = {"heapq": "#2196F3", "fibonacci": "#4CAF50", "dial": "#FF9800"}


# ─────────────────────────────────────────────────────────────────────
# Graph generator — 정수 가중치
# ─────────────────────────────────────────────────────────────────────

def make_graph_int_weight(V: int, k: int, seed: int = SEED) -> list:
    """adj[u] = [(v, w), ...]; w는 정수 [1, W_MAX]."""
    rng = random.Random(seed)
    adj = [[] for _ in range(V)]

    target_edges = V * k // 2
    for _ in range(target_edges):
        u = rng.randint(0, V - 1)
        v = rng.randint(0, V - 1)
        if u != v:
            w = rng.randint(1, W_MAX)
            adj[u].append((v, w))
            adj[v].append((u, w))

    # spanning path for connectivity
    nodes = list(range(V))
    rng.shuffle(nodes)
    for i in range(len(nodes) - 1):
        u, v = nodes[i], nodes[i + 1]
        w = rng.randint(1, W_MAX)
        adj[u].append((v, w))
        adj[v].append((u, w))

    return adj


# ─────────────────────────────────────────────────────────────────────
# A. heapq Dijkstra (lazy deletion, instrumented)
# ─────────────────────────────────────────────────────────────────────

def dijkstra_heapq(adj, source, V):
    dist = [float("inf")] * V
    dist[source] = 0
    heap = [(0, source)]
    push_count = 1
    pop_count  = 0
    skip_count = 0

    while heap:
        d, u = heapq.heappop(heap)
        pop_count += 1
        if d > dist[u]:
            skip_count += 1
            continue
        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
                push_count += 1

    return dist, {
        "push_count":  push_count,
        "pop_count":   pop_count,
        "skip_count":  skip_count,
        "skip_ratio":  skip_count / pop_count if pop_count else 0.0,
    }


# ─────────────────────────────────────────────────────────────────────
# B. Fibonacci Heap
# ─────────────────────────────────────────────────────────────────────

class FibNode:
    __slots__ = ("key", "val", "degree", "mark", "parent", "child",
                 "left", "right")

    def __init__(self, key, val):
        self.key    = key
        self.val    = val
        self.degree = 0
        self.mark   = False
        self.parent = None
        self.child  = None
        self.left   = self
        self.right  = self


class FibHeap:
    def __init__(self):
        self.min_node = None
        self.n        = 0
        self.insert_count      = 0
        self.extract_min_count = 0
        self.decrease_key_count = 0

    # ── 내부 유틸 ──────────────────────────────────────────────────

    @staticmethod
    def _link_to_root_list(heap_min, node):
        """node를 root list에 삽입 (heap.min 왼쪽)."""
        node.left        = heap_min.left
        node.right       = heap_min
        heap_min.left.right = node
        heap_min.left    = node

    @staticmethod
    def _remove_from_list(node):
        node.left.right = node.right
        node.right.left = node.left

    @staticmethod
    def _add_child(parent, child):
        child.parent = parent
        if parent.child is None:
            parent.child      = child
            child.left = child.right = child
        else:
            FibHeap._link_to_root_list(parent.child, child)
        parent.degree += 1
        child.mark = False

    # ── 공개 인터페이스 ────────────────────────────────────────────

    def insert(self, key, val):
        node = FibNode(key, val)
        if self.min_node is None:
            self.min_node = node
        else:
            self._link_to_root_list(self.min_node, node)
            if key < self.min_node.key:
                self.min_node = node
        self.n += 1
        self.insert_count += 1
        return node

    def extract_min(self):
        z = self.min_node
        if z is None:
            return None
        # z의 자식들을 root list로
        if z.child is not None:
            children = []
            c = z.child
            while True:
                children.append(c)
                c = c.right
                if c is z.child:
                    break
            for c in children:
                self._link_to_root_list(z, c)
                c.parent = None

        # z를 root list에서 제거
        self._remove_from_list(z)
        if z == z.right:
            self.min_node = None
        else:
            self.min_node = z.right
            self._consolidate()

        self.n -= 1
        self.extract_min_count += 1
        return z

    def _consolidate(self):
        max_degree = int(math.log2(self.n + 1)) + 2
        A = [None] * (max_degree + 1)

        # root list를 리스트로 수집
        roots = []
        x = self.min_node
        while True:
            roots.append(x)
            x = x.right
            if x is self.min_node:
                break

        for w in roots:
            x = w
            d = x.degree
            while d < len(A) and A[d] is not None:
                y = A[d]
                if x.key > y.key:
                    x, y = y, x
                # y를 x의 자식으로 연결
                self._remove_from_list(y)
                self._add_child(x, y)
                A[d] = None
                d += 1
            if d >= len(A):
                A.extend([None] * (d - len(A) + 1))
            A[d] = x

        # min 재설정
        self.min_node = None
        for node in A:
            if node is not None:
                node.left = node.right = node
                if self.min_node is None:
                    self.min_node = node
                else:
                    self._link_to_root_list(self.min_node, node)
                    if node.key < self.min_node.key:
                        self.min_node = node

    def decrease_key(self, node, new_key):
        if new_key > node.key:
            raise ValueError("new_key > current key")
        node.key = new_key
        p = node.parent
        if p is not None and node.key < p.key:
            self._cut(node, p)
            self._cascading_cut(p)
        if node.key < self.min_node.key:
            self.min_node = node
        self.decrease_key_count += 1

    def _cut(self, node, parent):
        self._remove_from_list(node)
        if node == node.right:
            parent.child = None
        elif parent.child is node:
            parent.child = node.right
        parent.degree -= 1
        node.left = node.right = node
        node.parent = None
        node.mark   = False
        self._link_to_root_list(self.min_node, node)

    def _cascading_cut(self, node):
        p = node.parent
        if p is not None:
            if not node.mark:
                node.mark = True
            else:
                self._cut(node, p)
                self._cascading_cut(p)

    def is_empty(self):
        return self.min_node is None


def dijkstra_fibonacci(adj, source, V):
    INF = float("inf")
    dist  = [INF] * V
    nodes = [None]  * V   # FibNode 참조

    fh = FibHeap()
    dist[source]  = 0
    nodes[source] = fh.insert(0, source)

    while not fh.is_empty():
        z = fh.extract_min()
        u = z.val
        d = z.key
        if d > dist[u]:
            continue
        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                if nodes[v] is None:
                    nodes[v] = fh.insert(nd, v)
                else:
                    fh.decrease_key(nodes[v], nd)

    return dist, {
        "insert_count":       fh.insert_count,
        "extract_min_count":  fh.extract_min_count,
        "decrease_key_count": fh.decrease_key_count,
    }


# ─────────────────────────────────────────────────────────────────────
# C. Dial's Bucket Queue (circular, integer weights)
# ─────────────────────────────────────────────────────────────────────

def dijkstra_dial(adj, source, V, W=W_MAX):
    """
    Dial's algorithm (bucket queue).
    buckets: W+1개 circular 배열, 각 bucket은 deque.
    현재 버킷 포인터 current_dist을 단조 증가시키며 pop.
    stale 항목은 dist[u] != d 로 판단 후 skip.
    """
    INF = float("inf")
    dist = [INF] * V
    dist[source] = 0

    num_buckets = W + 1
    buckets = [deque() for _ in range(num_buckets)]
    buckets[0].append(source)

    bucket_push_count = 1
    stale_skip_count  = 0
    settled_count     = 0

    current_dist = 0
    max_dist = V * W  # 최대 가능한 최단 경로

    while settled_count < V and current_dist <= max_dist:
        b_idx = current_dist % num_buckets
        while buckets[b_idx]:
            u = buckets[b_idx].popleft()
            if dist[u] != current_dist:
                stale_skip_count += 1
                continue
            settled_count += 1
            for v, w in adj[u]:
                nd = current_dist + w
                if nd < dist[v]:
                    dist[v] = nd
                    buckets[nd % num_buckets].append(v)
                    bucket_push_count += 1
        current_dist += 1

    return dist, {
        "bucket_push_count": bucket_push_count,
        "stale_skip_count":  stale_skip_count,
        "settled_count":     settled_count,
    }


# ─────────────────────────────────────────────────────────────────────
# Experiment runner
# ─────────────────────────────────────────────────────────────────────

def run_one(adj, source, V, algo):
    if algo == "heapq":
        t0 = time.perf_counter()
        dist, ops = dijkstra_heapq(adj, source, V)
        return time.perf_counter() - t0, dist, ops
    elif algo == "fibonacci":
        t0 = time.perf_counter()
        dist, ops = dijkstra_fibonacci(adj, source, V)
        return time.perf_counter() - t0, dist, ops
    elif algo == "dial":
        t0 = time.perf_counter()
        dist, ops = dijkstra_dial(adj, source, V)
        return time.perf_counter() - t0, dist, ops
    else:
        raise ValueError(algo)


def run_experiment() -> pd.DataFrame:
    rows = []

    # Fibonacci heap은 V=100k에서 Python overhead로 제한
    FIB_V_LIMIT = 20_000

    total = len(CONFIGS)
    print(f"총 조합: {total}개  (각 10회 반복)\n")

    with tqdm(total=total, desc="진행") as pbar:
        for (V, k, label) in CONFIGS:
            adj = make_graph_int_weight(V, k)
            E   = sum(len(nbrs) for nbrs in adj) // 2

            # 정답 (heapq 첫 실행)
            _, ref_dist, _ = run_one(adj, 0, V, "heapq")

            algos = ["heapq", "dial"]
            if V <= FIB_V_LIMIT:
                algos.append("fibonacci")

            for algo in algos:
                times = []
                last_ops = None

                for rep in range(REPEATS):
                    elapsed, dist, ops = run_one(adj, 0, V, algo)
                    times.append(elapsed)
                    if rep == 0:
                        last_ops = ops
                        # 정확도 검증 (첫 실행에서만)
                        max_err = max(
                            abs(d - r) if r != float("inf") else 0.0
                            for d, r in zip(dist, ref_dist)
                        )

                row = {
                    "V":       V,
                    "k":       k,
                    "label":   label,
                    "algo":    algo,
                    "E":       E,
                    "time_mean_ms": round(float(np.mean(times)) * 1000, 3),
                    "time_std_ms":  round(float(np.std(times))  * 1000, 3),
                    "max_err":      round(max_err, 6),
                    "correct":      max_err < 1e-6,
                }
                row.update({f"op_{k2}": v for k2, v in last_ops.items()})
                rows.append(row)

            pbar.update(1)
            pbar.write(
                f"  V={V:>7,}  k={k:>2}  ({label})"
                f"  algos={algos}"
            )

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────

def plot_results(df: pd.DataFrame, ts: str) -> None:

    # ── Fig 1: time comparison — sparse (k=8, V별) ───────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    k_list  = [8, 32, 64]
    titles  = ["sparse (k=8)", "dense-mid (k=32)", "dense-high (k=64)"]

    for ax, k_val, title in zip(axes, k_list, titles):
        sub = df[df["k"] == k_val]
        V_vals = sorted(sub["V"].unique())
        algos  = sorted(sub["algo"].unique())
        x = np.arange(len(V_vals))
        width = 0.25
        offsets = np.linspace(-(len(algos)-1)/2, (len(algos)-1)/2, len(algos)) * width

        for off, algo in zip(offsets, algos):
            t_vals = []
            for V in V_vals:
                row = sub[(sub["V"] == V) & (sub["algo"] == algo)]
                t_vals.append(row["time_mean_ms"].values[0] if len(row) else 0)
            ax.bar(x + off, t_vals, width=width * 0.9,
                   label=algo, color=DS_COLORS.get(algo, "#888"))

        ax.set_xticks(x)
        ax.set_xticklabels([f"{v:,}" for v in V_vals])
        ax.set_xlabel("V")
        ax.set_ylabel("Time (ms)")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)

    plt.suptitle("Wall-clock Time Comparison by Data Structure", fontsize=13)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"time_comparison_{ts}.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Fig 2: speedup ratio (vs heapq) ──────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, k_val in zip(axes, [8, 32]):
        sub_k = df[df["k"] == k_val]
        sub_hq = sub_k[sub_k["algo"] == "heapq"].set_index("V")

        for algo in ["dial", "fibonacci"]:
            sub_a = sub_k[sub_k["algo"] == algo].sort_values("V")
            if sub_a.empty:
                continue
            V_vals = sub_a["V"].values
            speedup = [
                sub_hq.loc[V, "time_mean_ms"] / sub_a[sub_a["V"] == V]["time_mean_ms"].values[0]
                if V in sub_hq.index else np.nan
                for V in V_vals
            ]
            ax.plot(V_vals, speedup, marker="o", lw=2,
                    color=DS_COLORS.get(algo, "#888"), label=algo)

        ax.axhline(1.0, color="gray", lw=1, ls="--", label="heapq baseline")
        ax.set_xscale("log")
        ax.set_title(f"Speedup vs heapq  (k={k_val})")
        ax.set_xlabel("V")
        ax.set_ylabel("Speedup  (heapq_time / algo_time)")
        ax.legend(fontsize=9)
        ax.grid(True, which="both", alpha=0.3)

    plt.suptitle("Speedup Ratio over heapq Baseline", fontsize=13)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"speedup_ratio_{ts}.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Fig 3: operation count comparison (V=10,000) ──────────────────
    sub10k = df[df["V"] == 10_000].copy()
    k_vals = sorted(sub10k["k"].unique())

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(k_vals))
    width = 0.25
    algos = sorted(sub10k["algo"].unique())
    offsets = np.linspace(-(len(algos)-1)/2, (len(algos)-1)/2, len(algos)) * width

    def total_ops(row):
        """각 알고리즘의 총 연산 수 (비교 가능한 단일 수치)."""
        if row["algo"] == "heapq":
            return row.get("op_push_count", 0) + row.get("op_pop_count", 0)
        elif row["algo"] == "fibonacci":
            return (row.get("op_insert_count", 0)
                    + row.get("op_extract_min_count", 0)
                    + row.get("op_decrease_key_count", 0))
        elif row["algo"] == "dial":
            return (row.get("op_bucket_push_count", 0)
                    + row.get("op_settled_count", 0))
        return 0

    sub10k["total_ops"] = sub10k.apply(total_ops, axis=1)

    for off, algo in zip(offsets, algos):
        op_vals = []
        for k_v in k_vals:
            r = sub10k[(sub10k["k"] == k_v) & (sub10k["algo"] == algo)]
            op_vals.append(r["total_ops"].values[0] if len(r) else 0)
        ax.bar(x + off, op_vals, width=width * 0.9,
               label=algo, color=DS_COLORS.get(algo, "#888"))

    ax.set_xticks(x)
    ax.set_xticklabels([f"k={k}" for k in k_vals])
    ax.set_ylabel("Total heap operations")
    ax.set_title("Total Heap Operations  (V=10,000)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"op_count_{ts}.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Fig 4: skip_ratio (heapq만, k별) ─────────────────────────────
    sub_hq = df[df["algo"] == "heapq"].copy()
    if "op_skip_ratio" in sub_hq.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        for k_val in sorted(sub_hq["k"].unique()):
            sub = sub_hq[sub_hq["k"] == k_val].sort_values("V")
            ax.plot(sub["V"], sub["op_skip_ratio"],
                    marker="o", lw=2,
                    color=K_COLORS.get(k_val, "#888"),
                    label=f"k={k_val}")
        ax.set_xscale("log")
        ax.set_title("heapq skip_ratio by k and V")
        ax.set_xlabel("V")
        ax.set_ylabel("skip_ratio")
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"skip_ratio_heapq_{ts}.png",
                    dpi=150, bbox_inches="tight")
        plt.close()

    print(f"Plots saved → {RESULTS_DIR}/")


# ─────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame) -> None:
    print("\n══════════════════════════════════════════════════════")
    print("  Wall-clock Time Summary (ms)  ±  std")
    print("══════════════════════════════════════════════════════")
    print(f"{'V':>8}  {'k':>4}  {'algo':>10}  {'mean_ms':>9}  {'std_ms':>8}  correct")
    print("─" * 60)
    for _, r in df.sort_values(["k", "V", "algo"]).iterrows():
        print(
            f"{r['V']:>8,}  {r['k']:>4}  {r['algo']:>10}"
            f"  {r['time_mean_ms']:>9.2f}  {r['time_std_ms']:>8.3f}"
            f"  {'✓' if r['correct'] else '✗ err=' + str(r['max_err'])}"
        )

    print("\n── Accuracy check ──────────────────────────────────")
    incorrect = df[~df["correct"]]
    if incorrect.empty:
        print("All algorithms produced correct results (max_err < 1e-6).")
    else:
        print(incorrect[["V", "k", "algo", "max_err"]])


# ─────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("Dijkstra 6차 실험 — 세 자료구조 비교")
    print(f"Started: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
    print(f"조합: {CONFIGS}\n")

    df = run_experiment()

    csv_path = RESULTS_DIR / f"experiment6_{ts}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nData saved → {csv_path}")

    print_summary(df)
    plot_results(df, ts)

    print(f"\nDone. All outputs → {RESULTS_DIR.resolve()}/")
