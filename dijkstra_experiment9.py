"""
Dijkstra 9차 실험: 실수 가중치 heapq vs Fibonacci heap
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
언어: Python + C++ (동일 조건 동시 비교)
가중치: float uniform [0.0, 1.0]
비교: heapq (lazy deletion) vs Fibonacci heap
Dial's 제외 (정수 가중치 전용)

C++ 코드: dijkstra_experiment9.cpp 컴파일·실행
"""

import heapq
import math
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

RESULTS_DIR = Path("results9")
RESULTS_DIR.mkdir(exist_ok=True)

REPEATS = 10
SEED    = 42

CONFIGS = [
    (1_000,    8, "sparse"),
    (10_000,   8, "sparse"),
    (100_000,  8, "sparse"),
    (1_000,   32, "dense-mid"),
    (10_000,  32, "dense-mid"),
    (100_000, 32, "dense-mid"),
    (1_000,   64, "dense-high"),
    (10_000,  64, "dense-high"),
    (100_000, 64, "dense-high"),
]

FIB_LIMIT_PY = 20_000  # Python Fibonacci V 제한

DS_COLORS  = {"heapq": "#2196F3", "fibonacci": "#4CAF50"}
LANG_COLORS = {"Python": "#E91E63", "C++": "#009688"}
K_LIST = [8, 32, 64]


# ─────────────────────────────────────────────────────────────────────
# Graph (float weights)
# ─────────────────────────────────────────────────────────────────────

def make_graph_float(V, k, seed=SEED):
    rng = random.Random(seed)
    adj = [[] for _ in range(V)]
    for _ in range(V * k // 2):
        u = rng.randint(0, V-1); v = rng.randint(0, V-1)
        if u != v:
            w = rng.uniform(0.0, 1.0)
            adj[u].append((v,w)); adj[v].append((u,w))
    nodes = list(range(V)); rng.shuffle(nodes)
    for i in range(len(nodes)-1):
        u,v = nodes[i],nodes[i+1]; w = rng.uniform(0.0, 1.0)
        adj[u].append((v,w)); adj[v].append((u,w))
    return adj


# ─────────────────────────────────────────────────────────────────────
# A. heapq (Python)
# ─────────────────────────────────────────────────────────────────────

def dijkstra_heapq(adj, source, V):
    dist = [float("inf")] * V
    dist[source] = 0.0
    heap = [(0.0, source)]
    push_c = pop_c = skip_c = 0
    push_c = 1
    while heap:
        d, u = heapq.heappop(heap); pop_c += 1
        if d > dist[u]: skip_c += 1; continue
        for v,w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd; heapq.heappush(heap, (nd, v)); push_c += 1
    return dist, {"push_count": push_c, "pop_count": pop_c, "skip_count": skip_c,
                  "skip_ratio": skip_c/pop_c if pop_c else 0.0}


# ─────────────────────────────────────────────────────────────────────
# B. Fibonacci Heap (Python, float key)
# ─────────────────────────────────────────────────────────────────────

class FibNode:
    __slots__ = ("key","val","degree","mark","parent","child","left","right")
    def __init__(self, key, val):
        self.key=key; self.val=val; self.degree=0; self.mark=False
        self.parent=self.child=None; self.left=self.right=self

class FibHeap:
    def __init__(self):
        self.min_node=None; self.n=0
        self.ins=self.ext=self.dec_=0

    def _add_root(self, x):
        x.parent=None
        if not self.min_node:
            self.min_node=x; x.left=x.right=x; return
        x.right=self.min_node; x.left=self.min_node.left
        self.min_node.left.right=x; self.min_node.left=x
        if x.key < self.min_node.key: self.min_node=x

    def insert(self, key, val):
        nd=FibNode(key,val)
        self._add_root(nd); self.n+=1; self.ins+=1
        return nd

    def _link(self, y, x):
        y.parent=x; y.mark=False
        if not x.child: x.child=y; y.left=y.right=y
        else:
            y.right=x.child; y.left=x.child.left
            x.child.left.right=y; x.child.left=y
        x.degree+=1

    def _consolidate(self):
        md=max(2, int(math.log2(self.n+1))+3)
        A=[None]*(md+1)
        roots=[]
        c=self.min_node
        while True:
            roots.append(c); c=c.right
            if c is self.min_node: break
        for w in roots:
            x=w; x.left=x.right=x
            d=x.degree
            while d<len(A) and A[d]:
                y=A[d]
                if x.key>y.key: x,y=y,x
                self._link(y,x); A[d]=None; d+=1
            if d>=len(A): A.extend([None]*(d-len(A)+1))
            A[d]=x
        self.min_node=None
        for nd in A:
            if nd is None: continue
            nd.left=nd.right=nd; nd.parent=None
            if not self.min_node: self.min_node=nd
            else:
                nd.right=self.min_node; nd.left=self.min_node.left
                self.min_node.left.right=nd; self.min_node.left=nd
                if nd.key<self.min_node.key: self.min_node=nd

    def extract_min(self):
        z=self.min_node
        if not z: return None
        if z.child:
            ch=[]; c=z.child
            while True:
                ch.append(c); c=c.right
                if c is z.child: break
            for ci in ch: ci.left=ci.right=ci; ci.parent=None; self._add_root(ci)
            z.child=None
        z.left.right=z.right; z.right.left=z.left
        if z is z.right: self.min_node=None
        else: self.min_node=z.right; self._consolidate()
        self.n-=1; self.ext+=1
        return z

    def _cut(self, x, y):
        if x.right is x: y.child=None
        else:
            if y.child is x: y.child=x.right
            x.left.right=x.right; x.right.left=x.left
        y.degree-=1; x.left=x.right=x; x.parent=None; x.mark=False
        self._add_root(x)

    def _ccut(self, y):
        z=y.parent
        if z:
            if not y.mark: y.mark=True
            else: self._cut(y,z); self._ccut(z)

    def decrease_key(self, node, new_key):
        node.key=new_key
        if node.parent and node.key<node.parent.key:
            p=node.parent; self._cut(node,p); self._ccut(p)
        if self.min_node and node.key<self.min_node.key: self.min_node=node
        self.dec_+=1

    def is_empty(self): return self.min_node is None


def dijkstra_fibonacci_py(adj, source, V):
    dist=[float("inf")]*V; dist[source]=0.0
    nodes=[None]*V; settled=[False]*V
    fh=FibHeap()
    nodes[source]=fh.insert(0.0, source)
    while not fh.is_empty():
        z=fh.extract_min()
        if z is None: break
        u=z.val
        if settled[u]: continue
        settled[u]=True
        for v,w in adj[u]:
            if settled[v]: continue
            nd=dist[u]+w
            if nd<dist[v]:
                dist[v]=nd
                if nodes[v] is None: nodes[v]=fh.insert(nd,v)
                else: fh.decrease_key(nodes[v], nd)
    return dist, {"insert_count": fh.ins, "extract_min_count": fh.ext, "decrease_key_count": fh.dec_}


# ─────────────────────────────────────────────────────────────────────
# Python experiment runner
# ─────────────────────────────────────────────────────────────────────

def run_python_experiment():
    rows = []
    total = sum(2 if V<=FIB_LIMIT_PY else 1 for V,k,label in CONFIGS)
    print(f"Python 실험: {total}개 (V≤{FIB_LIMIT_PY:,} 에서만 Fibonacci)\n")
    with tqdm(total=len(CONFIGS), desc="Python") as pbar:
        for V, k, label in CONFIGS:
            adj = make_graph_float(V, k)
            E   = sum(len(n2) for n2 in adj) // 2
            ref_dist, _ = dijkstra_heapq(adj, 0, V)

            algos = ["heapq"] + (["fibonacci"] if V<=FIB_LIMIT_PY else [])
            for algo in algos:
                times = []
                first_ops = None
                for rep in range(REPEATS):
                    t0 = time.perf_counter()
                    if algo == "heapq":
                        dist, ops = dijkstra_heapq(adj, 0, V)
                    else:
                        dist, ops = dijkstra_fibonacci_py(adj, 0, V)
                    times.append(time.perf_counter()-t0)
                    if rep == 0:
                        first_ops = ops
                        max_err = max(abs(d-r) if r!=float("inf") else 0.0
                                      for d,r in zip(dist, ref_dist))
                row = {
                    "V": V, "k": k, "label": label, "algo": algo,
                    "lang": "Python", "E": E,
                    "time_mean_ms": round(float(np.mean(times))*1000, 3),
                    "time_std_ms":  round(float(np.std(times))*1000, 3),
                    "max_err": round(max_err, 9),
                    "correct": max_err < 1e-9,
                }
                row.update({f"op_{k2}": v for k2,v in first_ops.items()})
                rows.append(row)
            pbar.update(1)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────
# C++ compile & run
# ─────────────────────────────────────────────────────────────────────

def run_cpp_experiment():
    print("\n── C++ 컴파일 중 ──")
    ret = subprocess.run(
        ["g++","-O2","-std=c++17","-o","dijkstra9_bin",
         "dijkstra_experiment9.cpp","-lm"],
        capture_output=True, text=True)
    if ret.returncode != 0:
        print("컴파일 실패:\n", ret.stderr); return None
    print("실행 중 ...")
    ret2 = subprocess.run(["./dijkstra9_bin"], capture_output=True, text=True)
    print(ret2.stderr)
    csvs = sorted(RESULTS_DIR.glob("experiment9_cpp_*.csv"))
    if not csvs: print("CSV 없음"); return None
    df = pd.read_csv(csvs[-1])
    df["lang"] = "C++"
    return df


# ─────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────

def plot_all(df_py, df_cpp, ts):
    if df_cpp is not None:
        df_all = pd.concat([df_py, df_cpp], ignore_index=True)
    else:
        df_all = df_py.copy()

    k_list  = [8, 32, 64]
    titles  = ["sparse (k=8)", "dense-mid (k=32)", "dense-high (k=64)"]

    # ── Fig 1: Python heapq vs fibonacci (bars, by V) ────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    sub_py = df_py.copy()
    for ax, k_val, title in zip(axes, k_list, titles):
        sub = sub_py[sub_py["k"]==k_val].sort_values("V")
        V_vals = sorted(sub["V"].unique())
        algos  = sorted(sub["algo"].unique())
        x = np.arange(len(V_vals)); w=0.35
        offs = [-w/2, w/2] if len(algos)==2 else [0]
        for off, algo in zip(offs, algos):
            t_vals=[sub[(sub["V"]==V)&(sub["algo"]==algo)]["time_mean_ms"].values[0]
                    if len(sub[(sub["V"]==V)&(sub["algo"]==algo)]) else 0
                    for V in V_vals]
            ax.bar(x+off, t_vals, w*0.9, label=algo, color=DS_COLORS.get(algo,"#888"))
        ax.set_xticks(x); ax.set_xticklabels([f"{v:,}" for v in V_vals])
        ax.set_title(f"Python — {title}"); ax.set_xlabel("V"); ax.set_ylabel("ms")
        ax.legend(fontsize=8); ax.grid(True, axis="y", alpha=0.3)
    plt.suptitle("Python: heapq vs Fibonacci (float weights [0,1])", fontsize=13)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"py_comparison_{ts}.png", dpi=150, bbox_inches="tight")
    plt.close()

    if df_cpp is None:
        print("C++ 결과 없음 — C++ 비교 그래프 생략")
        return

    # ── Fig 2: C++ heapq vs fibonacci ────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, k_val, title in zip(axes, k_list, titles):
        sub = df_cpp[df_cpp["k"]==k_val].sort_values("V")
        V_vals = sorted(sub["V"].unique())
        algos  = sorted(sub["algo"].unique())
        x = np.arange(len(V_vals)); w=0.35
        offs = [-w/2, w/2] if len(algos)==2 else [0]
        for off, algo in zip(offs, algos):
            t_vals=[sub[(sub["V"]==V)&(sub["algo"]==algo)]["time_mean_ms"].values[0]
                    if len(sub[(sub["V"]==V)&(sub["algo"]==algo)]) else 0
                    for V in V_vals]
            ax.bar(x+off, t_vals, w*0.9, label=algo, color=DS_COLORS.get(algo,"#888"))
        ax.set_xticks(x); ax.set_xticklabels([f"{v:,}" for v in V_vals])
        ax.set_title(f"C++ — {title}"); ax.set_xlabel("V"); ax.set_ylabel("ms")
        ax.legend(fontsize=8); ax.grid(True, axis="y", alpha=0.3)
    plt.suptitle("C++: heapq vs Fibonacci (float weights [0,1])", fontsize=13)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"cpp_comparison_{ts}.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Fig 3: Python vs C++ — heapq (float) ─────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, k_val, title in zip(axes, k_list, titles):
        sc = df_cpp[(df_cpp["k"]==k_val)&(df_cpp["algo"]=="heapq")].sort_values("V")
        sp = df_py [(df_py ["k"]==k_val)&(df_py ["algo"]=="heapq")].sort_values("V")
        common_V=sorted(set(sc["V"])&set(sp["V"]))
        if not common_V: continue
        t_c=[sc[sc["V"]==V]["time_mean_ms"].values[0] for V in common_V]
        t_p=[sp[sp["V"]==V]["time_mean_ms"].values[0] for V in common_V]
        x=np.arange(len(common_V))
        ax.bar(x-0.2,t_c,0.35,label="C++",color=LANG_COLORS["C++"])
        ax.bar(x+0.2,t_p,0.35,label="Python",color=LANG_COLORS["Python"])
        for xi,(tc,tp) in enumerate(zip(t_c,t_p)):
            ax.text(xi+0.2,tp,f"{tp/tc:.0f}×",ha="center",va="bottom",fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels([f"{v:,}" for v in common_V])
        ax.set_title(f"heapq — {title}"); ax.set_xlabel("V"); ax.set_ylabel("ms")
        ax.legend(); ax.grid(True,axis="y",alpha=0.3)
    plt.suptitle("heapq (float): Python vs C++ (숫자=Python/C++)", fontsize=13)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"heapq_py_vs_cpp_{ts}.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Fig 4: Python vs C++ — fibonacci (float) ─────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, k_val, title in zip(axes, k_list, titles):
        sc=df_cpp[(df_cpp["k"]==k_val)&(df_cpp["algo"]=="fibonacci")].sort_values("V")
        sp=df_py [(df_py ["k"]==k_val)&(df_py ["algo"]=="fibonacci")].sort_values("V")
        common_V=sorted(set(sc["V"])&set(sp["V"]))
        if not common_V: ax.set_title(f"fibonacci — {title}\n(없음)"); continue
        t_c=[sc[sc["V"]==V]["time_mean_ms"].values[0] for V in common_V]
        t_p=[sp[sp["V"]==V]["time_mean_ms"].values[0] for V in common_V]
        x=np.arange(len(common_V))
        ax.bar(x-0.2,t_c,0.35,label="C++",color=LANG_COLORS["C++"])
        ax.bar(x+0.2,t_p,0.35,label="Python",color=LANG_COLORS["Python"])
        for xi,(tc,tp) in enumerate(zip(t_c,t_p)):
            ax.text(xi+0.2,tp,f"{tp/tc:.0f}×",ha="center",va="bottom",fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels([f"{v:,}" for v in common_V])
        ax.set_title(f"fibonacci — {title}"); ax.set_xlabel("V"); ax.set_ylabel("ms")
        ax.legend(); ax.grid(True,axis="y",alpha=0.3)
    plt.suptitle("Fibonacci (float): Python vs C++ (숫자=Python/C++)", fontsize=13)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"fib_py_vs_cpp_{ts}.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nAll plots → {RESULTS_DIR}/")


# ─────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("Dijkstra 9차 실험 — 실수 가중치 heapq vs Fibonacci")
    print(f"Started: {datetime.now():%Y-%m-%d %H:%M:%S}\n")

    df_py  = run_python_experiment()
    py_csv = RESULTS_DIR / f"experiment9_python_{ts}.csv"
    df_py.to_csv(py_csv, index=False)
    print(f"Python CSV → {py_csv}")

    df_cpp = run_cpp_experiment()

    plot_all(df_py, df_cpp, ts)
    print(f"\nDone → {RESULTS_DIR.resolve()}/")
