"""
이론 탐침: α = 0.262가 weight distribution에 독립적인가?

만약 α가 weight distribution에 무관하면 →
'α는 그래프 구조(ER random graph)에서만 결정되는 지수'라는
강한 이론적 주장 가능.

테스트: U[1,100] vs U[1,10] vs U[1,1000] vs Exp(1) vs 상수(=1)
"""
import heapq, random, math
import numpy as np

SEED = 42
V = 10000
REPS = 5
K_VALS = [4, 8, 16, 32, 64, 128]

def dijkstra_skip(adj, src, V):
    dist = [float('inf')] * V
    dist[src] = 0
    heap = [(0, src)]
    push_c, pop_c, skip_c = 1, 0, 0
    while heap:
        d, u = heapq.heappop(heap); pop_c += 1
        if d > dist[u]: skip_c += 1; continue
        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd; heapq.heappush(heap, (nd, v)); push_c += 1
    return skip_c / pop_c if pop_c else 0.0

def make_graph(V, k, weight_fn, rng):
    adj = [[] for _ in range(V)]
    for _ in range(V * k // 2):
        u, v = rng.randint(0, V-1), rng.randint(0, V-1)
        if u != v:
            w = weight_fn(rng)
            adj[u].append((v, w)); adj[v].append((u, w))
    nodes = list(range(V)); rng.shuffle(nodes)
    for i in range(V - 1):
        w = weight_fn(rng)
        adj[nodes[i]].append((nodes[i+1], w))
        adj[nodes[i+1]].append((nodes[i], w))
    return adj

# 5가지 weight distributions
DISTS = {
    'U[1,10]':   lambda rng: rng.randint(1, 10),
    'U[1,100]':  lambda rng: rng.randint(1, 100),
    'U[1,1000]': lambda rng: rng.randint(1, 1000),
    'Exp(mean=10)': lambda rng: max(1, int(-10 * math.log(rng.random()))),
    'Constant=1':   lambda rng: 1,
}

print(f"{'k':>5}", end="")
for name in DISTS:
    print(f"  {name:>14}", end="")
print()
print("-" * (5 + 16 * len(DISTS)))

results = {}
for k in K_VALS:
    row = {}
    print(f"{k:>5}", end="", flush=True)
    for name, wfn in DISTS.items():
        ratios = []
        for rep in range(REPS):
            rng = random.Random(SEED + rep)
            adj = make_graph(V, k, wfn, rng)
            sr = dijkstra_skip(adj, 0, V)
            ratios.append(sr)
        row[name] = np.mean(ratios)
        print(f"  {row[name]:>14.4f}", end="", flush=True)
    results[k] = row
    print()

# Power law fit per distribution
print("\n── α 추정 (log-log 회귀) per weight distribution ──")
k_arr = np.array(K_VALS, dtype=float)
for name in DISTS:
    sr_arr = np.array([results[k][name] for k in K_VALS])
    # skip_ratio = 1 - c/k^α  →  log(1-skip_ratio) = log(c) - α*log(k)
    y = np.log(1 - sr_arr + 1e-9)
    x = np.log(k_arr)
    mask = np.isfinite(y) & (y < 0)
    if mask.sum() > 2:
        alpha, log_c = np.polyfit(x[mask], y[mask], 1)
        c = np.exp(log_c)
        print(f"  {name:>14}: α={-alpha:.4f}  c={c:.4f}")
    else:
        print(f"  {name:>14}: 피팅 불가 (상수 가중치)")
