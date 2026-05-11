/*
 * dijkstra_heap_compare.cpp
 * =========================
 * C++ benchmark: Dijkstra across heap variants on directed ER graphs.
 *
 * Variants:
 *   1. lazy_binary   – std::priority_queue (lazy deletion)
 *   2. indexed_binary– decrease-key indexed binary heap (zero stale pops)
 *   3. lazy_4ary     – lazy-deletion 4-ary heap
 *   4. lazy_8ary     – lazy-deletion 8-ary heap
 *
 * Graph: directed ER G(V, m = k*V), U[1,100] weights.
 * NOTE: This benchmark uses directed graphs (unlike theory_probe_v4 which
 * uses undirected). Skip-ratios are directionally consistent; absolute values
 * differ slightly from main paper results (undirected, mean degree k).
 *
 * Compile:
 *   g++ -O2 -std=c++17 -o heap_compare dijkstra_heap_compare.cpp
 * Run:
 *   ./heap_compare
 * Output: CSV to resultsHeapCompare/heap_compare_cpp_<timestamp>.csv
 */

#include <algorithm>
#include <cassert>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <queue>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// ── Graph generation ─────────────────────────────────────────────────────────
using Graph = std::vector<std::vector<std::pair<int,int>>>; // adj[u] = {(w,v)}

Graph make_graph(int V, int k, unsigned seed) {
    Graph g(V);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> vdist(0, V-1);
    std::uniform_int_distribution<int> wdist(1, 100);
    int m = V * k;
    for (int i = 0; i < m; ++i) {
        int u = vdist(rng), v = vdist(rng);
        g[u].push_back({wdist(rng), v});
    }
    // spanning path for connectivity
    for (int i = 0; i < V-1; ++i)
        g[i].push_back({wdist(rng), i+1});
    return g;
}

// ── 1. Lazy binary heap (std::priority_queue) ─────────────────────────────────
struct LazyResult { long long push_count; long long pop_count; long long skip_count; };

LazyResult dijkstra_lazy(const Graph& g, int src, int V) {
    std::vector<long long> dist(V, LLONG_MAX);
    std::priority_queue<std::pair<long long,int>,
                        std::vector<std::pair<long long,int>>,
                        std::greater<>> pq;
    dist[src] = 0;
    pq.push({0, src});
    long long push_cnt = 1, pop_cnt = 0, skip_cnt = 0;
    while (!pq.empty()) {
        auto [d, u] = pq.top(); pq.pop();
        ++pop_cnt;
        if (d > dist[u]) { ++skip_cnt; continue; }
        for (auto [w, v] : g[u]) {
            long long nd = d + w;
            if (nd < dist[v]) {
                dist[v] = nd;
                pq.push({nd, v});
                ++push_cnt;
            }
        }
    }
    return {push_cnt, pop_cnt, skip_cnt};
}

// ── 2. Indexed binary heap (decrease-key) ─────────────────────────────────────
class IndexedHeap {
    int n, sz;
    std::vector<long long> key;
    std::vector<int> heap, pos; // heap[i]=node, pos[node]=heap-position
    void swap_nodes(int i, int j) {
        std::swap(heap[i], heap[j]);
        pos[heap[i]] = i;
        pos[heap[j]] = j;
    }
    void sift_up(int i) {
        while (i > 0) {
            int p = (i-1)/2;
            if (key[heap[p]] > key[heap[i]]) { swap_nodes(p, i); i = p; }
            else break;
        }
    }
    void sift_down(int i) {
        while (true) {
            int l = 2*i+1, r = 2*i+2, s = i;
            if (l < sz && key[heap[l]] < key[heap[s]]) s = l;
            if (r < sz && key[heap[r]] < key[heap[s]]) s = r;
            if (s == i) break;
            swap_nodes(i, s); i = s;
        }
    }
public:
    IndexedHeap(int n) : n(n), sz(n), key(n, LLONG_MAX), heap(n), pos(n) {
        for (int i = 0; i < n; ++i) heap[i] = pos[i] = i;
    }
    void decrease_key(int node, long long new_key) {
        if (new_key < key[node]) {
            key[node] = new_key;
            sift_up(pos[node]);
        }
    }
    std::pair<long long,int> pop_min() {
        auto res = std::make_pair(key[heap[0]], heap[0]);
        swap_nodes(0, --sz);
        if (sz) sift_down(0);
        return res;
    }
    bool empty() const { return sz == 0; }
};

struct IndexedResult { long long push_count; long long pop_count; std::vector<long long> dist; };

IndexedResult dijkstra_indexed(const Graph& g, int src, int V) {
    IndexedHeap ih(V);
    std::vector<long long> dist(V, LLONG_MAX);
    dist[src] = 0;
    ih.decrease_key(src, 0);
    long long pop_cnt = 0, dec_cnt = 0;
    while (!ih.empty()) {
        auto [d, u] = ih.pop_min();
        ++pop_cnt;
        if (d == LLONG_MAX) break;
        for (auto [w, v] : g[u]) {
            long long nd = d + w;
            if (nd < dist[v]) {
                dist[v] = nd;
                ih.decrease_key(v, nd);
                ++dec_cnt;
            }
        }
    }
    return {dec_cnt, pop_cnt, dist};
}

// ── 3 & 4. Lazy d-ary heap ────────────────────────────────────────────────────
template<int D>
class DaryHeap {
    std::vector<std::pair<long long,int>> data;
    void sift_up(int i) {
        while (i > 0) {
            int p = (i-1)/D;
            if (data[p].first > data[i].first) { std::swap(data[p], data[i]); i = p; }
            else break;
        }
    }
    void sift_down(int i) {
        int n = (int)data.size();
        while (true) {
            int fc = D*i+1, best = i;
            int lc = std::min(fc+D, n);
            for (int c = fc; c < lc; ++c)
                if (data[c].first < data[best].first) best = c;
            if (best == i) break;
            std::swap(data[i], data[best]); i = best;
        }
    }
public:
    void push(long long d, int u) { data.push_back({d,u}); sift_up((int)data.size()-1); }
    std::pair<long long,int> pop() {
        std::swap(data[0], data.back());
        auto v = data.back(); data.pop_back();
        if (!data.empty()) sift_down(0);
        return v;
    }
    bool empty() const { return data.empty(); }
};

template<int D>
LazyResult dijkstra_dary(const Graph& g, int src, int V) {
    std::vector<long long> dist(V, LLONG_MAX);
    DaryHeap<D> heap;
    dist[src] = 0;
    heap.push(0, src);
    long long push_cnt = 1, pop_cnt = 0, skip_cnt = 0;
    while (!heap.empty()) {
        auto [d, u] = heap.pop();
        ++pop_cnt;
        if (d > dist[u]) { ++skip_cnt; continue; }
        for (auto [w, v] : g[u]) {
            long long nd = d + w;
            if (nd < dist[v]) {
                dist[v] = nd;
                heap.push(nd, v);
                ++push_cnt;
            }
        }
    }
    return {push_cnt, pop_cnt, skip_cnt};
}

// ── Experiment runner ─────────────────────────────────────────────────────────
struct Record {
    int V, k;
    std::string variant;
    double push_per_V, skip_ratio, time_mean_s, speedup_vs_indexed;
    int reps;
};

std::string ts_now() {
    auto t = std::time(nullptr);
    char buf[32]; std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", std::localtime(&t));
    return buf;
}

int main() {
    const std::vector<int> V_LIST = {500, 1000, 3000, 5000, 10000};
    const std::vector<int> K_LIST = {4, 8, 16, 32, 64, 128};
    const int REPS = 5, SRC = 0;

    std::cout << "Heap comparison (C++): V=";
    for (int v : V_LIST) std::cout << v << " ";
    std::cout << "\nk=";
    for (int k : K_LIST) std::cout << k << " ";
    std::cout << "\nreps=" << REPS << "\n\n";

    std::vector<Record> records;

    for (int V : V_LIST) {
        for (int k : K_LIST) {
            struct Stats { double time_s = 0; double push_v = 0; double skip = 0; };
            Stats lazy_b, indexed, lazy4, lazy8;

            for (int rep = 0; rep < REPS; ++rep) {
                auto g = make_graph(V, k, (unsigned)(rep * 99991 + V + k));

                // lazy binary
                {
                    auto t0 = std::chrono::steady_clock::now();
                    auto r = dijkstra_lazy(g, SRC, V);
                    double dt = std::chrono::duration<double>(
                        std::chrono::steady_clock::now()-t0).count();
                    lazy_b.time_s += dt;
                    lazy_b.push_v += (double)r.push_count / V;
                    lazy_b.skip   += r.pop_count > 0
                        ? (double)r.skip_count / r.pop_count : 0.0;
                }

                // indexed binary
                std::vector<long long> ref_dist;
                {
                    auto t0 = std::chrono::steady_clock::now();
                    auto r = dijkstra_indexed(g, SRC, V);
                    double dt = std::chrono::duration<double>(
                        std::chrono::steady_clock::now()-t0).count();
                    indexed.time_s += dt;
                    indexed.push_v += (double)r.push_count / V;
                    indexed.skip   += 0.0;
                    if (rep == 0) ref_dist = r.dist;
                }

                // lazy 4-ary
                {
                    auto t0 = std::chrono::steady_clock::now();
                    auto r = dijkstra_dary<4>(g, SRC, V);
                    double dt = std::chrono::duration<double>(
                        std::chrono::steady_clock::now()-t0).count();
                    lazy4.time_s += dt;
                    lazy4.push_v += (double)r.push_count / V;
                    lazy4.skip   += r.pop_count > 0
                        ? (double)r.skip_count / r.pop_count : 0.0;
                }

                // lazy 8-ary
                {
                    auto t0 = std::chrono::steady_clock::now();
                    auto r = dijkstra_dary<8>(g, SRC, V);
                    double dt = std::chrono::duration<double>(
                        std::chrono::steady_clock::now()-t0).count();
                    lazy8.time_s += dt;
                    lazy8.push_v += (double)r.push_count / V;
                    lazy8.skip   += r.pop_count > 0
                        ? (double)r.skip_count / r.pop_count : 0.0;
                }
            }

            // average
            auto avg = [&](Stats& s) -> Stats {
                return {s.time_s/REPS, s.push_v/REPS, s.skip/REPS};
            };
            auto lb = avg(lazy_b);
            auto ix = avg(indexed);
            auto l4 = avg(lazy4);
            auto l8 = avg(lazy8);
            double idx_t = ix.time_s;

            auto make_rec = [&](const std::string& name, const Stats& s) {
                Record r;
                r.V = V; r.k = k; r.variant = name;
                r.push_per_V = s.push_v;
                r.skip_ratio = s.skip;
                r.time_mean_s = s.time_s;
                r.speedup_vs_indexed = idx_t > 0 ? idx_t / s.time_s : 0.0;
                r.reps = REPS;
                return r;
            };

            records.push_back(make_rec("lazy_binary",    lb));
            records.push_back(make_rec("indexed_binary", ix));
            records.push_back(make_rec("lazy_4ary",      l4));
            records.push_back(make_rec("lazy_8ary",      l8));

            std::cout << "V=" << std::setw(6) << V
                      << " k=" << std::setw(3) << k
                      << "  lazy=" << std::fixed << std::setprecision(4) << lb.time_s
                      << "s  idx=" << ix.time_s
                      << "s  spd=" << (idx_t>0?idx_t/lb.time_s:0.0) << "x\n";
        }
    }

    // ── Print summary table ───────────────────────────────────────────────────
    std::cout << "\n" << std::string(80,'=') << "\n";
    std::cout << std::left << std::setw(7) << "V"
              << std::setw(5) << "k"
              << std::setw(16) << "variant"
              << std::right
              << std::setw(8) << "push/V"
              << std::setw(7) << "skip%"
              << std::setw(11) << "time(s)"
              << std::setw(11) << "spd/idx"
              << "\n" << std::string(80,'-') << "\n";
    for (const auto& r : records) {
        std::cout << std::left  << std::setw(7) << r.V
                  << std::setw(5) << r.k
                  << std::setw(16) << r.variant
                  << std::right << std::fixed << std::setprecision(3)
                  << std::setw(8) << r.push_per_V
                  << std::setw(6) << std::setprecision(1) << 100.0*r.skip_ratio << "%"
                  << std::setw(11) << std::setprecision(5) << r.time_mean_s
                  << std::setw(10) << std::setprecision(3) << r.speedup_vs_indexed << "x\n";
    }

    // ── Skip-ratio summary (V=3000) ───────────────────────────────────────────
    std::cout << "\n" << std::string(60,'=') << "\n";
    std::cout << "Skip-ratio by k (V=3000, C++):\n";
    std::cout << std::setw(4) << "k"
              << std::setw(12) << "lazy_bin"
              << std::setw(12) << "indexed"
              << std::setw(12) << "lazy_4ary"
              << std::setw(12) << "lazy_8ary" << "\n"
              << std::string(52,'-') << "\n";
    for (int k : K_LIST) {
        std::cout << std::setw(4) << k;
        for (const std::string& v : {"lazy_binary","indexed_binary","lazy_4ary","lazy_8ary"}) {
            double sr = 0;
            for (const auto& r : records)
                if (r.V==3000 && r.k==k && r.variant==v) { sr=r.skip_ratio; break; }
            std::cout << std::setw(12) << std::fixed << std::setprecision(4) << sr;
        }
        std::cout << "\n";
    }

    // ── Save CSV ──────────────────────────────────────────────────────────────
    std::string out_dir = "resultsHeapCompare";
    fs::create_directories(out_dir);
    std::string csv_path = out_dir + "/heap_compare_cpp_" + ts_now() + ".csv";
    std::ofstream ofs(csv_path);
    ofs << "V,k,variant,push_per_V,skip_ratio,time_mean_s,speedup_vs_indexed,reps\n";
    for (const auto& r : records) {
        ofs << r.V << "," << r.k << "," << r.variant << ","
            << std::fixed << std::setprecision(4) << r.push_per_V << ","
            << r.skip_ratio << ","
            << std::setprecision(6) << r.time_mean_s << ","
            << std::setprecision(4) << r.speedup_vs_indexed << ","
            << r.reps << "\n";
    }
    std::cout << "\nCSV: " << csv_path << "\n";
    return 0;
}
