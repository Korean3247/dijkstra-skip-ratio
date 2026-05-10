/*
 * Dijkstra 실험 A: V 범위 확장
 * ─────────────────────────────────────────────────────────
 * 목적: V-independence of skip_ratio 확인 (V=200K ~ 1M)
 *       현재 paper Table I 상한 V=100K를 훨씬 초과하는 규모에서
 *       skip_ratio가 여전히 k에만 의존함을 검증
 *
 * 조건: k = {8, 32, 64}, V = {200K, 500K, 1M}
 * 측정: skip_ratio, push_count/V, pop_count/V, 실행 시간
 * 반복: 5회 평균 (소스 고정: 0)
 * 환경: GCP e2-standard-4
 * 출력: resultsA/experimentA_<timestamp>.csv
 */

#include <bits/stdc++.h>
using namespace std;

static const int    SEED  = 42;
static const int    REPS  = 5;
static const int    W_MAX = 100;
static const int    INF   = INT_MAX / 2;

using Graph = vector<vector<pair<int,int>>>;

// ─── Graph generation (identical to exp7) ────────────────────────────

Graph make_graph(int V, int k, int seed = SEED) {
    mt19937 rng(seed);
    uniform_int_distribution<int> nd(0, V-1);
    uniform_int_distribution<int> wd(1, W_MAX);
    Graph adj(V);
    long long target = (long long)V * k / 2;
    for (long long i = 0; i < target; i++) {
        int u = nd(rng), v = nd(rng);
        if (u != v) {
            int w = wd(rng);
            adj[u].push_back({v, w});
            adj[v].push_back({u, w});
        }
    }
    // spanning path — guarantees connectivity
    vector<int> perm(V);
    iota(perm.begin(), perm.end(), 0);
    shuffle(perm.begin(), perm.end(), rng);
    for (int i = 0; i < V-1; i++) {
        int u = perm[i], v = perm[i+1], w = wd(rng);
        adj[u].push_back({v, w});
        adj[v].push_back({u, w});
    }
    return adj;
}

// ─── Lazy deletion Dijkstra with counters ────────────────────────────

struct LazyResult {
    vector<int> dist;
    long long push_c, pop_c, skip_c;
};

LazyResult dijkstra_lazy(const Graph& adj, int src, int V) {
    vector<int> dist(V, INF);
    dist[src] = 0;
    priority_queue<pair<int,int>, vector<pair<int,int>>, greater<>> pq;
    pq.push({0, src});
    long long push_c = 1, pop_c = 0, skip_c = 0;
    while (!pq.empty()) {
        auto [d, u] = pq.top(); pq.pop();
        pop_c++;
        if (d > dist[u]) { skip_c++; continue; }
        for (auto [v, w] : adj[u]) {
            int nd2 = d + w;
            if (nd2 < dist[v]) {
                dist[v] = nd2;
                pq.push({nd2, v});
                push_c++;
            }
        }
    }
    return {dist, push_c, pop_c, skip_c};
}

// ─── Timing ──────────────────────────────────────────────────────────

using Clock = chrono::high_resolution_clock;
double ms_since(Clock::time_point t0) {
    return chrono::duration<double,milli>(Clock::now()-t0).count();
}

// ─── Main ────────────────────────────────────────────────────────────

int main() {
    system("mkdir -p resultsA");
    time_t now = time(nullptr);
    char ts[20];
    strftime(ts, sizeof(ts), "%Y%m%d_%H%M%S", localtime(&now));
    string out = string("resultsA/experimentA_") + ts + ".csv";
    ofstream csv(out);
    csv << "V,k,label,E,time_mean_ms,time_std_ms,"
           "push_count,pop_count,skip_count,skip_ratio,push_per_V,pop_per_V\n";

    struct Cfg { int V; int k; const char* label; };
    vector<Cfg> cfgs = {
        {200000,   8,  "V200k-k8"},
        {200000,  32,  "V200k-k32"},
        {200000,  64,  "V200k-k64"},
        {500000,   8,  "V500k-k8"},
        {500000,  32,  "V500k-k32"},
        {500000,  64,  "V500k-k64"},
        {1000000,  8,  "V1M-k8"},
        {1000000, 32,  "V1M-k32"},
        {1000000, 64,  "V1M-k64"},
    };

    for (auto& cfg : cfgs) {
        int V = cfg.V, k = cfg.k;
        fprintf(stderr, "[%s] Building graph V=%d k=%d ... ", cfg.label, V, k);
        fflush(stderr);
        auto tb = Clock::now();
        Graph adj = make_graph(V, k);
        fprintf(stderr, "%.1fs\n", ms_since(tb)/1000.0);

        long long E = 0;
        for (auto& row : adj) E += row.size();
        E /= 2;

        fprintf(stderr, "  Running Dijkstra x%d ... ", REPS);
        fflush(stderr);
        vector<double> times;
        LazyResult ops;
        for (int r = 0; r < REPS; r++) {
            auto t0 = Clock::now();
            ops = dijkstra_lazy(adj, 0, V);
            times.push_back(ms_since(t0));
        }

        double mn = 0, sd = 0;
        for (double t : times) mn += t;  mn /= REPS;
        for (double t : times) sd += (t-mn)*(t-mn); sd = sqrt(sd/REPS);

        double skip_ratio = (double)ops.skip_c / ops.pop_c;
        double push_per_V = (double)ops.push_c / V;
        double pop_per_V  = (double)ops.pop_c  / V;

        fprintf(stderr, "  %.1f ms/run | skip_ratio=%.4f | push/V=%.4f\n",
                mn, skip_ratio, push_per_V);

        csv << V << "," << k << "," << cfg.label << "," << E << ","
            << fixed << setprecision(4)
            << mn << "," << sd << ","
            << ops.push_c << "," << ops.pop_c << "," << ops.skip_c << ","
            << skip_ratio << "," << push_per_V << "," << pop_per_V << "\n";
        csv.flush();
    }

    csv.close();
    fprintf(stderr, "Saved -> %s\n", out.c_str());
    return 0;
}
