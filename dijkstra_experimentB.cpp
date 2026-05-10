/*
 * Dijkstra 실험 B: Dial's W 민감도 분석
 * ─────────────────────────────────────────────────────────────
 * 목적: W가 커질수록 Dial's speedup이 어떻게 변하는지 정량화
 *       → Table IV "Integer weights, any k → Dial's" 조건에
 *         W 임계값(W*) 명시 가능성 검토
 *
 * 조건: V=10,000 고정
 *       k ∈ {8, 32, 64}
 *       W ∈ {100, 500, 1000, 5000, 10000}
 * 측정: heapq 시간, Dial's 시간, speedup = heapq/dial, skip_ratio
 * 반복: 10회 평균
 * 출력: resultsB/experimentB_<timestamp>.csv
 */

#include <bits/stdc++.h>
using namespace std;

static const int SEED  = 42;
static const int REPS  = 10;
static const int INF   = INT_MAX / 2;

using Graph = vector<vector<pair<int,int>>>;

// ─── Graph generation ────────────────────────────────────────────────

Graph make_graph(int V, int k, int W, int seed = SEED) {
    mt19937 rng(seed);
    uniform_int_distribution<int> nd(0, V-1);
    uniform_int_distribution<int> wd(1, W);
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

// ─── A. Lazy deletion (heapq) ────────────────────────────────────────

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

// ─── B. Dial's bucket queue ──────────────────────────────────────────

struct DialResult {
    vector<int> dist;
    long long push_c, skip_c, settled_c;
    long long bucket_scans;          // # of (cur, b) iterations where bucket checked
    long long nonempty_bucket_scans; // # where bucket was non-empty
};

DialResult dijkstra_dial(const Graph& adj, int src, int V, int W) {
    vector<int> dist(V, INF);
    dist[src] = 0;
    int nb = W + 1;
    vector<deque<int>> bkts(nb);
    bkts[0].push_back(src);
    long long push_c = 1, skip_c = 0, settled_c = 0;
    long long bucket_scans = 0, nonempty_bucket_scans = 0;
    long long cur = 0;
    long long maxd = (long long)V * W;   // conservative upper bound

    while (settled_c < V && cur <= maxd) {
        int b = (int)(cur % nb);
        bucket_scans++;
        if (!bkts[b].empty()) {
            nonempty_bucket_scans++;
            while (!bkts[b].empty()) {
                int u = bkts[b].front(); bkts[b].pop_front();
                if (dist[u] != (int)cur) { skip_c++; continue; }
                settled_c++;
                for (auto [v, w] : adj[u]) {
                    int nd2 = (int)cur + w;
                    if (nd2 < dist[v]) {
                        dist[v] = nd2;
                        bkts[nd2 % nb].push_back(v);
                        push_c++;
                    }
                }
            }
        }
        cur++;
    }
    return {dist, push_c, skip_c, settled_c, bucket_scans, nonempty_bucket_scans};
}

// ─── Timing ──────────────────────────────────────────────────────────

using Clock = chrono::high_resolution_clock;
double ms_since(Clock::time_point t0) {
    return chrono::duration<double,milli>(Clock::now()-t0).count();
}

// ─── Main ────────────────────────────────────────────────────────────

int main() {
    system("mkdir -p resultsB");
    time_t now = time(nullptr);
    char ts[20];
    strftime(ts, sizeof(ts), "%Y%m%d_%H%M%S", localtime(&now));
    string out = string("resultsB/experimentB_") + ts + ".csv";
    ofstream csv(out);
    csv << "V,k,W,algo,E,"
           "time_mean_ms,time_std_ms,correct,"
           "push_count,pop_count,skip_count,skip_ratio,"
           "bucket_scans,nonempty_bucket_scans,speedup\n";

    int V = 10000;
    vector<int> k_vals = {8, 32, 64};
    vector<int> W_vals = {100, 500, 1000, 5000, 10000};

    for (int k : k_vals) {
        for (int W : W_vals) {
            fprintf(stderr, "V=%d k=%d W=%d ... ", V, k, W);
            fflush(stderr);

            Graph adj = make_graph(V, k, W);
            int E = 0; for (auto& row : adj) E += row.size(); E /= 2;

            auto ref = dijkstra_lazy(adj, 0, V);
            double lazy_mean = 0.0;

            // ── heapq ──────────────────────────────────────────────
            {
                vector<double> times; LazyResult ops;
                for (int r = 0; r < REPS; r++) {
                    auto t0 = Clock::now();
                    ops = dijkstra_lazy(adj, 0, V);
                    times.push_back(ms_since(t0));
                }
                double mn = 0, sd = 0;
                for (double t : times) mn += t; mn /= REPS;
                for (double t : times) sd += (t-mn)*(t-mn); sd = sqrt(sd/REPS);
                long long err = 0;
                for (int i = 0; i < V; i++)
                    err = max(err, (long long)abs(ops.dist[i]-ref.dist[i]));
                double sr = (double)ops.skip_c / ops.pop_c;
                lazy_mean = mn;
                csv << V<<","<<k<<","<<W<<",heapq,"<<E<<","
                    << mn<<","<<sd<<","<<(err==0?"true":"false")<<","
                    << ops.push_c<<","<<ops.pop_c<<","<<ops.skip_c<<","<<sr<<","
                    << ",,," << "1.0\n";
            }

            // ── Dial's ─────────────────────────────────────────────
            {
                vector<double> times; DialResult ops;
                for (int r = 0; r < REPS; r++) {
                    auto t0 = Clock::now();
                    ops = dijkstra_dial(adj, 0, V, W);
                    times.push_back(ms_since(t0));
                }
                double mn = 0, sd = 0;
                for (double t : times) mn += t; mn /= REPS;
                for (double t : times) sd += (t-mn)*(t-mn); sd = sqrt(sd/REPS);
                long long err = 0;
                for (int i = 0; i < V; i++)
                    err = max(err, (long long)abs(ops.dist[i]-ref.dist[i]));
                double stale_ratio = (ops.settled_c+ops.skip_c) > 0
                    ? (double)ops.skip_c / (ops.settled_c+ops.skip_c) : 0;
                double speedup = lazy_mean / mn;
                csv << V<<","<<k<<","<<W<<",dial,"<<E<<","
                    << mn<<","<<sd<<","<<(err==0?"true":"false")<<","
                    << ops.push_c<<","<<(ops.settled_c+ops.skip_c)<<","
                    << ops.skip_c<<","<<stale_ratio<<","
                    << ops.bucket_scans<<","<<ops.nonempty_bucket_scans<<","
                    << speedup<<"\n";

                fprintf(stderr, "speedup=%.2fx\n", speedup);
            }
        }
    }
    csv.close();
    fprintf(stderr, "Saved -> %s\n", out.c_str());
    return 0;
}
