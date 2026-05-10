/*
 * Dijkstra 9차 실험: 실수 가중치 C++ 비교
 * ────────────────────────────────────────
 * A. std::priority_queue (lazy deletion, double key)
 * B. Fibonacci heap (from scratch, double key)
 *
 * 가중치: double uniform [0.0, 1.0]
 * 조건: k={8,32,64}, V={1k,10k,100k}
 * 출력: results9/experiment9_cpp_<timestamp>.csv
 */

#include <bits/stdc++.h>
using namespace std;

static const int    SEED  = 42;
static const int    REPS  = 10;
static const double W_LO  = 0.0;
static const double W_HI  = 1.0;
static const double INF   = 1e18;
static const int    FIB_LIMIT = 20000;

// ─── Graph (double weights) ──────────────────────────────────────────

using Graph = vector<vector<pair<int,double>>>;

Graph make_graph(int V, int k, int seed = SEED) {
    mt19937 rng(seed);
    uniform_int_distribution<int>  nd(0, V-1);
    uniform_real_distribution<double> wd(W_LO, W_HI);
    Graph adj(V);
    int target = V * k / 2;
    for (int i = 0; i < target; i++) {
        int u=nd(rng), v=nd(rng);
        if (u != v) {
            double w=wd(rng);
            adj[u].push_back({v,w}); adj[v].push_back({u,w});
        }
    }
    vector<int> perm(V); iota(perm.begin(),perm.end(),0);
    shuffle(perm.begin(),perm.end(),rng);
    for (int i=0;i<V-1;i++) {
        int u=perm[i],v=perm[i+1]; double w=wd(rng);
        adj[u].push_back({v,w}); adj[v].push_back({u,w});
    }
    return adj;
}

// ─── A. Lazy Deletion ────────────────────────────────────────────────

struct LazyResult {
    vector<double> dist;
    long long push_c, pop_c, skip_c;
};

LazyResult dijkstra_lazy(const Graph& adj, int src, int V) {
    vector<double> dist(V, INF);
    dist[src]=0;
    priority_queue<pair<double,int>,vector<pair<double,int>>,greater<>> pq;
    pq.push({0.0, src});
    long long push_c=1,pop_c=0,skip_c=0;
    while (!pq.empty()) {
        auto [d,u]=pq.top(); pq.pop(); pop_c++;
        if (d > dist[u]) { skip_c++; continue; }
        for (auto [v,w] : adj[u]) {
            double nd2=d+w;
            if (nd2 < dist[v]) {
                dist[v]=nd2; pq.push({nd2,v}); push_c++;
            }
        }
    }
    return {dist,push_c,pop_c,skip_c};
}

// ─── B. Fibonacci Heap (double key) ─────────────────────────────────

struct FNode {
    double key; int val,degree; bool mark;
    FNode *p,*c,*l,*r;
};

class FibHeap {
    FNode* H=nullptr; int n_=0;
public:
    long long ins=0,ext=0,dec_=0;

    void _add_root(FNode* x) {
        x->p=nullptr;
        if (!H) { H=x; x->l=x->r=x; return; }
        x->r=H; x->l=H->l; H->l->r=x; H->l=x;
        if (x->key < H->key) H=x;
    }

    void init_insert(FNode* x) {
        x->degree=0; x->mark=false; x->p=x->c=nullptr; x->l=x->r=x;
        _add_root(x); n_++; ins++;
    }

    void _link(FNode* y, FNode* x) {
        y->p=x; y->mark=false;
        if (!x->c) { x->c=y; y->l=y->r=y; }
        else { y->r=x->c; y->l=x->c->l; x->c->l->r=y; x->c->l=y; }
        x->degree++;
    }

    void _consolidate() {
        int md=max(2,(int)(log2(n_+1))+3);
        vector<FNode*> A(md,nullptr);
        vector<FNode*> roots;
        FNode* cur=H;
        do { roots.push_back(cur); cur=cur->r; } while (cur!=H);
        for (FNode* w : roots) {
            FNode* x=w; x->l=x->r=x;
            int d=x->degree;
            while (d<(int)A.size() && A[d]) {
                FNode* y=A[d];
                if (x->key>y->key) swap(x,y);
                _link(y,x); A[d++]=nullptr;
            }
            if (d>=(int)A.size()) A.resize(d+1,nullptr);
            A[d]=x;
        }
        H=nullptr;
        for (FNode* nd : A) {
            if (!nd) continue;
            nd->l=nd->r=nd; nd->p=nullptr;
            if (!H) H=nd;
            else {
                nd->r=H; nd->l=H->l; H->l->r=nd; H->l=nd;
                if (nd->key<H->key) H=nd;
            }
        }
    }

    FNode* extract_min() {
        FNode* z=H;
        if (!z) return nullptr;
        if (z->c) {
            vector<FNode*> ch;
            FNode* ci=z->c;
            do { ch.push_back(ci); ci=ci->r; } while (ci!=z->c);
            for (FNode* ci2:ch) { ci2->l=ci2->r=ci2; ci2->p=nullptr; _add_root(ci2); }
            z->c=nullptr;
        }
        z->l->r=z->r; z->r->l=z->l;
        if (z==z->r) H=nullptr;
        else { H=z->r; _consolidate(); }
        n_--; ext++;
        return z;
    }

    void _cut(FNode* x, FNode* y) {
        if (x->r==x) y->c=nullptr;
        else { if (y->c==x) y->c=x->r; x->l->r=x->r; x->r->l=x->l; }
        y->degree--;
        x->l=x->r=x; x->p=nullptr; x->mark=false;
        _add_root(x);
    }

    void _ccut(FNode* y) {
        FNode* z=y->p;
        if (z) { if (!y->mark) y->mark=true; else { _cut(y,z); _ccut(z); } }
    }

    void decrease_key(FNode* x, double k) {
        x->key=k;
        FNode* pp=x->p;                     // save parent BEFORE _cut clears x->p
        if (pp && x->key<pp->key) { _cut(x,pp); _ccut(pp); }
        if (H && x->key<H->key) H=x;
        dec_++;
    }

    bool empty() const { return !H; }
};

struct FibResult {
    vector<double> dist;
    long long ins_c,ext_c,dec_c;
};

FibResult dijkstra_fibonacci(const Graph& adj, int src, int V) {
    vector<double> dist(V, INF);
    dist[src]=0;
    vector<FNode> pool(V);
    vector<FNode*> nodes(V,nullptr);
    vector<bool> settled(V,false);
    FibHeap fh;
    pool[src]={0.0,src,0,false,nullptr,nullptr,nullptr,nullptr};
    nodes[src]=&pool[src];
    fh.init_insert(nodes[src]);
    while (!fh.empty()) {
        FNode* z=fh.extract_min();
        int u=z->val;
        if (settled[u]) continue;
        settled[u]=true;
        for (auto [v,w] : adj[u]) {
            if (settled[v]) continue;
            double nd2=dist[u]+w;
            if (nd2<dist[v]) {
                dist[v]=nd2;
                if (!nodes[v]) {
                    pool[v]={nd2,v,0,false,nullptr,nullptr,nullptr,nullptr};
                    nodes[v]=&pool[v];
                    fh.init_insert(nodes[v]);
                } else {
                    fh.decrease_key(nodes[v],nd2);
                }
            }
        }
    }
    return {dist,fh.ins,fh.ext,fh.dec_};
}

// ─── Timing ──────────────────────────────────────────────────────────

using Clock=chrono::high_resolution_clock;
double ms_since(Clock::time_point t0) {
    return chrono::duration<double,milli>(Clock::now()-t0).count();
}

// ─── Main ────────────────────────────────────────────────────────────

int main() {
    system("mkdir -p results9");
    time_t now=time(nullptr); char ts[20];
    strftime(ts,sizeof(ts),"%Y%m%d_%H%M%S",localtime(&now));
    string out=string("results9/experiment9_cpp_")+ts+".csv";
    ofstream csv(out);
    csv<<"V,k,label,algo,E,time_mean_ms,time_std_ms,max_err,correct,"
         "push_count,pop_count,skip_count,skip_ratio,"
         "insert_count,extract_min_count,decrease_key_count\n";

    struct Cfg{ int V,k; const char* label; };
    vector<Cfg> cfgs={
        {1000,8,"sparse"},{10000,8,"sparse"},{100000,8,"sparse"},
        {1000,32,"dense-mid"},{10000,32,"dense-mid"},{100000,32,"dense-mid"},
        {1000,64,"dense-high"},{10000,64,"dense-high"},{100000,64,"dense-high"}
    };

    for (auto& cfg:cfgs) {
        int V=cfg.V, k=cfg.k;
        fprintf(stderr,"V=%d k=%d (%s)...",V,k,cfg.label);
        Graph adj=make_graph(V,k);
        int E=0; for (auto& r:adj) E+=r.size(); E/=2;
        auto ref=dijkstra_lazy(adj,0,V);

        // heapq
        {
            vector<double> ts2; LazyResult ops;
            for (int r=0;r<REPS;r++) { auto t0=Clock::now(); ops=dijkstra_lazy(adj,0,V); ts2.push_back(ms_since(t0)); }
            double mn=0,sd=0;
            for (double t:ts2) mn+=t; mn/=REPS;
            for (double t:ts2) sd+=(t-mn)*(t-mn); sd=sqrt(sd/REPS);
            double err=0;
            for (int i=0;i<V;i++) err=max(err,abs(ops.dist[i]-ref.dist[i]));
            double sr=(double)ops.skip_c/ops.pop_c;
            csv<<V<<","<<k<<","<<cfg.label<<",heapq,"<<E<<","
               <<mn<<","<<sd<<","<<err<<","<<(err<1e-9?"true":"false")<<","
               <<ops.push_c<<","<<ops.pop_c<<","<<ops.skip_c<<","<<sr<<","
               <<",,"<<"\n";
        }
        // fibonacci
        if (V<=FIB_LIMIT) {
            vector<double> ts2; FibResult ops;
            for (int r=0;r<REPS;r++) { auto t0=Clock::now(); ops=dijkstra_fibonacci(adj,0,V); ts2.push_back(ms_since(t0)); }
            double mn=0,sd=0;
            for (double t:ts2) mn+=t; mn/=REPS;
            for (double t:ts2) sd+=(t-mn)*(t-mn); sd=sqrt(sd/REPS);
            double err=0;
            for (int i=0;i<V;i++) err=max(err,abs(ops.dist[i]-ref.dist[i]));
            csv<<V<<","<<k<<","<<cfg.label<<",fibonacci,"<<E<<","
               <<mn<<","<<sd<<","<<err<<","<<(err<1e-9?"true":"false")<<","
               <<",,,,"
               <<ops.ins_c<<","<<ops.ext_c<<","<<ops.dec_c<<"\n";
        }
        fprintf(stderr," done\n");
    }
    csv.close();
    fprintf(stderr,"Saved -> %s\n",out.c_str());
    return 0;
}
