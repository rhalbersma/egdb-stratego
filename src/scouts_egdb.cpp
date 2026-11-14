//          Copyright Rein Halbersma 2026.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

// Scout endgame VV-V: two red scouts against one blue scout, with the ISF scout two-square rule (ISF 10.2) per boundary.
// Boards: -board qa (Quick Arena, 8x8, lakes c4 c5 f4 f5) or -board classic (10x10, lakes c5 c6 d5 d6 g5 g6 h5 h6).
// Rules: a scout slides any number of vacant squares orthogonally and attacks from any distance along a vacant line;
// equal ranks are both removed. In VV-V every attack ends the game: red takes the blue scout, or blue trades its scout
// and red keeps one, so red wins after any attack. The side to move without a legal move loses.
// Two-square rule with limit K (-k 1|2|3|inf): per side the piece that moved last carries a pattern: the direction back
// along its last move, the length of that move, and per crossed boundary (the line between two adjacent squares) the
// number of consecutive moves of that piece that crossed it (1..K). A move may not cross a boundary with count K. A move
// adds 1 to the counts of the boundaries it crosses that were in the previous pattern of the same piece; all other
// counts drop. Moving the other piece starts a fresh pattern for that piece. Example (K = 3): a1-a5, a5-a3, a3-a7: the
// boundaries a3/a4 and a4/a5 have count 3, so the scout may go back to a6 and a5 but not to a4 or a3.
// Index: the blue scout is mirrored into the quadrant (a1-d4 or a1-e5), the red scouts are an unordered pair of the
// other squares, then the red pattern state (none or the pattern of one red scout) and the blue pattern state; each
// placement has its own number of pattern states, so the index uses an offset table per placement.
// Values: one byte per state and side to move: red to move t (win in 2t - 1 plies), blue to move t (loss in 2t plies),
// 255 = draw. Only red can win.
// Algorithm: leapfrog retrograde analysis, two plies per level, without move counters: the red wins at level t make
// their blue parents candidates, every candidate is verified once (all its moves lead to known red wins) and becomes a
// loss at t; the red parents of those losses are wins at t + 1. Moves backwards (unmoves) use precomputed pattern
// transition tables. -verify recomputes every value from its successors (one forward pass).
// Usage: scouts_egdb [-board qa|classic] [-k 1|2|3|inf] [-load] [-verify] [-pos <red> <blue> <r|b>]
//   e.g. scouts_egdb -board qa -k inf -pos Va1,Vh8 Vd4 r

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <format>
#include <iostream>
#include <omp.h>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

int W = 8, H = 8, N = 0, K = 3, ML = 0;
std::string board = "qa";
std::vector<std::vector<int>> grid;                     // grid[f][r] -> square, -1 = lake
std::vector<int> fo, ro;
std::vector<std::array<std::vector<int>, 4>> ray;       // squares along a direction until the edge or a lake
constexpr std::array<int, 4> DF{ 0, 1, 0, -1 }, DR{ 1, 0, -1, 0 };      // up, right, down, left
inline int rev(int d) { return d ^ 2; }
inline bool unlimited() { return K == 0; }
std::array<std::vector<int>, 4> msq;                    // board symmetries: identity, mirror files, mirror ranks, both
constexpr std::array<std::array<int, 4>, 4> mdir{ { { 0, 1, 2, 3 }, { 0, 3, 2, 1 }, { 2, 1, 0, 3 }, { 2, 3, 0, 1 } } };
std::vector<int> qidx, qsq;                             // quadrant
std::string nm(int x) { return std::format("{}{}", char('a' + fo[x]), ro[x] + 1); }

bool is_lake(int f, int r)
{
        if (board == "qa") { return (f == 2 || f == 5) && (r == 3 || r == 4); }
        return (f == 2 || f == 3 || f == 6 || f == 7) && (r == 4 || r == 5);
}

void init_board()
{
        if (board == "qa") { W = H = 8; } else if (board == "classic") { W = H = 10; } else { throw std::invalid_argument("board"); }
        grid.assign(W, std::vector<int>(H, -1));
        for (auto r = 0; r < H; ++r) { for (auto f = 0; f < W; ++f) { if (!is_lake(f, r)) { grid[f][r] = N++; fo.push_back(f); ro.push_back(r); } } }
        ray.resize(N);
        for (auto x = 0; x < N; ++x) {
                for (auto d = 0; d < 4; ++d) {
                        for (auto f = fo[x] + DF[d], r = ro[x] + DR[d]; f >= 0 && f < W && r >= 0 && r < H && grid[f][r] >= 0; f += DF[d], r += DR[d]) { ray[x][d].push_back(grid[f][r]); }
                        ML = std::max(ML, int(ray[x][d].size()));
                }
        }
        for (auto t = 0; t < 4; ++t) {
                msq[t].resize(N);
                for (auto x = 0; x < N; ++x) { msq[t][x] = grid[t & 1 ? W - 1 - fo[x] : fo[x]][t & 2 ? H - 1 - ro[x] : ro[x]]; }
        }
        qidx.assign(N, -1);
        for (auto x = 0; x < N; ++x) { if (fo[x] < W / 2 && ro[x] < H / 2) { qidx[x] = int(qsq.size()); qsq.push_back(x); } }
}

// Patterns: key = dir (2 bits) | len (4 bits) | counts (2 bits each, boundary 0 next to the scout's square).
using key_t = std::uint64_t;
inline int kdir(key_t k) { return int(k & 3); }
inline int klen(key_t k) { return int((k >> 2) & 15); }
inline int kcnt(key_t k, int i) { return int((k >> (6 + 2 * i)) & 3); }
std::vector<std::vector<key_t>> pats;                   // per square
std::vector<std::unordered_map<key_t, int>> pid;
std::vector<int> P;                                     // number of patterns per square
int PMAX = 0;
std::vector<std::vector<std::int16_t>> trans;           // trans[x][((p + 1) * 4 + dm) * ML + len - 1] -> pattern at y, -1 illegal
std::vector<std::vector<std::vector<int>>> inv;         // inv[y][q] -> patterns p at x (-1 = none) with trans = q
std::vector<std::vector<char>> ones;                    // ones[y][q]: all counts 1 (the move started a fresh pattern)
std::array<std::vector<std::vector<int>>, 4> pm;        // pattern under a symmetry

// the pattern after moving from x (pattern key old, 0 = none or another piece) in direction dm over len squares
key_t apply(key_t old, int dm, int len)
{
        auto const same = old != 0 && kdir(old) == dm;
        key_t k = key_t(rev(dm)) | key_t(len) << 2;
        for (auto i = 0; i < len; ++i) {
                auto const j = len - 1 - i;     // boundary j counted from x
                auto const oc = same && j < klen(old) ? kcnt(old, j) : 0;
                if (oc >= K) { return 0; }
                k |= key_t(oc + 1) << (6 + 2 * i);
        }
        return k;
}

void init_patterns()
{
        pats.assign(N, {});
        pid.assign(N, {});
        P.assign(N, 0);
        if (unlimited()) { return; }
        std::vector<std::pair<int, key_t>> todo;
        for (auto x = 0; x < N; ++x) { todo.push_back({ x, 0 }); }
        while (!todo.empty()) {
                auto const [x, old] = todo.back();
                todo.pop_back();
                for (auto dm = 0; dm < 4; ++dm) {
                        for (auto len = 1; len <= int(ray[x][dm].size()); ++len) {
                                auto const k = apply(old, dm, len);
                                if (k == 0) { break; }
                                auto const y = ray[x][dm][len - 1];
                                if (pid[y].emplace(k, int(pats[y].size())).second) { pats[y].push_back(k); todo.push_back({ y, k }); }
                        }
                }
        }
        for (auto x = 0; x < N; ++x) { P[x] = int(pats[x].size()); PMAX = std::max(PMAX, P[x]); }
        trans.assign(N, {});
        inv.assign(N, {});
        ones.assign(N, {});
        for (auto y = 0; y < N; ++y) {
                inv[y].assign(P[y], {});
                ones[y].assign(P[y], 0);
                for (auto q = 0; q < P[y]; ++q) {
                        auto ok = true;
                        for (auto i = 0; i < klen(pats[y][q]); ++i) { ok = ok && kcnt(pats[y][q], i) == 1; }
                        ones[y][q] = ok;
                }
        }
        for (auto x = 0; x < N; ++x) {
                trans[x].assign(std::size_t(P[x] + 1) * 4 * ML, -1);
                for (auto p = -1; p < P[x]; ++p) {
                        for (auto dm = 0; dm < 4; ++dm) {
                                for (auto len = 1; len <= int(ray[x][dm].size()); ++len) {
                                        auto const k = apply(p < 0 ? 0 : pats[x][p], dm, len);
                                        if (k == 0) { break; }
                                        auto const y = ray[x][dm][len - 1];
                                        auto const q = pid[y].at(k);
                                        trans[x][((p + 1) * 4 + dm) * ML + len - 1] = std::int16_t(q);
                                        inv[y][q].push_back(p);
                                }
                        }
                }
        }
        for (auto t = 0; t < 4; ++t) {
                pm[t].assign(N, {});
                for (auto x = 0; x < N; ++x) {
                        for (auto k : pats[x]) {
                                auto m = k;
                                m &= ~key_t(3);
                                m |= key_t(mdir[t][kdir(k)]);
                                pm[t][x].push_back(pid[msq[t][x]].at(m));
                        }
                }
        }
}

inline int trans_of(int x, int p, int dm, int len) { return trans[x][std::size_t(((p + 1) * 4 + dm) * ML + len - 1)]; }

// A state: two red scouts, one blue scout, the restricted red scout (rp, -1 = none) and its pattern rq, the blue pattern bq.
struct St
{
        std::array<int, 2> r{};
        int b = 0, rp = -1, rq = -1, bq = -1, stm = 0;
        bool occupied(int x) const { return x == r[0] || x == r[1] || x == b; }
};

// Index
int M = 0;
std::uint64_t NPAIR = 0, NPL = 0, S = 0;
std::vector<std::uint64_t> off;
std::vector<std::pair<int, int>> pairs;
inline std::uint64_t prank(int a, int c) { return std::uint64_t(c) * (c - 1) / 2 + a; }

struct Placement { int b, r0, r1; };
inline Placement placement(std::uint64_t pl)
{
        auto const qi = pl / NPAIR, pr = pl % NPAIR;
        auto const b = qsq[qi];
        auto [a, c] = pairs[pr];
        if (a >= b) { ++a; }
        if (c >= b) { ++c; }
        return { b, a, c };
}

void init_index()
{
        M = N - 1;
        NPAIR = std::uint64_t(M) * (M - 1) / 2;
        for (auto c = 1; c < M; ++c) { for (auto a = 0; a < c; ++a) { pairs.push_back({ a, c }); } }
        NPL = qsq.size() * NPAIR;
        off.assign(NPL + 1, 0);
        for (std::uint64_t pl = 0; pl < NPL; ++pl) {
                auto const [b, r0, r1] = placement(pl);
                off[pl + 1] = off[pl] + std::uint64_t(1 + P[r0] + P[r1]) * std::uint64_t(1 + P[b]);
        }
        S = off[NPL];
}

std::uint64_t encode(St const& s)
{
        auto const t = (fo[s.b] >= W / 2 ? 1 : 0) | (ro[s.b] >= H / 2 ? 2 : 0);
        auto const b = msq[t][s.b];
        std::array<int, 2> r{ msq[t][s.r[0]], msq[t][s.r[1]] };
        auto rp = s.rp;
        auto const rq = rp < 0 ? -1 : t == 0 ? s.rq : pm[t][s.r[rp]][s.rq];
        auto const bq = s.bq < 0 ? -1 : t == 0 ? s.bq : pm[t][s.b][s.bq];
        if (r[0] > r[1]) { std::swap(r[0], r[1]); if (rp >= 0) { rp = 1 - rp; } }
        auto const a = r[0] - (r[0] > b ? 1 : 0), c = r[1] - (r[1] > b ? 1 : 0);
        auto const pl = std::uint64_t(qidx[b]) * NPAIR + prank(a, c);
        auto const rs = rp < 0 ? 0 : 1 + (rp == 0 ? rq : P[r[0]] + rq);
        auto const bs = bq < 0 ? 0 : 1 + bq;
        return off[pl] + std::uint64_t(rs) * std::uint64_t(1 + P[b]) + std::uint64_t(bs);
}

inline St make_state(Placement const& pl, int rs, int bs, int stm)
{
        St s;
        s.r = { pl.r0, pl.r1 };
        s.b = pl.b;
        s.stm = stm;
        if (rs > 0) { --rs; if (rs < P[pl.r0]) { s.rp = 0; s.rq = rs; } else { s.rp = 1; s.rq = rs - P[pl.r0]; } }
        if (bs > 0) { s.bq = bs - 1; }
        return s;
}

St decode(std::uint64_t idx, int stm)
{
        auto const pl = std::uint64_t(std::upper_bound(off.begin(), off.end(), idx) - off.begin() - 1);
        auto const p = placement(pl);
        auto const rem = idx - off[pl];
        auto const nb = std::uint64_t(1 + P[p.b]);
        return make_state(p, int(rem / nb), int(rem % nb), stm);
}

// f(child, capture): every legal move of the side to move; returns the number of moves
template<class F>
int for_each_move(St const& s, F&& f)
{
        auto n = 0;
        if (s.stm == 0) {
                for (auto i = 0; i < 2; ++i) {
                        auto const x = s.r[i], p = s.rp == i ? s.rq : -1;
                        for (auto dm = 0; dm < 4; ++dm) {
                                auto const& ry = ray[x][dm];
                                for (auto len = 1; len <= int(ry.size()); ++len) {
                                        auto const y = ry[len - 1];
                                        if (y == s.r[1 - i]) { break; }
                                        auto const q = unlimited() ? -1 : trans_of(x, p, dm, len);
                                        if (!unlimited() && q < 0) { break; }
                                        ++n;
                                        if (y == s.b) { f(s, true); break; }
                                        auto c = s;
                                        c.stm = 1;
                                        c.r[i] = y;
                                        c.rp = unlimited() ? -1 : i;
                                        c.rq = q;
                                        f(c, false);
                                }
                        }
                }
        } else {
                auto const x = s.b, p = s.bq;
                for (auto dm = 0; dm < 4; ++dm) {
                        auto const& ry = ray[x][dm];
                        for (auto len = 1; len <= int(ry.size()); ++len) {
                                auto const y = ry[len - 1];
                                auto const q = unlimited() ? -1 : trans_of(x, p, dm, len);
                                if (!unlimited() && q < 0) { break; }
                                ++n;
                                if (y == s.r[0] || y == s.r[1]) { f(s, true); break; }
                                auto c = s;
                                c.stm = 0;
                                c.b = y;
                                c.bq = q;
                                f(c, false);
                        }
                }
        }
        return n;
}

// f(parent): every state with the side mv to move that has a non-capturing move to the child
template<class F>
void for_each_parent(St const& c, int mv, F&& f)
{
        if (unlimited()) {
                auto const back = [&](int y, auto&& put) {
                        for (auto d = 0; d < 4; ++d) {
                                for (auto x : ray[y][d]) {
                                        if (c.occupied(x)) { break; }
                                        auto pp = c;
                                        pp.stm = mv;
                                        put(pp, x);
                                        f(pp);
                                }
                        }
                };
                if (mv == 0) {
                        for (auto i = 0; i < 2; ++i) { back(c.r[i], [i](St& pp, int x) { pp.r[i] = x; }); }
                } else {
                        back(c.b, [](St& pp, int x) { pp.b = x; });
                }
                return;
        }
        auto const i = mv == 0 ? c.rp : 0;
        auto const q = mv == 0 ? c.rq : c.bq;
        if (q < 0) { return; }                          // nobody moved last: an entry state without parents
        auto const y = mv == 0 ? c.r[i] : c.b;
        auto const k = pats[y][q];
        auto const& ry = ray[y][kdir(k)];
        auto const len = klen(k);
        for (auto l = 0; l < len; ++l) { if (c.occupied(ry[l])) { return; } }
        auto const x = ry[len - 1];
        auto base = c;
        base.stm = mv;
        if (mv == 0) { base.r[i] = x; } else { base.b = x; }
        for (auto p : inv[y][q]) {
                auto pp = base;
                if (mv == 0) { pp.rp = p < 0 ? -1 : i; pp.rq = p; } else { pp.bq = p; }
                f(pp);
        }
        if (mv == 0 && ones[y][q]) {                    // the other red scout carried the previous pattern
                auto const o = 1 - i;
                for (auto p = 0; p < P[c.r[o]]; ++p) { auto pp = base; pp.rp = o; pp.rq = p; f(pp); }
        }
}

struct Bits
{
        std::vector<std::uint64_t> w;
        void assign(std::uint64_t n) { w.assign((n + 63) / 64, 0); }
        void clear() { std::ranges::fill(w, 0); }
        void set(std::uint64_t i) { std::atomic_ref<std::uint64_t>(w[i >> 6]).fetch_or(std::uint64_t{1} << (i & 63), std::memory_order_relaxed); }
        bool any() const { return std::ranges::any_of(w, [](auto v) { return v != 0; }); }
        std::uint64_t count() const { std::uint64_t n = 0; for (auto v : w) { n += std::uint64_t(std::popcount(v)); } return n; }
        template<class F>
        void for_each(F&& f) const
        {
#pragma omp parallel for schedule(dynamic, 1 << 10)
                for (std::int64_t k = 0; k < std::int64_t(w.size()); ++k) {
                        for (auto b = w[k]; b != 0; b &= b - 1) { f(std::uint64_t(k) * 64 + std::uint64_t(std::countr_zero(b))); }
                }
        }
};

constexpr std::uint8_t UNK = 255;
std::array<std::vector<std::uint8_t>, 2> val;
inline bool cas(std::uint8_t& v, std::uint8_t to)
{
        std::uint8_t e = UNK;
        return std::atomic_ref<std::uint8_t>(v).compare_exchange_strong(e, to, std::memory_order_relaxed);
}
inline std::uint8_t ld(std::uint8_t const& v) { return std::atomic_ref<std::uint8_t const>(v).load(std::memory_order_relaxed); }

// parallel over every state: f(idx, state) for both sides to move
template<class F>
void for_each_state(int stm, F&& f)
{
#pragma omp parallel for schedule(dynamic, 64)
        for (std::int64_t pl = 0; pl < std::int64_t(NPL); ++pl) {
                auto const p = placement(std::uint64_t(pl));
                auto const nr = 1 + P[p.r0] + P[p.r1], nb = 1 + P[p.b];
                auto idx = off[pl];
                for (auto rs = 0; rs < nr; ++rs) { for (auto bs = 0; bs < nb; ++bs, ++idx) { f(idx, make_state(p, rs, bs, stm)); } }
        }
}

using clock_t_ = std::chrono::steady_clock;
double secs(clock_t_::time_point t0) { return std::chrono::duration<double>(clock_t_::now() - t0).count(); }
std::string kname() { return unlimited() ? std::string("inf") : std::to_string(K); }

void build()
{
        auto const t0 = clock_t_::now();
        for (auto& v : val) { v.assign(S, UNK); }
        Bits Wf, Lf, Cf;
        Wf.assign(S); Lf.assign(S); Cf.assign(S);
        std::atomic<std::uint64_t> stale_red{ 0 }, stale_blue{ 0 };
        // seeds: red attacks (win in 1 ply), blue without moves (loss in 0), blue with attacks only (candidates)
        for_each_state(0, [&](std::uint64_t idx, St const& s) {
                auto cap = false;
                if (for_each_move(s, [&](St const&, bool c) { cap = cap || c; }) == 0) { ++stale_red; }
                if (cap) { val[0][idx] = 1; Wf.set(idx); }
        });
        for_each_state(1, [&](std::uint64_t idx, St const& s) {
                auto quiet = 0;
                auto const n = for_each_move(s, [&](St const&, bool c) { quiet += !c; });
                if (n == 0) { ++stale_blue; val[1][idx] = 0; Lf.set(idx); } else if (quiet == 0) { Cf.set(idx); }
        });
        if (stale_red > 0) { throw std::runtime_error(std::format("{} red states without a move", stale_red.load())); }
        std::cout << std::format("seeds: {} red wins in 1, {} blue without a move ({:.0f} s)\n", Wf.count(), stale_blue.load(), secs(t0)) << std::flush;
        auto const red_parents = [&](std::uint8_t t) {
                Lf.for_each([&](std::uint64_t i) {
                        for_each_parent(decode(i, 1), 0, [&](St const& pp) { auto const j = encode(pp); if (cas(val[0][j], t)) { Wf.set(j); } });
                });
                Lf.clear();
        };
        red_parents(1);
        for (auto t = 1; Wf.any(); ++t) {
                if (t > 254) { throw std::runtime_error("win longer than 507 plies"); }
                auto const nw = Wf.count();
                Wf.for_each([&](std::uint64_t i) {
                        for_each_parent(decode(i, 0), 1, [&](St const& pp) { auto const j = encode(pp); if (ld(val[1][j]) == UNK) { Cf.set(j); } });
                });
                Wf.clear();
                Cf.for_each([&](std::uint64_t i) {
                        if (val[1][i] != UNK) { return; }
                        auto ok = true;
                        for_each_move(decode(i, 1), [&](St const& c, bool cap) { if (ok && !cap && ld(val[0][encode(c)]) == UNK) { ok = false; } });
                        if (ok) { val[1][i] = std::uint8_t(t); Lf.set(i); }
                });
                Cf.clear();
                auto const nl = Lf.count();
                red_parents(std::uint8_t(t + 1));
                std::cout << std::format("level {:3}: red wins {:12} blue losses {:12} ({:.0f} s)\n", t, nw, nl, secs(t0)) << std::flush;
        }
        std::cout << std::format("built in {:.0f} s\n", secs(t0));
}

// Bellman check: every value equals the one computed from its successors
void verify()
{
        auto const t0 = clock_t_::now();
        std::atomic<std::uint64_t> bad{ 0 };
        for_each_state(0, [&](std::uint64_t idx, St const& s) {
                int best = UNK;
                for_each_move(s, [&](St const& c, bool cap) {
                        auto const v = cap ? 1 : val[1][encode(c)] == UNK ? int(UNK) : val[1][encode(c)] + 1;
                        best = std::min(best, v);
                });
                if (best != val[0][idx]) { ++bad; }
        });
        for_each_state(1, [&](std::uint64_t idx, St const& s) {
                auto worst = 0;
                auto const n = for_each_move(s, [&](St const& c, bool cap) {
                        auto const v = cap ? 1 : int(val[0][encode(c)]);
                        worst = std::max(worst, v);
                });
                auto const expect = n == 0 ? 0 : worst;
                if (expect != val[1][idx]) { ++bad; }
        });
        std::cout << std::format("verification: {} errors ({:.0f} s)\n", bad.load(), secs(t0));
}

bool in_line(St const& s)                       // a red scout and the blue scout see each other along a vacant line
{
        for (auto d = 0; d < 4; ++d) {
                for (auto y : ray[s.b][d]) { if (y == s.r[0] || y == s.r[1]) { return true; } }
        }
        return false;
}

void diagram(St const& s)
{
        std::cout << "    +" << std::string(2 * W + 1, '-') << "+\n";
        for (auto r = H - 1; r >= 0; --r) {
                std::cout << std::format("{:3} | ", r + 1);
                for (auto f = 0; f < W; ++f) {
                        auto const x = grid[f][r];
                        auto c = x < 0 ? '~' : '.';
                        if (x >= 0 && (x == s.r[0] || x == s.r[1])) { c = 'V'; }
                        if (x >= 0 && x == s.b) { c = 'v'; }
                        std::cout << c << ' ';
                }
                std::cout << "|\n";
        }
        std::cout << "    +" << std::string(2 * W + 1, '-') << "+\n      ";
        for (auto f = 0; f < W; ++f) { std::cout << char('a' + f) << ' '; }
        std::cout << "  (V rood, v blauw)\n";
}

inline int value(St const& s) { return val[s.stm][encode(s)]; }
std::string value_text(St const& s)
{
        auto const v = value(s);
        if (v == UNK) { return "remise"; }
        return s.stm == 0 ? std::format("rood wint in {} plies", 2 * v - 1) : std::format("blauw verliest in {} plies", 2 * v);
}

// main line: red takes the fastest win, blue the slowest loss
void pv(St s)
{
        std::cout << "PV:";
        auto move = 1;
        if (s.stm == 1) { std::cout << " 1. …"; }
        for (auto ply = 0; ply < 600 && value(s) != UNK; ++ply) {
                auto best = s.stm == 0 ? 1000 : -1;
                St next{};
                int fr = -1, to = -1;
                auto capture = false;
                auto const me = s.stm;
                // replay the moves to recover from/to squares
                for (auto i = 0; i < (me == 0 ? 2 : 1); ++i) {
                        auto const x = me == 0 ? s.r[i] : s.b;
                        for (auto dm = 0; dm < 4; ++dm) {
                                for (auto len = 1; len <= int(ray[x][dm].size()); ++len) {
                                        auto const y = ray[x][dm][len - 1];
                                        if (me == 0 && y == s.r[1 - i]) { break; }
                                        auto const p = me == 0 ? (s.rp == i ? s.rq : -1) : s.bq;
                                        auto const q = unlimited() ? -1 : trans_of(x, p, dm, len);
                                        if (!unlimited() && q < 0) { break; }
                                        auto const cap = me == 0 ? y == s.b : (y == s.r[0] || y == s.r[1]);
                                        auto c = s;
                                        c.stm = 1 - me;
                                        if (me == 0) { c.r[i] = y; c.rp = unlimited() ? -1 : i; c.rq = q; } else { c.b = y; c.bq = q; }
                                        auto const v = cap ? (me == 0 ? 0 : 1) : value(c) == UNK ? 2000 : value(c);
                                        if ((me == 0 && v < best) || (me == 1 && v > best && v != 2000)) { best = v; next = c; fr = x; to = y; capture = cap; }
                                        if (cap) { break; }
                                }
                        }
                }
                if (fr < 0) { break; }
                if (me == 0) { std::cout << std::format(" {}.", move); }
                std::cout << std::format(" V{}{}{}", nm(fr), capture ? 'x' : '-', nm(to));
                if (me == 1) { ++move; }
                if (capture) { break; }
                s = next;
        }
        std::cout << '\n';
}

void report()
{
        // per side to move: [subset][win, draw]; subsets: all, nobody restricted, nobody restricted and not in line
        std::array<std::array<std::array<std::atomic<std::uint64_t>, 2>, 3>, 2> cnt{};
        std::array<std::atomic<int>, 2> maxt{};
        std::atomic<int> longest_t{ 0 };
        for (auto stm = 0; stm < 2; ++stm) {
                for_each_state(stm, [&](std::uint64_t idx, St const& s) {
                        auto const v = val[stm][idx];
                        auto const w = v != UNK ? 0 : 1;
                        ++cnt[stm][0][w];
                        if (w == 0) { auto m = maxt[stm].load(); while (v > m && !maxt[stm].compare_exchange_weak(m, v)) {} }
                        if (s.rp < 0 && s.bq < 0) {
                                ++cnt[stm][1][w];
                                if (!in_line(s)) { ++cnt[stm][2][w]; }
                                if (stm == 0 && w == 0) {
                                        auto m = longest_t.load();
                                        while (v > m && !longest_t.compare_exchange_weak(m, v)) {}
                                }
                        }
                });
        }
        auto const pct = [](std::uint64_t a, std::uint64_t n) { return n == 0 ? 0.0 : 100.0 * double(a) / double(n); };
        std::cout << std::format("\nVV-V, bord {}, k = {}: {} toestanden per kant aan zet (symmetrisch; volledig x4)\n", board, kname(), S);
        char const* names[] = { "alle toestanden", "niemand beperkt", "niemand beperkt, niet op slag" };
        for (auto g = 0; g < 3; ++g) {
                std::cout << std::format("{:31}", names[g]);
                for (auto stm = 0; stm < 2; ++stm) {
                        auto const n = cnt[stm][g][0] + cnt[stm][g][1];
                        std::cout << std::format("  {} aan zet: winst rood {:6.2f}% remise {:6.2f}% (n = {})", stm == 0 ? "rood" : "blauw", pct(cnt[stm][g][0], n), pct(cnt[stm][g][1], n), n);
                }
                std::cout << '\n';
        }
        std::cout << std::format("langste winst: rood aan zet {} plies, blauw aan zet {} plies\n", 2 * maxt[0] - 1, 2 * maxt[1]);
        // the first longest red win with nobody restricted (the unrestricted state is the first of each placement)
        std::uint64_t li = 0;
        for (std::uint64_t pl = 0; pl < NPL; ++pl) { if (val[0][off[pl]] == longest_t) { li = off[pl]; break; } }
        if (longest_t > 0) {
                auto const s = decode(li, 0);
                std::cout << std::format("langste winst zonder beperking, rood aan zet: {} plies, rood V{} V{}, blauw V{}\n", 2 * longest_t - 1, nm(s.r[0]), nm(s.r[1]), nm(s.b));
                diagram(s);
                pv(s);
        }
}

// -strategic: the positions with nobody restricted that are drawn whichever side is to move, per blue square
void strategic()
{
        std::uint64_t n = 0, shown = 0;
        std::vector<std::uint64_t> per_blue(N, 0), per_red(N, 0);
        for (std::uint64_t pl = 0; pl < NPL; ++pl) {
                if (val[0][off[pl]] != UNK || val[1][off[pl]] != UNK) { continue; }
                auto const p = placement(pl);
                ++n; ++per_blue[p.b]; ++per_red[p.r0]; ++per_red[p.r1];
                if (shown < 3) {
                        ++shown;
                        auto const st = make_state(p, 0, 0, 0);
                        std::cout << std::format("\nstrategische remise: rood V{} V{}, blauw V{}\n", nm(p.r0), nm(p.r1), nm(p.b));
                        diagram(st);
                }
        }
        std::cout << std::format("\nstrategische remises (niemand beperkt, remise met rood en met blauw aan zet): {} van {} plaatsingen ({:.2f}%)\n", n, NPL, 100.0 * double(n) / double(NPL));
        std::cout << "per veld van de blauwe verkenner (kwadrant):\n";
        for (auto x : qsq) { std::cout << std::format("  {}: {}", nm(x), per_blue[x]); }
        std::cout << '\n';
}

std::string dump_name() { return std::format("dump/scouts_{}_k{}_VV-V.u8", board, kname()); }

void save()
{
        auto f = std::fopen(dump_name().c_str(), "wb");
        if (!f) { std::cout << "kan " << dump_name() << " niet schrijven\n"; return; }
        auto ok = true;
        for (auto const& v : val) { ok = ok && std::fwrite(v.data(), 1, v.size(), f) == v.size(); }
        ok = std::fclose(f) == 0 && ok;
        std::cout << (ok ? "weggeschreven: " : "SCHRIJFFOUT: ") << dump_name() << '\n';
        if (!ok) { std::remove(dump_name().c_str()); }
}

bool load()
{
        auto f = std::fopen(dump_name().c_str(), "rb");
        if (!f) { return false; }
        auto ok = true;
        for (auto& v : val) { v.assign(S, UNK); ok = ok && std::fread(v.data(), 1, v.size(), f) == v.size(); }
        std::fclose(f);
        return ok;
}

int square(std::string const& s)
{
        auto const f = s[0] - 'a', r = std::stoi(s.substr(1)) - 1;
        if (f < 0 || f >= W || r < 0 || r >= H || grid[f][r] < 0) { throw std::invalid_argument("square " + s); }
        return grid[f][r];
}

}       // namespace

int main(int argc, char* argv[])
{
        try {
                auto do_load = false, do_verify = false, do_strategic = false;
                std::vector<std::string> pos;
                for (auto i = 1; i < argc; ++i) {
                        std::string const a = argv[i];
                        if (a == "-board") { board = argv[++i]; }
                        else if (a == "-k") { std::string const v = argv[++i]; K = v == "inf" ? 0 : std::stoi(v); if (K < 0 || K > 3) { throw std::invalid_argument("k"); } }
                        else if (a == "-load") { do_load = true; }
                        else if (a == "-verify") { do_verify = true; }
                        else if (a == "-strategic") { do_strategic = true; }
                        else if (a == "-pos") { pos = { argv[i + 1], argv[i + 2], argv[i + 3] }; i += 3; }
                        else { throw std::invalid_argument(a); }
                }
                init_board();
                init_patterns();
                init_index();
                std::cout << std::format("bord {} ({} velden), k = {}: patronen per veld max {}, {} plaatsingen, {} toestanden per kant aan zet, {:.2f} GB\n",
                        board, N, kname(), PMAX, NPL, S, 2.0 * double(S) / 1e9) << std::flush;
                if (do_load && load()) { std::cout << "geladen: " << dump_name() << '\n'; } else { build(); save(); }
                if (do_verify) { verify(); }
                report();
                if (do_strategic) { strategic(); }
                if (!pos.empty()) {
                        St s;
                        auto const rs = pos[0];
                        auto const c = rs.find(',');
                        s.r = { square(rs.substr(1, c - 1)), square(rs.substr(c + 2)) };
                        s.b = square(pos[1].substr(1));
                        s.stm = pos[2] == "b" ? 1 : 0;
                        std::cout << '\n' << value_text(s) << '\n';
                        diagram(s);
                        pv(s);
                }
        } catch (std::exception const& e) {
                std::cerr << "fout: " << e.what() << '\n';
                return 1;
        }
}
