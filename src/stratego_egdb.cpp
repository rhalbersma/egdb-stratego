//          Copyright Rein Halbersma 2026.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

// Stratego endgame databases for small sets of ranked pieces with the tweeveldenregel (two-square rule) in the state,
// classic 10x10 board with two lakes. At most 2 pieces per side.
// Rules: one square orthogonally; in an attack the higher rank wins, equal ranks are both removed (M > K > L > S);
// a side without pieces has lost (both without pieces: draw); the side to move without a legal move loses, unless the
// opponent could not move either: draw.
// Tweeveldenregel with limit K: per side at most one restricted piece (square, direction back, count 0..K-1) that may
// still move back in that direction `count` more times. Moving piece p in direction e: in its restricted direction
// (count > 0 needed) -> (p, reverse e, count - 1), otherwise (p, reverse e, K - 1). When the restricted piece
// disappears, its side is free. So a piece makes at most K consecutive moves between the same two squares.
// Rank collapsing: only red-blue comparisons matter, so ranks are renumbered to the fewest levels (K vs L+S = L vs 2S,
// 2L vs L = 2S vs S). Colours: red is the side with the highest piece, on equal highest pieces the side with more pieces.
// Each database is written to dump/k<limit>_<key>.db (one byte per index * 2 + side to move, see encode()) and
// dump/k<limit>_<key>.bin (only the states in which nobody is restricted, placement * 2 + side to move); the directory
// dump/ must exist.
// Values: odd n = win in n plies for the side to move, even n = loss in n plies (0: no move or no pieces), 254 = draw,
// 253 = unreachable restriction state.
// Output per database: result percentages per side to move, longest win, verification, and the longest red win with
// red to move as a diagram with principal variation (winner: fastest win, loser: slowest loss).
// Usage: stratego_egdb [-k <limit>|-k inf] [-load] <red> <blue> [<red> <blue> ...] [-pos <red> <blue> <r|b>], e.g. "L S" "LL S" "KL L".
// -k: at most <limit> consecutive moves of one piece between the same two squares (default 3), inf = no rule.
// -load: read the databases from dump/ (as written by an earlier run) instead of computing them.
// -pos <red> <blue> <r|b>: after building or loading, print the value, a diagram and the main line of one position,
//   e.g. stratego_egdb -k 3 -load L LS -pos Le5,Se1 Lf2 r

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <format>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace {

constexpr int W = 10, H = 10, N = 92;
constexpr std::uint8_t INVALID = 253, DRAW = 254, UNKNOWN = 255;
std::array<int, N> fo{}, ro{};
std::array<std::array<int, 4>, N> nb{};
std::vector<std::pair<int, int>> pairs;
inline int rev(int d) { return d ^ 2; }
// Tweeveldenregel limit: a piece may make at most K consecutive moves between the same two squares (K = 3: count 0..2).
// K = 0 means unlimited (no restriction state at all).
int K = 3;
bool load = false;      // -load: read existing databases from dump/ instead of computing them
inline bool unlimited() { return K == 0; }

inline std::uint64_t prank(int a, int b) { return std::uint64_t(b) * (b - 1) / 2 + a; }
inline std::uint64_t binom(int n, int k) { return k == 0 ? 1 : k == 1 ? std::uint64_t(n) : std::uint64_t(n) * (n - 1) / 2; }
int rank_of(char c) { return c == 'M' ? 7 : c == 'K' ? 6 : c == 'L' ? 5 : c == 'S' ? 4 : throw std::invalid_argument("rank"); }
char letter(int r) { return r == 7 ? 'M' : r == 6 ? 'K' : r == 5 ? 'L' : 'S'; }

// Rank collapsing: only comparisons between the two sides matter, so ranks are renumbered to the fewest levels
// (S, L, K, M) that keep every red-blue comparison. Adjacent ranks held by one side only merge into one level.
// Then colours are normalised: red is the side with the highest piece, on a tie the side with more pieces.
// Returns true if the colours were swapped.
bool canonical_ranks(std::array<std::vector<int>, 2>& rk)
{
        std::vector<std::pair<int, int>> v;             // (rank, side mask)
        for (auto s = 0; s < 2; ++s) {
                for (auto r : rk[s]) {
                        auto it = std::ranges::find(v, r, &std::pair<int, int>::first);
                        if (it == v.end()) { v.push_back({ r, 1 << s }); } else { it->second |= 1 << s; }
                }
        }
        std::ranges::sort(v);
        std::map<int, int> level;
        auto l = 0;
        for (auto k = 0u; k < v.size(); ++k) {
                if (k > 0 && !(v[k].second == v[k - 1].second && v[k].second != 3)) { ++l; }
                level[v[k].first] = 4 + l;
        }
        for (auto& side : rk) { for (auto& r : side) { r = level[r]; } std::ranges::sort(side, std::greater{}); }
        auto const top = [](std::vector<int> const& x) { return x.empty() ? 0 : x[0]; };
        if (top(rk[1]) > top(rk[0]) || (top(rk[1]) == top(rk[0]) && rk[1].size() > rk[0].size())) {
                std::swap(rk[0], rk[1]);
                return true;
        }
        return false;
}
std::string nm(int s) { return std::format("{}{}", static_cast<char>('a' + fo[s]), ro[s] + 1); }

struct Restr { int sq = -1; int dir = 0; int cnt = 0; };

struct Side
{
        std::array<int, 2> sq{ -1, -1 }, rk{ 0, 0 };
        int n = 0;
        Restr r;
        int at(int x) const { return n > 0 && sq[0] == x ? 0 : n > 1 && sq[1] == x ? 1 : -1; }
        void normalise() { if (n == 2 && (rk[0] < rk[1] || (rk[0] == rk[1] && sq[0] > sq[1]))) { std::swap(sq[0], sq[1]); std::swap(rk[0], rk[1]); } }
        void remove(int i) { if (r.sq == sq[i]) { r = {}; } if (i == 0) { sq[0] = sq[1]; rk[0] = rk[1]; } sq[1] = -1; rk[1] = 0; --n; }
        std::string key() const { std::string s; for (auto i = 0; i < n; ++i) { s += letter(rk[i]); } return s; }
};

struct Pos
{
        std::array<Side, 2> side;
        int stm = 0;
        bool occupied(int x) const { return side[0].at(x) >= 0 || side[1].at(x) >= 0; }
        std::string key() const { return side[0].key() + "-" + side[1].key(); }
};

struct Slice
{
        std::array<std::array<int, 2>, 2> rk{};         // ranks per side (sorted descending)
        std::array<int, 2> n{};
        std::array<std::uint64_t, 4> radix{};           // red group(s), blue group(s): at most 2 groups per side
        std::array<int, 4> gside{}, gcount{}, gfirst{};
        int ng = 0;
        std::uint64_t place = 0, ws = 0, bs = 0, size = 0;
        std::vector<std::uint8_t> val;                  // index * 2 + side to move
};
std::map<std::string, Slice> slices;

std::uint64_t encode(Slice const& sl, Pos const& p)
{
        std::array<int, 4> used{};
        auto nu = 0;
        std::uint64_t idx = 0;
        for (auto g = 0; g < sl.ng; ++g) {
                auto const& s = p.side[sl.gside[g]];
                std::array<int, 2> c{};
                for (auto k = 0; k < sl.gcount[g]; ++k) {
                        auto const x = s.sq[sl.gfirst[g] + k];
                        auto below = 0;
                        for (auto u = 0; u < nu; ++u) { below += used[u] < x; }
                        c[k] = x - below;
                }
                for (auto k = 0; k < sl.gcount[g]; ++k) { used[nu++] = s.sq[sl.gfirst[g] + k]; }
                idx = idx * sl.radix[g] + (sl.gcount[g] == 1 ? std::uint64_t(c[0]) : prank(c[0], c[1]));
        }
        auto const ri = [](Side const& s) -> std::uint64_t {
                if (s.r.sq < 0) { return 0; }
                return 1 + std::uint64_t(((s.r.sq == s.sq[0] ? 0 : 1) * 4 + s.r.dir) * K + s.r.cnt);
        };
        return (idx * sl.ws + ri(p.side[0])) * sl.bs + ri(p.side[1]);
}

Pos decode(Slice const& sl, std::uint64_t idx, int stm)
{
        Pos p;
        p.stm = stm;
        auto const bsi = static_cast<int>(idx % sl.bs); idx /= sl.bs;
        auto const wsi = static_cast<int>(idx % sl.ws); idx /= sl.ws;
        std::array<std::uint64_t, 4> digit{};
        for (auto g = sl.ng - 1; g >= 0; --g) { digit[g] = idx % sl.radix[g]; idx /= sl.radix[g]; }
        std::array<int, 4> used{};
        auto nu = 0;
        for (auto s = 0; s < 2; ++s) { p.side[s].n = sl.n[s]; for (auto k = 0; k < sl.n[s]; ++k) { p.side[s].rk[k] = sl.rk[s][k]; } }
        for (auto g = 0; g < sl.ng; ++g) {
                std::array<int, 2> c{};
                if (sl.gcount[g] == 1) { c[0] = int(digit[g]); } else { c = { pairs[digit[g]].first, pairs[digit[g]].second }; }
                std::array<int, 2> placed{};
                for (auto k = 0; k < sl.gcount[g]; ++k) {
                        auto x = c[k];
                        std::array<int, 4> su = used;
                        std::sort(su.begin(), su.begin() + nu);
                        for (auto u = 0; u < nu; ++u) { if (su[u] <= x) { ++x; } }
                        placed[k] = x;
                }
                for (auto k = 0; k < sl.gcount[g]; ++k) { p.side[sl.gside[g]].sq[sl.gfirst[g] + k] = placed[k]; used[nu++] = placed[k]; }
        }
        auto const rd = [](Side& s, int v) { if (v > 0) { --v; s.r = { s.sq[v / (4 * K)], (v / K) % 4, v % K }; } };
        rd(p.side[0], wsi);
        rd(p.side[1], bsi);
        return p;
}

inline bool valid(Pos const& p)
{
        for (auto const& s : p.side) { if (s.r.sq >= 0 && nb[s.r.sq][s.r.dir] < 0) { return false; } }
        return true;
}

inline Restr after_move(Restr const& r, int from, int to, int d)
{
        if (unlimited()) { return {}; }
        if (r.sq == from && r.dir == d) { return { to, rev(d), r.cnt - 1 }; }
        return { to, rev(d), K - 1 };
}
inline bool allowed(Restr const& r, int from, int d) { return !(r.sq == from && r.dir == d && r.cnt == 0); }

// Precomputed canonical form per raw material: code = 3 bits per rank, 2 ranks per side (sorted descending, 0 = none).
struct Canon
{
        bool swap = false;
        std::array<int, 8> to{};                        // raw rank -> canonical rank
        std::string key;
        Slice const* slice = nullptr;                   // set once the canonical slice is built
};
std::array<Canon, 4096> canon_table;

inline int material_code(Pos const& p)
{
        auto code = 0, shift = 0;
        for (auto const& sd : p.side) { for (auto i = 0; i < 2; ++i) { code |= (i < sd.n ? sd.rk[i] : 0) << shift; shift += 3; } }
        return code;
}

void init_canon()
{
        for (auto code = 0; code < 4096; ++code) {
                std::array<std::vector<int>, 2> rk;
                for (auto s = 0; s < 2; ++s) {
                        for (auto i = 0; i < 2; ++i) { auto const r = (code >> (3 * (2 * s + i))) & 7; if (r >= 4) { rk[s].push_back(r); } }
                }
                if (rk[0].empty() || rk[1].empty()) { continue; }
                auto mapped = rk;
                auto& c = canon_table[code];
                c.swap = canonical_ranks(mapped);
                std::vector<std::pair<int, int>> v;
                for (auto s = 0; s < 2; ++s) { for (auto r : rk[s]) { auto it = std::ranges::find(v, r, &std::pair<int, int>::first); if (it == v.end()) { v.push_back({ r, 1 << s }); } else { it->second |= 1 << s; } } }
                std::ranges::sort(v);
                auto l = 0;
                for (auto k = 0u; k < v.size(); ++k) { if (k > 0 && !(v[k].second == v[k - 1].second && v[k].second != 3)) { ++l; } c.to[v[k].first] = 4 + l; }
                for (auto r : mapped[0]) { c.key += letter(r); }
                c.key += '-';
                for (auto r : mapped[1]) { c.key += letter(r); }
        }
}

// rank collapsing and colour normalisation in place; returns the canonical table entry (no allocations)
inline Canon const& canonicalise(Pos& p)
{
        auto const& t = canon_table[material_code(p)];
        for (auto& sd : p.side) { for (auto i = 0; i < sd.n; ++i) { sd.rk[i] = t.to[sd.rk[i]]; } sd.normalise(); }
        if (t.swap) { std::swap(p.side[0], p.side[1]); p.stm ^= 1; }
        return t;
}

std::uint8_t child_value(Pos const& c0)
{
        auto const& me0 = c0.side[c0.stm];
        auto const& op0 = c0.side[1 - c0.stm];
        if (me0.n == 0 && op0.n == 0) { return DRAW; }
        if (me0.n == 0) { return 0; }
        if (op0.n == 0) { return 1; }
        auto c = c0;
        auto const& t = canonicalise(c);
        auto const v = t.slice->val[encode(*t.slice, c) * 2 + std::uint64_t(c.stm)];
        return v == UNKNOWN ? DRAW : v;
}

template<class F>
int for_each_move(Pos const& p, F&& f)
{
        auto n = 0;
        auto const me = p.stm, op = 1 - p.stm;
        for (auto i = 0; i < p.side[me].n; ++i) {
                auto const from = p.side[me].sq[i];
                for (auto d = 0; d < 4; ++d) {
                        auto const to = nb[from][d];
                        if (to < 0 || p.side[me].at(to) >= 0 || !allowed(p.side[me].r, from, d)) { continue; }
                        ++n;
                        auto c = p;
                        c.stm = op;
                        auto& M = c.side[me];
                        auto& O = c.side[op];
                        auto const j = p.side[op].at(to);
                        M.r = after_move(p.side[me].r, from, to, d);
                        M.sq[i] = to;
                        if (j < 0) { M.normalise(); f(c, true, from, to); continue; }
                        auto const a = p.side[me].rk[i], b = p.side[op].rk[j];
                        if (a >= b) { O.remove(j); }
                        if (a <= b) { M.remove(i); }                    // the moved piece is gone: its side is free
                        M.normalise(); O.normalise();
                        f(c, false, from, to);
                }
        }
        return n;
}

inline std::uint8_t backup(std::uint8_t c) { return c >= DRAW ? DRAW : static_cast<std::uint8_t>(c + 1); }
inline int pref(std::uint8_t v) { return v >= DRAW ? 0 : (v & 1) ? 1000 - v : -1000 + v; }

std::uint8_t terminal(Pos const& p)
{
        auto q = p;
        q.stm ^= 1;
        return for_each_move(q, [](Pos const&, bool, auto&&...) {}) > 0 ? std::uint8_t{0} : DRAW;
}

std::string describe(Pos const& p)
{
        std::string s = "rood";
        for (auto i = 0; i < p.side[0].n; ++i) { s += std::format(" {}{}", letter(p.side[0].rk[i]), nm(p.side[0].sq[i])); }
        s += ", blauw";
        for (auto i = 0; i < p.side[1].n; ++i) { s += std::format(" {}{}", letter(p.side[1].rk[i]), nm(p.side[1].sq[i])); }
        return s;
}


void diagram(Pos const& p)
{
        auto lake = [](int f, int r) { return (r == 4 || r == 5) && (f == 2 || f == 3 || f == 6 || f == 7); };
        std::cout << "    +" << std::string(2 * W + 1, '-') << "+\n";
        for (auto r = H - 1; r >= 0; --r) {
                std::cout << std::format("{:3} | ", r + 1);
                for (auto f = 0; f < W; ++f) {
                        auto c = lake(f, r) ? '~' : '.';
                        for (auto s = 0; s < 2; ++s) {
                                for (auto i = 0; i < p.side[s].n; ++i) {
                                        auto const x = p.side[s].sq[i];
                                        if (fo[x] == f && ro[x] == r) { c = s == 0 ? letter(p.side[s].rk[i]) : static_cast<char>(letter(p.side[s].rk[i]) + 32); }
                                }
                        }
                        std::cout << c << ' ';
                }
                std::cout << "|\n";
        }
        std::cout << "    +" << std::string(2 * W + 1, '-') << "+\n      a b c d e f g h i j      (hoofdletters rood, kleine letters blauw)\n";
}

// principal variation: the winner takes the fastest win, the loser the slowest loss
void pv(Pos p)
{
        std::cout << "PV:";
        for (auto ply = 0; ply < 240; ++ply) {
                if (p.side[0].n == 0 || p.side[1].n == 0) { break; }
                auto const target = child_value(p);
                auto found = false; Pos next{}; int fr = -1, tt = -1;
                for_each_move(p, [&](Pos const& c, bool, auto from, auto to) {
                        if (found || backup(child_value(c)) != target) { return; }
                        found = true; next = c; fr = from; tt = to;
                });
                if (!found) { break; }
                auto const me = p.stm, op = 1 - p.stm;
                auto const i = p.side[me].at(fr), j = p.side[op].at(tt);
                auto const pc = letter(p.side[me].rk[i]);
                if (me == 0) { std::cout << std::format(" {}.", ply / 2 + 1); }
                std::cout << ' ' << pc << nm(fr) << (j >= 0 ? 'x' : '-') << nm(tt);
                if (j >= 0) {
                        auto const a = p.side[me].rk[i], b = p.side[op].rk[j];
                        if (a == b) { std::cout << "(ruil)"; } else if (a < b) { std::cout << "(sneuvelt)"; }
                }
                p = next;
        }
        std::cout << '\n';
}

void build(std::string const& red0, std::string const& blue0)
{
        std::array<std::vector<int>, 2> rk;
        for (auto ch : red0) { rk[0].push_back(rank_of(ch)); }
        for (auto ch : blue0) { rk[1].push_back(rank_of(ch)); }
        canonical_ranks(rk);
        std::string red, blue;
        for (auto r : rk[0]) { red += letter(r); }
        for (auto r : rk[1]) { blue += letter(r); }
        auto const key = red + "-" + blue;
        if (slices.contains(key)) { return; }
        for (auto s = 0; s < 2; ++s) {                  // smaller materials first
                auto const& v = s == 0 ? red : blue;
                for (auto i = 0u; i < v.size(); ++i) {
                        if (i > 0 && v[i] == v[i - 1]) { continue; }
                        auto sub = v; sub.erase(i, 1);
                        if (!sub.empty()) { s == 0 ? build(sub, blue) : build(red, sub); }
                }
        }
        using clock = std::chrono::steady_clock;
        auto const t0 = clock::now();
        auto& sl = slices[key];
        for (auto& c : canon_table) { if (c.key == key) { c.slice = &sl; } }
        auto free = N;
        sl.place = 1;
        for (auto s = 0; s < 2; ++s) {
                auto const& v = s == 0 ? red : blue;
                sl.n[s] = int(v.size());
                for (auto k = 0u; k < v.size(); ++k) { sl.rk[s][k] = rank_of(v[k]); }
                for (auto k = 0u; k < v.size();) {
                        auto j = k; while (j < v.size() && v[j] == v[k]) { ++j; }
                        sl.gside[sl.ng] = s; sl.gcount[sl.ng] = int(j - k); sl.gfirst[sl.ng] = int(k);
                        sl.radix[sl.ng] = binom(free, int(j - k)); sl.place *= sl.radix[sl.ng]; free -= int(j - k);
                        ++sl.ng; k = j;
                }
        }
        sl.ws = unlimited() ? 1 : 1 + 4 * std::uint64_t(K) * std::uint64_t(sl.n[0]);
        sl.bs = unlimited() ? 1 : 1 + 4 * std::uint64_t(K) * std::uint64_t(sl.n[1]);
        sl.size = sl.place * sl.ws * sl.bs;
        auto const size = sl.size;
        sl.val.assign(2 * size, UNKNOWN);
        if (load) {
                auto const full = std::format("dump/k{}_{}.db", unlimited() ? std::string("inf") : std::to_string(K), key);
                if (auto f = std::fopen(full.c_str(), "rb")) {
                        auto const got = std::fread(sl.val.data(), 1, sl.val.size(), f);
                        std::fclose(f);
                        if (got == sl.val.size()) { std::cout << std::format("{:6} geladen uit {}\n", key, full); return; }
                }
        }
        std::array<std::vector<std::uint8_t>, 2> cnt, out, done;
        for (auto s = 0; s < 2; ++s) { cnt[s].assign(size, 0); out[s].assign(size, UNKNOWN); done[s].assign(size, 0); }

#pragma omp parallel for schedule(dynamic, 1 << 14)
        for (std::int64_t ii = 0; ii < static_cast<std::int64_t>(size); ++ii) {
                auto const i = static_cast<std::uint64_t>(ii);
                for (auto s = 0; s < 2; ++s) {
                        auto const p = decode(sl, i, s);
                        if (!valid(p)) { sl.val[(i) * 2 + s] = INVALID; continue; }
                        auto inside = 0;
                        auto best = UNKNOWN;
                        auto const n = for_each_move(p, [&](Pos const& c, bool in, auto&&...) {
                                if (in) { ++inside; return; }
                                auto const v = backup(child_value(c));
                                if (best == UNKNOWN || pref(v) > pref(best)) { best = v; }
                        });
                        if (n == 0) { sl.val[(i) * 2 + s] = terminal(p); continue; }
                        if (inside == 0) { sl.val[(i) * 2 + s] = best; continue; }
                        cnt[s][i] = static_cast<std::uint8_t>(inside);
                        out[s][i] = best;
                }
        }

        std::vector<std::vector<std::uint64_t>> bucket(INVALID);
        for (auto s = 0; s < 2; ++s) {
                for (auto i = std::uint64_t{0}; i < size; ++i) {
                        auto const v = sl.val[(i) * 2 + s];
                        if (v < INVALID) { bucket[v].push_back(i << 1 | std::uint64_t(s)); }
                        else if (v == UNKNOWN && out[s][i] < DRAW && (out[s][i] & 1)) { bucket[out[s][i]].push_back(i << 1 | std::uint64_t(s)); }
                }
        }
        auto longest = 0;
        for (auto lv = 0; lv + 1 < INVALID; ++lv) {
                auto& q = bucket[lv];
                for (auto qi = std::size_t{0}; qi < q.size(); ++qi) {
                        auto const i = q[qi] >> 1;
                        auto const s = static_cast<int>(q[qi] & 1);
                        auto& v = sl.val[(i) * 2 + s];
                        if (v == UNKNOWN) { v = static_cast<std::uint8_t>(lv); }
                        if (v != lv || done[s][i]) { continue; }
                        done[s][i] = 1;
                        longest = lv;
                        auto const child = decode(sl, i, s);
                        auto const mv = 1 - s;
                        auto const& R = child.side[mv].r;
                        auto visit_plain = [&](Pos const& pp) {         // unlimited: one parent state per unmove
                                auto const pi = encode(sl, pp);
                                auto& pv = sl.val[(pi) * 2 + mv];
                                if (pv != UNKNOWN) { return; }
                                if (lv % 2 == 0) { pv = static_cast<std::uint8_t>(lv + 1); bucket[lv + 1].push_back(pi << 1 | std::uint64_t(mv)); return; }
                                if (--cnt[mv][pi] != 0) { return; }
                                auto const o = out[mv][pi];
                                if (o == UNKNOWN) { pv = static_cast<std::uint8_t>(lv + 1); bucket[lv + 1].push_back(pi << 1 | std::uint64_t(mv)); }
                                else if (o < DRAW && !(o & 1)) { auto const L = std::max(lv + 1, int(o)); if (L == lv + 1) { pv = static_cast<std::uint8_t>(L); } bucket[L].push_back(pi << 1 | std::uint64_t(mv)); }
                        };
                        if (unlimited()) {
                                for (auto k = 0; k < child.side[mv].n; ++k) {
                                        auto const y = child.side[mv].sq[k];
                                        for (auto d = 0; d < 4; ++d) {
                                                auto const x = nb[y][d];
                                                if (x < 0 || child.occupied(x)) { continue; }
                                                auto pp = child;
                                                pp.stm = mv;
                                                pp.side[mv].sq[k] = x;
                                                pp.side[mv].normalise();
                                                visit_plain(pp);
                                        }
                                }
                                continue;
                        }
                        if (R.sq < 0) { continue; }                     // entry state: no parent inside the slice
                        auto const y = R.sq, x = nb[y][R.dir], e = rev(R.dir);
                        if (x < 0 || child.occupied(x)) { continue; }
                        auto parent = child;
                        parent.stm = mv;
                        auto& P = parent.side[mv];
                        P.sq[P.at(y)] = x;
                        P.normalise();
                        auto visit = [&](Restr const& prev) {
                                auto pp = parent;
                                pp.side[mv].r = prev;
                                auto const pi = encode(sl, pp);
                                auto& pv = sl.val[(pi) * 2 + mv];
                                if (pv != UNKNOWN) { return; }
                                if (lv % 2 == 0) {
                                        pv = static_cast<std::uint8_t>(lv + 1);
                                        bucket[lv + 1].push_back(pi << 1 | std::uint64_t(mv));
                                        return;
                                }
                                if (--cnt[mv][pi] != 0) { return; }
                                auto const o = out[mv][pi];
                                if (o == UNKNOWN) {
                                        pv = static_cast<std::uint8_t>(lv + 1);
                                        bucket[lv + 1].push_back(pi << 1 | std::uint64_t(mv));
                                } else if (o < DRAW && !(o & 1)) {
                                        auto const L = std::max(lv + 1, int(o));
                                        if (L == lv + 1) { pv = static_cast<std::uint8_t>(L); }
                                        bucket[L].push_back(pi << 1 | std::uint64_t(mv));
                                }
                        };
                        if (R.cnt < K - 1) {
                                visit({ x, e, R.cnt + 1 });
                        } else {
                                visit({});
                                for (auto d = 0; d < 4; ++d) {
                                        if (d == e || nb[x][d] < 0) { continue; }
                                        for (auto c = 0; c < K; ++c) { visit({ x, d, c }); }
                                }
                                for (auto k = 0; k < child.side[mv].n; ++k) {
                                        auto const other = child.side[mv].sq[k];
                                        if (other == y) { continue; }
                                        for (auto d = 0; d < 4; ++d) {
                                                if (nb[other][d] < 0) { continue; }
                                                for (auto c = 0; c < K; ++c) { visit({ other, d, c }); }
                                        }
                                }
                        }
                }
                std::vector<std::uint64_t>().swap(q);
        }
        std::ranges::replace(sl.val, UNKNOWN, DRAW);

        std::uint64_t errors = 0;
#pragma omp parallel for schedule(dynamic, 1 << 14) reduction(+ : errors)
        for (std::int64_t ii = 0; ii < static_cast<std::int64_t>(size); ++ii) {
                auto const i = static_cast<std::uint64_t>(ii);
                for (auto s = 0; s < 2; ++s) {
                        if (sl.val[(i) * 2 + s] == INVALID) { continue; }
                        auto const p = decode(sl, i, s);
                        errors += encode(sl, p) != i;
                        auto best = UNKNOWN;
                        auto const n = for_each_move(p, [&](Pos const& c, bool, auto&&...) {
                                auto const v = backup(child_value(c));
                                if (best == UNKNOWN || pref(v) > pref(best)) { best = v; }
                        });
                        if (n == 0) { best = terminal(p); }
                        errors += best != sl.val[(i) * 2 + s];
                }
        }

        // statistics over valid states, and over states in which nobody is restricted
        std::array<std::array<std::uint64_t, 3>, 2> all{}, fresh{};
        std::array<int, 2> lw{};
        std::array<std::uint64_t, 2> lwi{};
        for (auto s = 0; s < 2; ++s) {
                for (auto i = std::uint64_t{0}; i < size; ++i) {
                        auto const v = sl.val[(i) * 2 + s];
                        if (v == INVALID) { continue; }
                        auto const k = v >= DRAW ? 1 : ((s == 0) == bool(v & 1)) ? 0 : 2;     // 0 red wins, 1 draw, 2 blue wins
                        ++all[s][k];
                        if (i % (sl.ws * sl.bs) == 0) {
                                ++fresh[s][k];
                                if (k == 0 && v > lw[s]) { lw[s] = v; lwi[s] = i; }
                        }
                }
        }
        std::cout << std::format("{:6} {:11} toestanden per kant, {:5.1f}s, langste {:3} plies, verify {}\n", key, size,
                std::chrono::duration<double>(clock::now() - t0).count(), longest, errors == 0 ? "OK" : std::format("{} ERRORS", errors));
        for (auto s = 0; s < 2; ++s) {
                auto const pct = [&](auto const& a, int k) { auto const t = double(a[s][0] + a[s][1] + a[s][2]); return 100.0 * double(a[s][k]) / t; };
                std::cout << std::format("       {} aan zet | alle toestanden: rood {:5.1f}% remise {:5.1f}% blauw {:5.1f}% | niemand beperkt: rood {:5.1f}% remise {:5.1f}% blauw {:5.1f}%",
                        s == 0 ? "rood " : "blauw", pct(all, 0), pct(all, 1), pct(all, 2), pct(fresh, 0), pct(fresh, 1), pct(fresh, 2));
                if (lw[s]) { std::cout << std::format("  langste rode winst {} plies: {}", lw[s], describe(decode(sl, lwi[s], s))); }
                std::cout << '\n';
        }
        {       // values of the states in which nobody is restricted: placement-major, side to move minor
                std::vector<std::uint8_t> fv(2 * sl.place);
                for (auto i = std::uint64_t{0}; i < sl.place; ++i) { for (auto s = 0; s < 2; ++s) { fv[2 * i + s] = sl.val[(i * sl.ws * sl.bs) * 2 + s]; } }
                auto const name = std::format("dump/k{}_{}.bin", unlimited() ? std::string("inf") : std::to_string(K), key);
                if (auto f = std::fopen(name.c_str(), "wb")) { std::fwrite(fv.data(), 1, fv.size(), f); std::fclose(f); }
                // the whole database: one byte per (index, side to move), index * 2 + stm, see encode()
                auto const full = std::format("dump/k{}_{}.db", unlimited() ? std::string("inf") : std::to_string(K), key);
                if (auto f = std::fopen(full.c_str(), "wb")) { std::fwrite(sl.val.data(), 1, sl.val.size(), f); std::fclose(f); }
        }
        if (lw[0]) {
                std::cout << std::format("\nlangste rode winst met rood aan zet in {}: {} plies\n", key, lw[0]);
                auto const p = decode(sl, lwi[0], 0);
                diagram(p);
                pv(p);
                std::cout << '\n';
        }
        std::cout << std::flush;
}

} // namespace

int main(int argc, char* argv[])
{
        auto lake = [](int f, int r) { return (r == 4 || r == 5) && (f == 2 || f == 3 || f == 6 || f == 7); };
        std::array<std::array<int, H>, W> id{};
        auto n = 0;
        for (auto r = 0; r < H; ++r) { for (auto f = 0; f < W; ++f) { if (lake(f, r)) { id[f][r] = -1; continue; } fo[n] = f; ro[n] = r; id[f][r] = n++; } }
        int const df[4] = { 0, 1, 0, -1 }, dr[4] = { 1, 0, -1, 0 };
        for (auto s = 0; s < N; ++s) { for (auto d = 0; d < 4; ++d) { auto const f = fo[s] + df[d], r = ro[s] + dr[d]; nb[s][d] = (f < 0 || f >= W || r < 0 || r >= H) ? -1 : id[f][r]; } }
        for (auto b = 1; b < N; ++b) { for (auto a = 0; a < b; ++a) { pairs.push_back({ a, b }); } }
        init_canon();
        auto a = 1;
        if (argc > 2 && std::string(argv[1]) == "-k") { K = std::string(argv[2]) == "inf" ? 0 : std::stoi(argv[2]); a = 3; }
        std::cout << "tweeveldenregel: " << (unlimited() ? std::string("onbeperkt") : std::format("max {} zetten achter elkaar tussen twee velden", K)) << "\n\n";
        if (a < argc && std::string(argv[a]) == "-load") { load = true; ++a; }
        // -pos <red> <blue> <r|b>, e.g. -pos Le5,Se1 Lf2 r: diagram and main line (the databases must be built or loaded)
        std::array<std::string, 3> pos_arg{};
        auto have_pos = false;
        for (auto b = a; b < argc; ++b) { if (std::string(argv[b]) == "-pos" && b + 3 < argc) { pos_arg = { argv[b + 1], argv[b + 2], argv[b + 3] }; have_pos = true; argc = b; break; } }
        for (; a + 1 < argc; a += 2) {
                std::string red = argv[a], blue = argv[a + 1];
                auto const desc = [](std::string& s) { std::ranges::sort(s, [](char x, char y) { return rank_of(x) > rank_of(y); }); };
                desc(red); desc(blue);
                build(red, blue);
        }
        if (have_pos) {
                Pos p;
                for (auto s = 0; s < 2; ++s) {
                        auto const& t = pos_arg[s];
                        for (std::size_t i = 0; i < t.size();) {
                                auto const j = std::min(t.find(',', i), t.size());
                                auto const tok = t.substr(i, j - i);
                                auto const f = tok[1] - 'a', r = std::stoi(tok.substr(2)) - 1;
                                for (auto x = 0; x < N; ++x) { if (fo[x] == f && ro[x] == r) { p.side[s].sq[p.side[s].n] = x; } }
                                p.side[s].rk[p.side[s].n++] = rank_of(tok[0]);
                                i = j + 1;
                        }
                        p.side[s].normalise();
                }
                p.stm = pos_arg[2] == "r" ? 0 : 1;
                auto const v = child_value(p);
                std::cout << "\nwaarde voor de kant aan zet: " << (v >= DRAW ? std::string("remise") : std::format("{} in {} plies", (v & 1) ? "winst" : "verlies", int(v))) << "\n";
                diagram(p);
                pv(p);
        }
}
