//          Copyright Rein Halbersma 2026.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

// Stratego endgame databases for small sets of ranked pieces with the tweeveldenregel (two-square rule) in the state,
// classic 10x10 board with two lakes. At most 2 pieces per side.
// Rules: one square orthogonally; in an attack the higher rank wins, equal ranks are both removed (X > M > K > L > S > E),
// except that the spy E wins when it attacks the marshal X;
// a side without pieces has lost (both without pieces: draw); the side to move without a legal move loses, unless the
// opponent could not move either: draw.
// Tweeveldenregel with limit K: per side at most one restricted piece (square, direction back, count 0..K-1) that may
// still move back in that direction `count` more times. Moving piece p in direction e: in its restricted direction
// (count > 0 needed) -> (p, reverse e, count - 1), otherwise (p, reverse e, K - 1). When the restricted piece
// disappears, its side is free. So a piece makes at most K consecutive moves between the same two squares.
// Rank collapsing: only red-blue comparisons matter, so ranks are renumbered to the fewest levels (K vs L+S = L vs 2S,
// 2L vs L = 2S vs S). Colours: red is the side with the highest piece, on equal highest pieces the side with more pieces,
// on equal numbers the side with the stronger sorted ranks.
// Each database is written to dump/k<limit>_<key>.db (one byte per index * 2 + side to move, see encode()) and
// dump/k<limit>_<key>.bin (only the states in which nobody is restricted, placement * 2 + side to move); the directory
// dump/ must exist.
// Values: odd n = win in n plies for the side to move, even n = loss in n plies (0: no move or no pieces), 254 = draw,
// 253 = unreachable restriction state.
// Output per database: result percentages per side to move, longest win, verification, and the longest red win with
// red to move as a diagram with principal variation (winner: fastest win, loser: slowest loss).
// Usage: stratego_egdb [-k <limit>|-k inf] [-load] [-prefer] [-lean <key>] <red> <blue> [<red> <blue> ...] [-pos <red> <blue> <r|b>], e.g. "L S" "LL S" "KL L".
// -k: at most <limit> consecutive moves of one piece between the same two squares (default 3), inf = no rule.
// -load: read the databases from dump/ (as written by an earlier run) instead of computing them.
// -lean <key>: build that database with the board symmetry (the unique highest red piece in a1-e5) and one byte per
//   entry; only for databases in which red is the only side that can win (e.g. KL-LS).
// -prefer: in main lines the winner prefers moving its lowest piece towards the opponent's lowest piece (among the
//   moves that keep the fastest win).
// -pos <red> <blue> <r|b>: after building or loading, print the value, a diagram and the main line of one position,
//   e.g. stratego_egdb -k 3 -load L LS -pos Le5,Se1 Lf2 r

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <stdexcept>
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
// Values in memory are 16 bits (some 4-piece wins are longer than 252 plies). Files use one byte per entry when every
// value fits (253 = invalid, 254 = draw), otherwise two bytes (little endian, 65533 = invalid, 65534 = draw), with
// the extensions .db16 and .bin16.
using value_t = std::uint16_t;
constexpr value_t INVALID = 65533, DRAW = 65534, UNKNOWN = 65535;
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
// Ranks: E (spy) < S < L < K < M < X (marshal). The spy wins when it attacks the marshal and loses every other fight
// except against the other spy (both removed); every piece that attacks the spy wins.
constexpr int SPY = 1, MARSHAL = 8;
int rank_of(char c)
{
        switch (c) {
        case 'E': return SPY;
        case 'S': return 4;
        case 'L': return 5;
        case 'K': return 6;
        case 'M': return 7;
        case 'X': return MARSHAL;
        default: throw std::invalid_argument("rank");
        }
}
char letter(int r) { return r == SPY ? 'E' : r == MARSHAL ? 'X' : r == 7 ? 'M' : r == 6 ? 'K' : r == 5 ? 'L' : 'S'; }
// +1: the attacker wins, -1: the defender wins, 0: both are removed (canonical ranks: the spy code only occurs when
// it faces an enemy marshal, see rank_map)
inline int attack(int a, int b) { return a == SPY && b == MARSHAL ? 1 : a > b ? 1 : a < b ? -1 : 0; }

// Rank collapsing: only comparisons between the two sides matter, so ranks are renumbered to the fewest levels
// (S, L, K, M) that keep every red-blue comparison. Adjacent ranks held by one side only merge into one level.
// Then colours are normalised: red is the side with the highest piece, on a tie the side with more pieces.
// Returns true if the colours were swapped.
// Raw rank -> canonical rank. Only comparisons between red and blue matter, so the ranks are renumbered to the fewest
// levels that keep every red-blue comparison: adjacent ranks held by one side only merge into one level. A spy that
// faces an enemy marshal keeps its special role: then the spy and the marshal never merge with another rank and keep
// the codes SPY and MARSHAL, while the other ranks become S, L, K, M from the bottom up. Otherwise the spy is just the
// lowest rank and the marshal the highest.
std::array<int, 9> rank_map(std::array<std::vector<int>, 2> const& rk)
{
        auto const has = [&](int s, int r) { return std::ranges::find(rk[s], r) != rk[s].end(); };
        auto const special = (has(0, SPY) && has(1, MARSHAL)) || (has(1, SPY) && has(0, MARSHAL));
        auto const is_special = [&](int r) { return special && (r == SPY || r == MARSHAL); };
        std::vector<std::pair<int, int>> v;             // (rank, side mask)
        for (auto s = 0; s < 2; ++s) {
                for (auto r : rk[s]) {
                        auto it = std::ranges::find(v, r, &std::pair<int, int>::first);
                        if (it == v.end()) { v.push_back({ r, 1 << s }); } else { it->second |= 1 << s; }
                }
        }
        std::ranges::sort(v);
        std::vector<int> level(v.size());
        for (auto k = 1uz; k < v.size(); ++k) {
                auto const merge = !is_special(v[k].first) && !is_special(v[k - 1].first) && v[k].second == v[k - 1].second && v[k].second != 3;
                level[k] = level[k - 1] + (merge ? 0 : 1);
        }
        auto base = -1;
        for (auto k = 0uz; k < v.size(); ++k) { if (!is_special(v[k].first)) { base = level[k]; break; } }
        std::array<int, 9> to{};
        for (auto k = 0uz; k < v.size(); ++k) {
                auto const r = v[k].first;
                to[r] = is_special(r) ? r : 4 + level[k] - base;
                if (to[r] > 7 && !is_special(r)) { throw std::invalid_argument("too many distinct ranks"); }
        }
        return to;
}

// Rank collapsing (rank_map) and colour normalisation: red is the side with the highest piece, on equal highest
// pieces the side with more pieces, on equal numbers the side with the stronger sorted ranks. Returns true if the
// colours were swapped.
bool canonical_ranks(std::array<std::vector<int>, 2>& rk)
{
        auto const to = rank_map(rk);
        for (auto& side : rk) { for (auto& r : side) { r = to[r]; } std::ranges::sort(side, std::greater{}); }
        auto const top = [](std::vector<int> const& x) { return x.empty() ? 0 : x[0]; };
        // red: the highest piece, then more pieces, then the stronger sorted ranks (KL-KS, not KS-KL)
        if (top(rk[1]) > top(rk[0]) || (top(rk[1]) == top(rk[0]) && rk[1].size() > rk[0].size()) ||
            (top(rk[1]) == top(rk[0]) && rk[1].size() == rk[0].size() && rk[1] > rk[0])) {
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
        std::vector<value_t> val;                       // index * 2 + side to move
        // lean storage (-lean): one byte per entry, only for databases in which red is the only side that can win;
        // the red king (the unique highest red piece) is mirrored into the quadrant a1-e5 (sym)
        bool lean = false, sym = false, bits = false;
        int symg = 0;                                   // the unique piece mirrored into a1-e5 (a group of one)
        std::array<int, 4> gord{ 0, 1, 2, 3 };          // index order of the groups (the symmetric group first)
        std::vector<std::uint8_t> val8, out8, cnt4;
};

// Board symmetries: identity, mirror files (a<->j), mirror ranks (1<->10), both. The lakes map onto themselves and no
// square is fixed, so a position whose unique piece is mirrored into the quadrant a1-e5 has exactly one image there.
std::array<std::array<int, N>, 4> msq{};
constexpr std::array<std::array<int, 4>, 4> mdir{ { { 0, 1, 2, 3 }, { 0, 3, 2, 1 }, { 2, 1, 0, 3 }, { 2, 3, 0, 1 } } };
std::array<int, N> qidx{};                       // index of a square in the quadrant, -1 outside
std::vector<int> qsq;                            // the quadrant squares
std::string lean_key;                            // -lean <key>: build this database in lean, symmetric form
std::string bits_key;                            // -bits <key> / -check <key>: win/draw bits for red to move only
bool check_bits = false;                         // -check: build <key> normally as well and compare every state

inline void symmetrize(Slice const& sl, Pos& p)
{
        auto const k = p.side[sl.gside[sl.symg]].sq[sl.gfirst[sl.symg]];        // the unique piece
        auto const t = (fo[k] >= W / 2 ? 1 : 0) | (ro[k] >= H / 2 ? 2 : 0);
        if (t == 0) { return; }
        for (auto& sd : p.side) {
                for (auto i = 0; i < sd.n; ++i) { sd.sq[i] = msq[t][sd.sq[i]]; }
                if (sd.r.sq >= 0) { sd.r.sq = msq[t][sd.r.sq]; sd.r.dir = mdir[t][sd.r.dir]; }
                sd.normalise();
        }
}
std::map<std::string, Slice> slices;

std::uint64_t encode_raw(Slice const& sl, Pos const& p);
std::uint64_t encode(Slice const& sl, Pos const& p)
{
        if (!sl.sym) { return encode_raw(sl, p); }
        auto q = p;
        symmetrize(sl, q);
        return encode_raw(sl, q);
}

std::uint64_t encode_raw(Slice const& sl, Pos const& p)
{
        std::array<int, 4> used{};
        auto nu = 0;
        std::uint64_t idx = 0;
        for (auto gi = 0; gi < sl.ng; ++gi) {
                auto const g = sl.gord[gi];
                auto const& s = p.side[sl.gside[g]];
                std::array<int, 2> c{};
                for (auto k = 0; k < sl.gcount[g]; ++k) {
                        auto const x = s.sq[sl.gfirst[g] + k];
                        auto below = 0;
                        for (auto u = 0; u < nu; ++u) { below += used[u] < x; }
                        c[k] = (g == sl.symg && sl.sym) ? qidx[x] : x - below;
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
        for (auto gi = sl.ng - 1; gi >= 0; --gi) { auto const g = sl.gord[gi]; digit[g] = idx % sl.radix[g]; idx /= sl.radix[g]; }
        std::array<int, 4> used{};
        auto nu = 0;
        for (auto s = 0; s < 2; ++s) { p.side[s].n = sl.n[s]; for (auto k = 0; k < sl.n[s]; ++k) { p.side[s].rk[k] = sl.rk[s][k]; } }
        for (auto gi = 0; gi < sl.ng; ++gi) {
                auto const g = sl.gord[gi];
                std::array<int, 2> c{};
                if (sl.gcount[g] == 1) { c[0] = int(digit[g]); } else { c = { pairs[digit[g]].first, pairs[digit[g]].second }; }
                std::array<int, 2> placed{};
                for (auto k = 0; k < sl.gcount[g]; ++k) {
                        auto x = c[k];
                        if (g == sl.symg && sl.sym) { placed[k] = qsq[x]; continue; }
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
        std::array<int, 9> to{};                        // raw rank -> canonical rank
        std::string key;
        Slice const* slice = nullptr;                   // set once the canonical slice is built
};
std::array<Canon, 65536> canon_table;            // code: 4 bits per rank, 2 ranks per side

inline int material_code(Pos const& p)
{
        auto code = 0, shift = 0;
        for (auto const& sd : p.side) { for (auto i = 0; i < 2; ++i) { code |= (i < sd.n ? sd.rk[i] : 0) << shift; shift += 4; } }
        return code;
}

void init_canon()
{
        for (auto code = 0; code < 65536; ++code) {
                std::array<std::vector<int>, 2> rk;
                auto ok = true;
                for (auto s = 0; s < 2; ++s) {
                        for (auto i = 0; i < 2; ++i) {
                                auto const r = (code >> (4 * (2 * s + i))) & 15;
                                if (r == 0) { continue; }
                                if (r == 2 || r == 3 || r > MARSHAL) { ok = false; } else { rk[s].push_back(r); }
                        }
                }
                if (!ok || rk[0].empty() || rk[1].empty()) { continue; }
                auto& c = canon_table[code];
                c.to = rank_map(rk);
                auto mapped = rk;
                c.swap = canonical_ranks(mapped);
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

// lean byte encoding: red to move stores (n + 1) / 2 for a red win in n (odd) plies, blue to move stores n / 2 for a
// blue loss in n (even) plies; 253 = draw, 254 = invalid, 255 = unknown. Up to 504 plies.
inline std::uint8_t to8(value_t v, int stm)
{
        if (v == DRAW) { return 253; }
        if (v == INVALID) { return 254; }
        if (v == UNKNOWN) { return 255; }
        if (int(v & 1) != (stm == 0 ? 1 : 0)) { throw std::runtime_error("lean: blue can win in this database"); }
        auto const b = stm == 0 ? (v + 1) / 2 : v / 2;
        if (b > 252) { throw std::runtime_error("lean: win longer than 504 plies"); }
        return static_cast<std::uint8_t>(b);
}
inline value_t from8(std::uint8_t b, int stm)
{
        if (b == 253) { return DRAW; }
        if (b == 254) { return INVALID; }
        if (b == 255) { return UNKNOWN; }
        return static_cast<value_t>(stm == 0 ? 2 * b - 1 : 2 * b);
}
inline value_t slice_value(Slice const& sl, std::uint64_t i, int stm)
{
        return sl.lean ? from8(sl.val8[i * 2 + std::uint64_t(stm)], stm) : sl.val[i * 2 + std::uint64_t(stm)];
}

value_t child_value(Pos const& c0)
{
        auto const& me0 = c0.side[c0.stm];
        auto const& op0 = c0.side[1 - c0.stm];
        if (me0.n == 0 && op0.n == 0) { return DRAW; }
        if (me0.n == 0) { return 0; }
        if (op0.n == 0) { return 1; }
        auto c = c0;
        auto const& t = canonicalise(c);
        auto const v = slice_value(*t.slice, encode(*t.slice, c), c.stm);
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
                        auto const fight = attack(a, b);
                        if (fight >= 0) { O.remove(j); }
                        if (fight <= 0) { M.remove(i); }                // the moved piece is gone: its side is free
                        M.normalise(); O.normalise();
                        f(c, false, from, to);
                }
        }
        return n;
}

inline value_t backup(value_t c) { return c >= DRAW ? DRAW : static_cast<value_t>(c + 1); }
inline int pref(value_t v) { return v >= DRAW ? 0 : (v & 1) ? 100000 - v : -100000 + v; }

value_t terminal(Pos const& p)
{
        auto q = p;
        q.stm ^= 1;
        return for_each_move(q, [](Pos const&, bool, auto&&...) {}) > 0 ? value_t{0} : DRAW;
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
// shortest-path distances on the board (lakes excluded), filled in main
std::array<std::array<int, N>, N> dist{};
// -prefer: among the moves that keep the best value, the winning side prefers moving its lowest piece towards the
// opponent's lowest piece, then any move of its lowest piece, then the other moves
bool prefer = false;

// principal variation: the winner takes the fastest win, the loser the slowest loss
void pv(Pos p)
{
        std::cout << "PV:";
        auto move = 1;
        if (p.stm == 1) { std::cout << " 1. \u2026"; }
        for (auto ply = 0; ply < 400; ++ply) {
                if (p.side[0].n == 0 || p.side[1].n == 0) { break; }
                auto const target = child_value(p);
                auto const me = p.stm, op = 1 - p.stm;
                auto const winning = target < DRAW && (target & 1);
                auto const low = [](Side const& sd) { auto i = 0; for (auto k = 1; k < sd.n; ++k) { if (sd.rk[k] < sd.rk[i]) { i = k; } } return i; };
                auto const my_low_rank = p.side[me].rk[low(p.side[me])];
                auto const their_low = p.side[op].sq[low(p.side[op])];
                auto best_score = -1; Pos next{}; int fr = -1, tt = -1;
                for_each_move(p, [&](Pos const& c, bool, auto from, auto to) {
                        if (backup(child_value(c)) != target) { return; }
                        auto score = 0;
                        if (prefer && winning && p.side[me].rk[p.side[me].at(from)] == my_low_rank) {
                                score = 1;
                                if (dist[to][their_low] < dist[from][their_low]) { score = 2; }
                        }
                        if (score > best_score) { best_score = score; next = c; fr = from; tt = to; }
                });
                if (best_score < 0) { break; }
                auto const i = p.side[me].at(fr), j = p.side[op].at(tt);
                auto const pc = letter(p.side[me].rk[i]);
                if (me == 0) { std::cout << std::format(" {}.", move); }
                std::cout << ' ' << pc << nm(fr) << (j >= 0 ? 'x' : '-') << nm(tt);
                if (j >= 0) {
                        auto const a = p.side[me].rk[i], b = p.side[op].rk[j];
                        auto const fight = attack(a, b); if (fight == 0) { std::cout << "(ruil)"; } else if (fight < 0) { std::cout << "(sneuvelt)"; }
                }
                if (me == 1) { ++move; }
                p = next;
        }
        std::cout << '\n';
}

// Symmetric index: the first unique piece (red first) is mirrored into a1-e5 and indexed first.
void make_sym(Slice& sl)
{
        sl.sym = true;
        sl.symg = -1;
        for (auto g = 0; g < sl.ng; ++g) { if (sl.gcount[g] == 1) { sl.symg = g; break; } }
        if (sl.symg < 0) { throw std::invalid_argument("symmetric build needs a unique piece"); }
        if (sl.lean && (sl.symg != 0 || sl.gside[0] != 0)) { throw std::invalid_argument("-lean needs a unique highest red piece"); }
        auto gi = 0;
        sl.gord[gi++] = sl.symg;
        for (auto g = 0; g < sl.ng; ++g) { if (g != sl.symg) { sl.gord[gi++] = g; } }
        sl.place = 1;
        auto left = N;
        for (auto k = 0; k < sl.ng; ++k) {
                auto const g = sl.gord[k];
                sl.radix[g] = g == sl.symg ? qsq.size() : binom(left, sl.gcount[g]);
                sl.place *= sl.radix[g];
                left -= sl.gcount[g];
        }
}

// Atomic bitset over 64-bit words, for the frontier sets of the lean build.
struct Bits
{
        std::vector<std::uint64_t> w;
        void assign(std::uint64_t n) { w.assign((n + 63) / 64, 0); }
        void clear() { std::ranges::fill(w, 0); }
        void set(std::uint64_t i) { std::atomic_ref<std::uint64_t>(w[i >> 6]).fetch_or(std::uint64_t{1} << (i & 63), std::memory_order_relaxed); }
        void reset(std::uint64_t i) { std::atomic_ref<std::uint64_t>(w[i >> 6]).fetch_and(~(std::uint64_t{1} << (i & 63)), std::memory_order_relaxed); }
        bool test(std::uint64_t i) { return (std::atomic_ref<std::uint64_t>(w[i >> 6]).load(std::memory_order_relaxed) >> (i & 63)) & 1; }
        template<class F>
        void for_each(F&& f) const         // parallel over the words, f(i) for every set bit
        {
#pragma omp parallel for schedule(dynamic, 1 << 12)
                for (std::int64_t k = 0; k < static_cast<std::int64_t>(w.size()); ++k) {
                        for (auto b = w[k]; b != 0; b &= b - 1) { f(std::uint64_t(k) * 64 + std::uint64_t(std::countr_zero(b))); }
                }
        }
};

// Every parent index of a child (in the child's frame; encode() maps it to the canonical image), parent to move: mv.
template<class F>
void for_each_parent(Slice const& sl, Pos const& child, int mv, F&& f)
{
        auto const& R = child.side[mv].r;
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
                                f(encode(sl, pp));
                        }
                }
                return;
        }
        if (R.sq < 0) { return; }                       // entry state: no parent inside the database
        auto const y = R.sq, x = nb[y][R.dir], e = rev(R.dir);
        if (x < 0 || child.occupied(x)) { return; }
        auto parent = child;
        parent.stm = mv;
        auto& P = parent.side[mv];
        P.sq[P.at(y)] = x;
        P.normalise();
        auto const visit = [&](Restr const& prev) { auto pp = parent; pp.side[mv].r = prev; f(encode(sl, pp)); };
        if (R.cnt < K - 1) { visit({ x, e, R.cnt + 1 }); return; }
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

// Lean, symmetric build for one database in which only red can win (-lean): one byte per entry (to8) and three
// bitsets, no move counters. Leapfrog, two plies per iteration: the blue-to-move losses at level lv make their
// red-to-move parents wins at lv + 1; the blue-to-move parents of those wins become candidates, and every candidate is
// verified once (all its moves must lead to final red wins) and then becomes a loss at lv + 2.
void build_lean(Slice& sl, std::string const& key, std::chrono::steady_clock::time_point t0)
{
        using clock = std::chrono::steady_clock;
        auto const size = sl.size;
        auto const name = std::format("dump/k{}_{}.db8", unlimited() ? std::string("inf") : std::to_string(K), key);
        sl.val8.assign(2 * size, 255);
        if (load) {
                if (auto f = std::fopen(name.c_str(), "rb")) {
                        auto const got = std::fread(sl.val8.data(), 1, sl.val8.size(), f);
                        std::fclose(f);
                        if (got == sl.val8.size()) { std::cout << std::format("{:6} geladen uit {}\n", key, name); return; }
                }
        }
        auto const at = [&](std::uint64_t i, int s) { return std::atomic_ref<std::uint8_t>(sl.val8[i * 2 + std::uint64_t(s)]); };
        Bits A, B, T;                   // A, B: frontier / candidates (alternating roles), T: red win found by a capture
        A.assign(size); B.assign(size); T.assign(size);
        std::vector<std::vector<std::uint64_t>> pending(506);  // blue-to-move losses with a known level, sparse
        auto const elapsed = [&] { return std::chrono::duration<double>(clock::now() - t0).count(); };

#pragma omp parallel
        {
                std::vector<std::vector<std::uint64_t>> local(506);
#pragma omp for schedule(dynamic, 1 << 14)
                for (std::int64_t ii = 0; ii < static_cast<std::int64_t>(size); ++ii) {
                        auto const i = static_cast<std::uint64_t>(ii);
                        for (auto s = 0; s < 2; ++s) {
                                auto const p = decode(sl, i, s);
                                if (!valid(p)) { sl.val8[i * 2 + s] = 254; continue; }
                                auto inside = 0;
                                auto best = UNKNOWN;
                                auto const n = for_each_move(p, [&](Pos const& c, bool in, auto&&...) {
                                        if (in) { ++inside; return; }
                                        auto const v = backup(child_value(c));
                                        if (best == UNKNOWN || pref(v) > pref(best)) { best = v; }
                                });
                                auto const v = n == 0 ? terminal(p) : inside == 0 ? best : UNKNOWN;
                                if (s == 0) {           // red to move: a win by a capture is provisional unless no move stays inside
                                        auto const w = v != UNKNOWN ? v : best;
                                        if (w < DRAW && (w & 1)) { sl.val8[i * 2] = to8(w, 0); T.set(i); }
                                        else if (v != UNKNOWN) { sl.val8[i * 2] = to8(v, 0); }
                                } else if (v != UNKNOWN) {
                                        sl.val8[i * 2 + 1] = to8(v, 1);
                                        if (v < DRAW) { local[v].push_back(i); }
                                }
                        }
                }
#pragma omp critical
                for (auto l = 0; l < 506; ++l) { pending[l].insert(pending[l].end(), local[l].begin(), local[l].end()); }
        }
        std::cout << std::format("{:6} initialisatie klaar na {:.0f}s\n", key, elapsed()) << std::flush;

        auto longest = 0;
        auto* F = &A;                   // blue-to-move losses at the current level
        auto* G = &B;
        for (auto i : pending[0]) { F->set(i); }
        for (auto lv = 0; lv + 2 <= 504; lv += 2) {
                auto const win = to8(static_cast<value_t>(lv + 1), 0);
                // 1. losses at lv -> their red-to-move parents win at lv + 1 (also overriding a slower capture win)
                std::uint64_t nl = 0;
                F->for_each([&](std::uint64_t i) {
#pragma omp atomic
                        ++nl;
                        auto const child = decode(sl, i, 1);
                        for_each_parent(sl, child, 0, [&](std::uint64_t pi) {
                                auto a = at(pi, 0);
                                auto cur = a.load(std::memory_order_relaxed);
                                while (cur == 255 || (cur < 253 && cur > win && T.test(pi))) {
                                        if (a.compare_exchange_weak(cur, win)) { T.reset(pi); G->set(pi); return; }
                                }
                        });
                });
                if (nl > 0) { longest = lv; }
                F->clear();
                // 2. capture wins at lv + 1 that no faster win replaced are final now
#pragma omp parallel for schedule(dynamic, 1 << 12)
                for (std::int64_t k = 0; k < static_cast<std::int64_t>(T.w.size()); ++k) {
                        for (auto b = T.w[k]; b != 0; b &= b - 1) {
                                auto const i = std::uint64_t(k) * 64 + std::uint64_t(std::countr_zero(b));
                                if (sl.val8[i * 2] == win) { G->set(i); }
                        }
                }
                // 3. wins at lv + 1 -> their unknown blue-to-move parents become candidates (in F, now empty)
                std::uint64_t nw = 0;
                G->for_each([&](std::uint64_t i) {
#pragma omp atomic
                        ++nw;
                        auto const child = decode(sl, i, 0);
                        for_each_parent(sl, child, 1, [&](std::uint64_t pi) { if (at(pi, 1).load(std::memory_order_relaxed) == 255) { F->set(pi); } });
                });
                if (nw > 0) { longest = lv + 1; }
                G->clear();
                // 4. verify every candidate once: all moves must lead to final red wins (inside: at most lv + 1)
                std::vector<std::uint64_t> later;
                F->for_each([&](std::uint64_t i) {
                        auto const p = decode(sl, i, 1);
                        auto ok = true;
                        auto L = 0;
                        for_each_move(p, [&](Pos const& c, bool in, auto&&...) {
                                if (!ok) { return; }
                                value_t v;
                                if (in) {
                                        auto const b = sl.val8[encode(sl, c) * 2];
                                        if (b > win) { ok = false; return; }     // unknown, draw, or a capture win still provisional
                                        v = from8(b, 0);
                                } else {
                                        v = child_value(c);
                                        if (!(v < DRAW && (v & 1))) { ok = false; return; }
                                }
                                L = std::max(L, int(v) + 1);
                        });
                        if (!ok) { return; }
                        sl.val8[i * 2 + 1] = to8(static_cast<value_t>(L), 1);
                        if (L == lv + 2) { G->set(i); }
                        else {
#pragma omp critical
                                later.push_back(i);
                        }
                });
                F->clear();
                for (auto i : later) { pending[from8(sl.val8[i * 2 + 1], 1)].push_back(i); }
                for (auto i : pending[lv + 2]) { G->set(i); }
                std::swap(F, G);
                if (lv % 50 == 48) { std::cout << std::format("{:6} niveau {} na {:.0f}s\n", key, lv + 2, elapsed()) << std::flush; }
        }
        A = {}; B = {}; T = {};
        for (auto& b : sl.val8) { if (b == 255) { b = 253; } }

        std::uint64_t errors = 0;
#pragma omp parallel for schedule(dynamic, 1 << 14) reduction(+ : errors)
        for (std::int64_t ii = 0; ii < static_cast<std::int64_t>(size); ++ii) {
                auto const i = static_cast<std::uint64_t>(ii);
                for (auto s = 0; s < 2; ++s) {
                        if (sl.val8[i * 2 + s] == 254) { continue; }
                        auto const p = decode(sl, i, s);
                        errors += encode(sl, p) != i;
                        auto best = UNKNOWN;
                        auto const n = for_each_move(p, [&](Pos const& c, bool, auto&&...) {
                                auto const v = backup(child_value(c));
                                if (best == UNKNOWN || pref(v) > pref(best)) { best = v; }
                        });
                        if (n == 0) { best = terminal(p); }
                        errors += best != from8(sl.val8[i * 2 + s], s);
                }
        }

        std::array<std::array<std::uint64_t, 3>, 2> all{}, fresh{};
        std::array<int, 2> lw{};
        std::array<std::uint64_t, 2> lwi{};
        for (auto s = 0; s < 2; ++s) {
                for (auto i = std::uint64_t{0}; i < size; ++i) {
                        auto const v = from8(sl.val8[i * 2 + s], s);
                        if (v == INVALID) { continue; }
                        auto const k = v >= DRAW ? 1 : ((s == 0) == bool(v & 1)) ? 0 : 2;
                        ++all[s][k];
                        if (i % (sl.ws * sl.bs) == 0) { ++fresh[s][k]; if (k == 0 && v > lw[s]) { lw[s] = v; lwi[s] = i; } }
                }
        }
        std::cout << std::format("{:6} {:11} toestanden per kant (symmetrisch, lean), {:5.1f}s, langste {:3} plies, verify {}\n", key, size,
                std::chrono::duration<double>(clock::now() - t0).count(), longest, errors == 0 ? "OK" : std::format("{} ERRORS", errors));
        for (auto s = 0; s < 2; ++s) {
                auto const pct = [&](auto const& a, int k) { auto const t = double(a[s][0] + a[s][1] + a[s][2]); return 100.0 * double(a[s][k]) / t; };
                std::cout << std::format("       {} aan zet | alle toestanden: rood {:5.1f}% remise {:5.1f}% blauw {:5.1f}% | niemand beperkt: rood {:5.1f}% remise {:5.1f}% blauw {:5.1f}%",
                        s == 0 ? "rood " : "blauw", pct(all, 0), pct(all, 1), pct(all, 2), pct(fresh, 0), pct(fresh, 1), pct(fresh, 2));
                if (lw[s]) { std::cout << std::format("  langste rode winst {} plies: {}", lw[s], describe(decode(sl, lwi[s], s))); }
                std::cout << '\n';
        }
        if (auto f = std::fopen(name.c_str(), "wb")) { std::fwrite(sl.val8.data(), 1, sl.val8.size(), f); std::fclose(f); }
        if (lw[0]) {
                std::cout << std::format("\nlangste rode winst met rood aan zet in {}: {} plies\n", key, lw[0]);
                auto const p = decode(sl, lwi[0], 0);
                diagram(p);
                pv(p);
                std::cout << '\n';
        }
        std::cout << std::flush;
}

// -bits / -check: win/draw for red to move only, as bits, for a database in which only red can win, plus the distance
// to mate of the red-to-move states in which red is not restricted (enough to get blue to move with one ply for every
// state in which red is not restricted). Leapfrog by plies; a red win by a capture is provisional (T) until its level.
struct BitsDb
{
        Slice sl;
        Bits W;                                 // red to move wins
        std::vector<std::uint8_t> dtm;          // red to move, red not restricted: (n + 1) / 2 for a win in n plies, 0 otherwise
        bool red_free(std::uint64_t i) const { return (i / sl.bs) % sl.ws == 0; }
        std::uint64_t dtm_index(std::uint64_t i) const { return (i / (sl.ws * sl.bs)) * sl.bs + i % sl.bs; }
};

// blue to move, by one ply over red-to-move bits: {lost, distance or -1 if not stored}
std::pair<bool, int> blue_value(BitsDb& db, Pos const& p)
{
        auto lost = true;
        auto L = 0;
        auto const n = for_each_move(p, [&](Pos const& c, bool in, auto&&...) {
                if (!lost) { return; }
                if (in) {
                        auto const j = encode(db.sl, c);
                        if (!db.W.test(j)) { lost = false; return; }
                        if (L >= 0) { L = db.red_free(j) ? std::max(L, 2 * db.dtm[db.dtm_index(j)] - 1 + 1) : -1; }
                } else {
                        auto const v = child_value(c);
                        if (v >= DRAW) { lost = false; return; }
                        if (!(v & 1)) { throw std::runtime_error("bits: blue can win in this database"); }
                        if (L >= 0) { L = std::max(L, int(v) + 1); }
                }
        });
        if (n == 0) { auto const t = terminal(p); return { t == 0, t == 0 ? 0 : -1 }; }
        return { lost, lost ? L : -1 };
}

BitsDb build_bits(Slice const& base, std::string const& key)
{
        using clock = std::chrono::steady_clock;
        auto const t0 = clock::now();
        auto const elapsed = [&] { return std::chrono::duration<double>(clock::now() - t0).count(); };
        BitsDb db;
        auto& sl = db.sl;
        sl.rk = base.rk; sl.n = base.n; sl.ng = base.ng; sl.gside = base.gside; sl.gcount = base.gcount; sl.gfirst = base.gfirst;
        make_sym(sl);
        sl.ws = base.ws; sl.bs = base.bs;
        sl.size = sl.place * sl.ws * sl.bs;
        auto const size = sl.size;
        db.W.assign(size);
        db.dtm.assign(sl.place * sl.bs, 0);
        Bits F, G, T;
        F.assign(size); G.assign(size); T.assign(size);
        std::vector<std::vector<std::uint64_t>> pending(506);
        auto const record = [&](std::uint64_t i, int v) { if (db.red_free(i)) { db.dtm[db.dtm_index(i)] = static_cast<std::uint8_t>((v + 1) / 2); } };
        auto const red_capture_win = [&](Pos const& p) {
                auto best = UNKNOWN;
                for_each_move(p, [&](Pos const& c, bool in, auto&&...) {
                        if (in) { return; }
                        auto const v = backup(child_value(c));
                        if (best == UNKNOWN || pref(v) > pref(best)) { best = v; }
                });
                return best < DRAW && (best & 1) ? int(best) : -1;
        };

        // initialisation: provisional red capture wins (T), blue losses that need no move inside the database
#pragma omp parallel
        {
                std::vector<std::vector<std::uint64_t>> local(506);
#pragma omp for schedule(dynamic, 1 << 14)
                for (std::int64_t ii = 0; ii < static_cast<std::int64_t>(size); ++ii) {
                        auto const i = static_cast<std::uint64_t>(ii);
                        auto const r = decode(sl, i, 0);
                        if (!valid(r)) { continue; }
                        if (red_capture_win(r) > 0) { T.set(i); }
                        auto const b = decode(sl, i, 1);
                        auto inside = 0;
                        auto ok = true;
                        auto L = 0;
                        auto const n = for_each_move(b, [&](Pos const& c, bool in, auto&&...) {
                                if (in) { ++inside; return; }
                                auto const v = child_value(c);
                                if (v >= DRAW) { ok = false; return; }
                                if (!(v & 1)) { throw std::runtime_error("bits: blue can win in this database"); }
                                L = std::max(L, int(v) + 1);
                        });
                        if (n == 0) { if (terminal(b) == 0) { local[0].push_back(i); } continue; }
                        if (inside == 0 && ok) { local[L].push_back(i); }
                }
#pragma omp critical
                for (auto l = 0; l < 506; ++l) { pending[l].insert(pending[l].end(), local[l].begin(), local[l].end()); }
        }
        // the value of every provisional win, by rank among the T bits (prefix counts per 8 words)
        std::vector<std::uint64_t> trank(T.w.size() / 8 + 2, 0);
        for (auto k = 0uz; k < T.w.size(); ++k) { trank[k / 8 + 1] += std::uint64_t(std::popcount(T.w[k])); }
        for (auto k = 1uz; k < trank.size(); ++k) { trank[k] += trank[k - 1]; }
        auto const rank = [&](std::uint64_t i) {
                auto const k = i >> 6;
                auto r = trank[k / 8];
                for (auto m = (k / 8) * 8; m < k; ++m) { r += std::uint64_t(std::popcount(T.w[m])); }
                return r + std::uint64_t(std::popcount(T.w[k] & ((std::uint64_t{1} << (i & 63)) - 1)));
        };
        std::vector<std::uint8_t> tval(trank.back());
        T.for_each([&](std::uint64_t i) { tval[rank(i)] = static_cast<std::uint8_t>((red_capture_win(decode(sl, i, 0)) + 1) / 2); });
        auto const tmax = tval.empty() ? 0 : 2 * int(*std::ranges::max_element(tval)) - 1;      // longest provisional win
        std::cout << std::format("{:6} bits: initialisatie klaar na {:.0f}s, {} voorlopige slagwinsten\n", key, elapsed(), tval.size()) << std::flush;

        for (auto i : pending[0]) { F.set(i); }
        auto pend_max = 0;
        for (auto l = 0; l < 506; ++l) { if (!pending[l].empty()) { pend_max = l; } }
        auto frontier = pending[0].size();
        for (auto lv = 0; lv + 2 <= 504; lv += 2) {
                if (frontier == 0 && lv > pend_max && lv + 1 > tmax) { break; }        // nothing can change any more
                auto const win = lv + 1;
                // 1. blue losses at lv: their red parents win at lv + 1 (also before a slower capture win)
                F.for_each([&](std::uint64_t i) {
                        for_each_parent(sl, decode(sl, i, 1), 0, [&](std::uint64_t pi) {
                                auto const old = std::atomic_ref<std::uint64_t>(db.W.w[pi >> 6]).fetch_or(std::uint64_t{1} << (pi & 63));
                                if (!((old >> (pi & 63)) & 1)) { G.set(pi); record(pi, win); }
                        });
                });
                F.clear();
                // 2. provisional capture wins of exactly lv + 1 that nothing faster replaced
                T.for_each([&](std::uint64_t i) {
                        if (tval[rank(i)] != (win + 1) / 2) { return; }
                        auto const old = std::atomic_ref<std::uint64_t>(db.W.w[i >> 6]).fetch_or(std::uint64_t{1} << (i & 63));
                        if (!((old >> (i & 63)) & 1)) { G.set(i); record(i, win); }
                });
                // 3. red wins at lv + 1: their blue parents become candidates (in F)
                G.for_each([&](std::uint64_t i) { for_each_parent(sl, decode(sl, i, 0), 1, [&](std::uint64_t pi) { F.set(pi); }); });
                G.clear();
                // 4. every candidate once: all its moves must lead to final red wins
                std::vector<std::pair<int, std::uint64_t>> later;
                frontier = 0;
                F.for_each([&](std::uint64_t i) {
                        auto const p = decode(sl, i, 1);
                        auto ok = true;
                        auto L = lv + 2;
                        for_each_move(p, [&](Pos const& c, bool in, auto&&...) {
                                if (!ok) { return; }
                                if (in) { if (!db.W.test(encode(sl, c))) { ok = false; } return; }
                                auto const v = child_value(c);
                                if (v >= DRAW) { ok = false; return; }
                                if (!(v & 1)) { throw std::runtime_error("bits: blue can win in this database"); }
                                L = std::max(L, int(v) + 1);
                        });
                        if (!ok) { return; }
                        if (L == lv + 2) {
                                G.set(i);
#pragma omp atomic
                                ++frontier;
                        } else {
#pragma omp critical
                                later.push_back({ L, i });
                        }
                });
                F.clear();
                for (auto [L, i] : later) { pending[L].push_back(i); pend_max = std::max(pend_max, L); }
                for (auto i : pending[lv + 2]) { G.set(i); }
                frontier += pending[lv + 2].size();
                std::swap(F, G);
                if (lv % 50 == 48) { std::cout << std::format("{:6} bits: niveau {} na {:.0f}s\n", key, lv + 2, elapsed()) << std::flush; }
        }
        std::cout << std::format("{:6} bits: klaar na {:.0f}s\n", key, elapsed()) << std::flush;
        return db;
}

// statistics of a bits database (red to move from the bits, blue to move by one ply)
void report_bits(BitsDb& db, std::string const& key)
{
        auto& sl = db.sl;
        std::array<std::array<std::uint64_t, 2>, 2> all{}, fresh{};     // [stm][win, total]
        std::array<int, 2> lw{};
        for (auto s = 0; s < 2; ++s) {
                std::uint64_t a0 = 0, a1 = 0, f0 = 0, f1 = 0;
                auto l = 0;
#pragma omp parallel for schedule(dynamic, 1 << 14) reduction(+ : a0, a1, f0, f1) reduction(max : l)
                for (std::int64_t ii = 0; ii < static_cast<std::int64_t>(sl.size); ++ii) {
                        auto const i = static_cast<std::uint64_t>(ii);
                        auto const p = decode(sl, i, s);
                        if (!valid(p)) { continue; }
                        auto const fr = i % (sl.ws * sl.bs) == 0;
                        bool w;
                        auto v = -1;
                        if (s == 0) { w = db.W.test(i); if (w && db.red_free(i)) { v = 2 * db.dtm[db.dtm_index(i)] - 1; } }
                        else { auto const [lost, L] = blue_value(db, p); w = lost; v = L; }
                        ++a1; a0 += w;
                        if (fr) { ++f1; f0 += w; if (w) { l = std::max(l, v); } }
                }
                all[s] = { a0, a1 }; fresh[s] = { f0, f1 }; lw[s] = l;
        }
        std::cout << std::format("{:6} {:11} toestanden per kant (symmetrisch, bits)\n", key, sl.size);
        for (auto s = 0; s < 2; ++s) {
                std::cout << std::format("       {} aan zet | alle toestanden: rood {:5.1f}% | niemand beperkt: rood {:5.1f}%, langste rode winst {} plies\n",
                        s == 0 ? "rood " : "blauw", 100.0 * double(all[s][0]) / double(all[s][1]), 100.0 * double(fresh[s][0]) / double(fresh[s][1]), lw[s]);
        }
        std::cout << std::flush;
}

// -check: every state of the normal database against the bits database
void check_bits_against(Slice const& sl, BitsDb& db, std::string const& key)
{
        std::uint64_t states = 0, wdl = 0, dist = 0, dists = 0;
#pragma omp parallel for schedule(dynamic, 1 << 14) reduction(+ : states, wdl, dist, dists)
        for (std::int64_t ii = 0; ii < static_cast<std::int64_t>(sl.size); ++ii) {
                auto const i = static_cast<std::uint64_t>(ii);
                for (auto s = 0; s < 2; ++s) {
                        auto const v = sl.val[i * 2 + std::uint64_t(s)];
                        if (v == INVALID) { continue; }
                        ++states;
                        auto const p = decode(sl, i, s);
                        auto const redwin = v < DRAW && (bool(v & 1) == (s == 0));
                        if (v < DRAW && !redwin) { ++wdl; continue; }            // a blue win: not representable
                        if (s == 0) {
                                auto const j = encode(db.sl, p);
                                if (db.W.test(j) != redwin) { ++wdl; continue; }
                                if (redwin && db.red_free(j)) { ++dists; dist += int(2 * db.dtm[db.dtm_index(j)] - 1) != int(v); }
                        } else {
                                auto const [lost, L] = blue_value(db, p);
                                if (lost != redwin) { ++wdl; continue; }
                                if (redwin && L >= 0) { ++dists; dist += L != int(v); }
                        }
                }
        }
        std::cout << std::format("{:6} check: {} toestanden, {} uitslagverschillen; {} afstanden vergeleken, {} verschillen\n", key, states, wdl, dists, dist) << std::flush;
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
                        sl.radix[sl.ng] = binom(free, int(j - k));
                        sl.place *= sl.radix[sl.ng]; free -= int(j - k);
                        ++sl.ng; k = j;
                }
        }
        if (key == lean_key) { sl.lean = true; make_sym(sl); }
        sl.ws = unlimited() ? 1 : 1 + 4 * std::uint64_t(K) * std::uint64_t(sl.n[0]);
        sl.bs = unlimited() ? 1 : 1 + 4 * std::uint64_t(K) * std::uint64_t(sl.n[1]);
        sl.size = sl.place * sl.ws * sl.bs;
        auto const size = sl.size;
        if (sl.lean) { build_lean(sl, key, t0); return; }
        if (key == bits_key && !check_bits) { auto db = build_bits(sl, key); report_bits(db, key); return; }
        sl.val.assign(2 * size, UNKNOWN);
        if (load) {
                auto const base = std::format("dump/k{}_{}.db", unlimited() ? std::string("inf") : std::to_string(K), key);
                if (auto f = std::fopen((base + "16").c_str(), "rb")) {
                        auto const got = std::fread(sl.val.data(), sizeof(value_t), sl.val.size(), f);
                        std::fclose(f);
                        if (got == sl.val.size()) { std::cout << std::format("{:6} geladen uit {}16\n", key, base); return; }
                } else if (auto g = std::fopen(base.c_str(), "rb")) {
                        std::vector<std::uint8_t> b(sl.val.size());
                        auto const got = std::fread(b.data(), 1, b.size(), g);
                        std::fclose(g);
                        if (got == b.size()) {
                                for (auto i = 0uz; i < b.size(); ++i) { sl.val[i] = b[i] == 253 ? INVALID : b[i] == 254 ? DRAW : b[i]; }
                                std::cout << std::format("{:6} geladen uit {}\n", key, base);
                                return;
                        }
                }
        }
        std::array<std::vector<std::uint8_t>, 2> cnt, done;
        std::array<std::vector<value_t>, 2> out;
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
                        if (v == UNKNOWN) { v = static_cast<value_t>(lv); }
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
                                if (lv % 2 == 0) { pv = static_cast<value_t>(lv + 1); bucket[lv + 1].push_back(pi << 1 | std::uint64_t(mv)); return; }
                                if (--cnt[mv][pi] != 0) { return; }
                                auto const o = out[mv][pi];
                                if (o == UNKNOWN) { pv = static_cast<value_t>(lv + 1); bucket[lv + 1].push_back(pi << 1 | std::uint64_t(mv)); }
                                else if (o < DRAW && !(o & 1)) { auto const L = std::max(lv + 1, int(o)); if (L == lv + 1) { pv = static_cast<value_t>(L); } bucket[L].push_back(pi << 1 | std::uint64_t(mv)); }
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
                                        pv = static_cast<value_t>(lv + 1);
                                        bucket[lv + 1].push_back(pi << 1 | std::uint64_t(mv));
                                        return;
                                }
                                if (--cnt[mv][pi] != 0) { return; }
                                auto const o = out[mv][pi];
                                if (o == UNKNOWN) {
                                        pv = static_cast<value_t>(lv + 1);
                                        bucket[lv + 1].push_back(pi << 1 | std::uint64_t(mv));
                                } else if (o < DRAW && !(o & 1)) {
                                        auto const L = std::max(lv + 1, int(o));
                                        if (L == lv + 1) { pv = static_cast<value_t>(L); }
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
        // states in which nobody is restricted, split by parity: the side to move "has the opposition" when the sum of
        // file + rank over all pieces is odd (1 vs 1: red to move at odd distance, or blue to move at even distance)
        std::array<std::array<std::array<std::uint64_t, 3>, 2>, 2> par{};     // [side to move][opposition][result]
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
                                auto const q = decode(sl, i, s);
                                auto sum = 0;
                                for (auto const& sd : q.side) { for (auto j = 0; j < sd.n; ++j) { sum += fo[sd.sq[j]] + ro[sd.sq[j]]; } }
                                ++par[s][sum & 1][k];
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
        for (auto s = 0; s < 2; ++s) {
                auto const pct = [&](int o, int k) { auto const t = double(par[s][o][0] + par[s][o][1] + par[s][o][2]); return t > 0 ? 100.0 * double(par[s][o][k]) / t : 0.0; };
                std::cout << std::format("       {} aan zet, niemand beperkt | met oppositie: rood {:5.1f}% remise {:5.1f}% blauw {:5.1f}% | zonder oppositie: rood {:5.1f}% remise {:5.1f}% blauw {:5.1f}%\n",
                        s == 0 ? "rood " : "blauw", pct(1, 0), pct(1, 1), pct(1, 2), pct(0, 0), pct(0, 1), pct(0, 2));
        }
        {       // values of the states in which nobody is restricted: placement-major, side to move minor
                std::vector<value_t> fv(2 * sl.place);
                for (auto i = std::uint64_t{0}; i < sl.place; ++i) { for (auto s = 0; s < 2; ++s) { fv[2 * i + s] = sl.val[(i * sl.ws * sl.bs) * 2 + s]; } }
                auto const narrow = std::ranges::all_of(sl.val, [](value_t v) { return v < 253 || v >= INVALID; });
                auto const write = [&](std::vector<value_t> const& v, std::string const& ext) {
                        auto const name = std::format("dump/k{}_{}.{}{}", unlimited() ? std::string("inf") : std::to_string(K), key, ext, narrow ? "" : "16");
                        auto f = std::fopen(name.c_str(), "wb");
                        if (!f) { return; }
                        if (narrow) {
                                std::vector<std::uint8_t> b(v.size());
                                for (auto i = 0uz; i < v.size(); ++i) { b[i] = v[i] == INVALID ? 253 : v[i] == DRAW ? 254 : static_cast<std::uint8_t>(v[i]); }
                                std::fwrite(b.data(), 1, b.size(), f);
                        } else {
                                std::fwrite(v.data(), sizeof(value_t), v.size(), f);
                        }
                        std::fclose(f);
                };
                write(fv, "bin");       // nobody restricted
                write(sl.val, "db");    // the whole database, index * 2 + stm, see encode()
        }
        if (lw[0] && key != bits_key) {
                std::cout << std::format("\nlangste rode winst met rood aan zet in {}: {} plies\n", key, lw[0]);
                auto const p = decode(sl, lwi[0], 0);
                diagram(p);
                pv(p);
                std::cout << '\n';
        }
        if (key == bits_key && check_bits) { auto db = build_bits(sl, key); report_bits(db, key); check_bits_against(sl, db, key); }
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
        for (auto x = 0; x < N; ++x) {          // symmetries and the quadrant a1-e5
                auto const f = fo[x], r = ro[x];
                int const ff[4] = { f, W - 1 - f, f, W - 1 - f }, rr[4] = { r, r, H - 1 - r, H - 1 - r };
                for (auto t = 0; t < 4; ++t) { for (auto y = 0; y < N; ++y) { if (fo[y] == ff[t] && ro[y] == rr[t]) { msq[t][x] = y; } } }
                qidx[x] = -1;
                if (f < W / 2 && r < H / 2) { qidx[x] = int(qsq.size()); qsq.push_back(x); }
        }
        for (auto x = 0; x < N; ++x) {          // breadth-first distances from every square
                for (auto& v : dist[x]) { v = 1000; }
                std::vector<int> q{ x };
                dist[x][x] = 0;
                for (auto h = 0uz; h < q.size(); ++h) { for (auto y : nb[q[h]]) { if (y >= 0 && dist[x][y] == 1000) { dist[x][y] = dist[x][q[h]] + 1; q.push_back(y); } } }
        }
        auto a = 1;
        if (argc > 2 && std::string(argv[1]) == "-k") { K = std::string(argv[2]) == "inf" ? 0 : std::stoi(argv[2]); a = 3; }
        std::cout << "tweeveldenregel: " << (unlimited() ? std::string("onbeperkt") : std::format("max {} zetten achter elkaar tussen twee velden", K)) << "\n\n";
        if (a < argc && std::string(argv[a]) == "-load") { load = true; ++a; }
        if (a < argc && std::string(argv[a]) == "-prefer") { prefer = true; ++a; }
        if (a + 1 < argc && std::string(argv[a]) == "-lean") { lean_key = argv[a + 1]; a += 2; }
        if (a + 1 < argc && (std::string(argv[a]) == "-bits" || std::string(argv[a]) == "-check")) { check_bits = std::string(argv[a]) == "-check"; bits_key = argv[a + 1]; a += 2; }
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
