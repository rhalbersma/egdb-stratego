//          Copyright Rein Halbersma 2026.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

// Stratego endgame databases without the two-squares rule:
//   2 lieutenants (white) vs 1 sergeant (black), then 2 lieutenants vs 2 sergeants,
// on the classic 10x10 board with the two lakes.
// Rules: one square orthogonally; a lieutenant attacking a sergeant captures it; a sergeant attacking a lieutenant
// is lost (a legal move, so it is included); the side to move without a legal move loses, unless the opponent could
// not move either: then it is a draw. A move that leaves the opponent without pieces wins.
// Forward iteration per ply with OpenMP; values: even = loss in n plies for the side to move, odd = win, DRAW.

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <format>
#include <iostream>
#include <vector>

namespace {

constexpr int W = 10, H = 10, N = 92;
constexpr std::uint8_t DRAW = 254, UNKNOWN = 255;
std::array<std::array<int, 4>, N> nb;
std::array<int, N> fo, ro;
std::vector<std::pair<int, int>> pairs;                 // pair rank -> (a, b), a < b

inline std::uint64_t prank(int a, int b) { if (a > b) std::swap(a, b); return std::uint64_t(b) * (b - 1) / 2 + a; }
inline int compress(int s, int a, int b) { return s - (a < s) - (b < s); }
inline int expand(int c, int a, int b) { if (a <= c) ++c; if (b <= c) ++c; return c; }

struct Pos { int a, b; int s1, s2; int ns; int stm; };      // ns = number of sergeants (1 or 2); s2 unused if ns == 1

constexpr std::uint64_t P2 = std::uint64_t{N} * (N - 1) / 2;
constexpr std::uint64_t S21 = P2 * (N - 2);
constexpr std::uint64_t S22 = P2 * ((N - 2) * (N - 3) / 2);

std::array<std::vector<std::uint8_t>, 2> v21, v22;       // per side to move

inline std::uint64_t encode(Pos const& p)
{
        auto const [a, b] = std::minmax(p.a, p.b);
        if (p.ns == 1) { return prank(a, b) * (N - 2) + static_cast<std::uint64_t>(compress(p.s1, a, b)); }
        return prank(a, b) * ((N - 2) * (N - 3) / 2) + prank(compress(p.s1, a, b), compress(p.s2, a, b));
}

inline Pos decode(std::uint64_t i, int ns, int stm)
{
        if (ns == 1) {
                auto const [a, b] = pairs[i / (N - 2)];
                return { a, b, expand(static_cast<int>(i % (N - 2)), a, b), -1, 1, stm };
        }
        auto const [a, b] = pairs[i / ((N - 2) * (N - 3) / 2)];
        auto const [c1, c2] = pairs[i % ((N - 2) * (N - 3) / 2)];
        return { a, b, expand(c1, a, b), expand(c2, a, b), 2, stm };
}

inline std::uint8_t value(Pos const& p)
{
        auto& v = p.ns == 1 ? v21 : v22;
        return v[p.stm][encode(p)];
}

// children of p: calls f(child_value, final) where final = child from a smaller (finished) database or a game end
template<class F>
int children(Pos const& p, F&& f)
{
        auto n = 0;
        auto const occupied = [&](int x) { return x == p.a || x == p.b || x == p.s1 || (p.ns == 2 && x == p.s2); };
        if (p.stm == 0) {
                for (auto k = 0; k < 2; ++k) {
                        auto const from = k ? p.b : p.a, other = k ? p.a : p.b;
                        for (auto d = 0; d < 4; ++d) {
                                auto const to = nb[from][d];
                                if (to < 0 || to == other) { continue; }
                                ++n;
                                auto c = p;
                                (k ? c.b : c.a) = to;
                                c.stm = 1;
                                if (to == p.s1 || (p.ns == 2 && to == p.s2)) {              // capture
                                        if (p.ns == 1) { auto e = c; e.ns = 0; f(std::uint8_t{0}, true, e, from, to); continue; }      // black has no pieces: lost
                                        c.s1 = to == p.s1 ? p.s2 : p.s1; c.s2 = -1; c.ns = 1;
                                        f(v21[1][encode(c)], true, c, from, to);
                                } else {
                                        f(value(c), false, c, from, to);
                                }
                        }
                }
        } else {
                for (auto k = 0; k < p.ns; ++k) {
                        auto const from = k ? p.s2 : p.s1;
                        for (auto d = 0; d < 4; ++d) {
                                auto const to = nb[from][d];
                                if (to < 0 || (p.ns == 2 && to == (k ? p.s1 : p.s2))) { continue; }
                                ++n;
                                auto c = p;
                                c.stm = 0;
                                if (to == p.a || to == p.b) {                               // the sergeant attacks and is lost
                                        if (p.ns == 1) { auto e = c; e.ns = 0; f(std::uint8_t{1}, true, e, from, to); continue; }      // white to move with black gone: white has won
                                        c.s1 = k ? p.s1 : p.s2; c.s2 = -1; c.ns = 1;
                                        f(v21[0][encode(c)], true, c, from, to);
                                } else {
                                        (k ? c.s2 : c.s1) = to;
                                        f(value(c), false, c, from, to);
                                }
                        }
                }
        }
        (void)occupied;
        return n;
}

std::uint8_t evaluate(Pos const& p, int ply, int sub_max)
{
        auto all_win = true, any = false;
        auto win = false;
        auto n = children(p, [&](std::uint8_t c, bool fin, auto&&...) {
                any = true;
                (void)fin;
                if (c >= DRAW || c >= ply) { all_win = false; return; }
                if ((c & 1) == 0) { win = true; all_win = false; }     // child lost in c < ply plies
        });
        (void)sub_max;
        if (n == 0) {
                if (ply != 0) { return UNKNOWN; }
                auto q = p;
                q.stm ^= 1;
                auto const m = children(q, [](std::uint8_t, bool, auto&&...) {});
                return m > 0 ? std::uint8_t{0} : DRAW;
        }
        if (ply == 0) { return UNKNOWN; }
        // win: some child lost in exactly ply - 1 (earlier ones would have been found before)
        if (win) {
                auto exact = false;
                children(p, [&](std::uint8_t c, bool, auto&&...) { if (c < DRAW && (c & 1) == 0 && c == ply - 1) { exact = true; } });
                if (exact) { return static_cast<std::uint8_t>(ply); }
        }
        if (all_win && any) { return static_cast<std::uint8_t>(ply); }
        return UNKNOWN;
}

void solve(int ns, std::array<std::vector<std::uint8_t>, 2>& v, std::uint64_t size, int sub_max, char const* label)
{
        auto const t0 = std::chrono::steady_clock::now();
        for (auto& x : v) { x.assign(size, UNKNOWN); }
        // positions where two sergeants would overlap cannot occur (compact index), nothing to mark
        auto last = 0;
        for (auto ply = 0; ply < 253; ++ply) {
                std::uint64_t changed = 0;
#pragma omp parallel for schedule(dynamic, 1 << 14) reduction(+ : changed)
                for (std::int64_t ii = 0; ii < static_cast<std::int64_t>(size); ++ii) {
                        auto const i = static_cast<std::uint64_t>(ii);
                        for (auto s = 0; s < 2; ++s) {
                                if (v[s][i] != UNKNOWN) { continue; }
                                auto const r = evaluate(decode(i, ns, s), ply, sub_max);
                                if (r != UNKNOWN) { v[s][i] = r; ++changed; }
                        }
                }
                if (changed) { last = ply; std::cout << std::format("  {} ply {:3}: {:10}\n", label, ply, changed) << std::flush; }
                if (ply > 0 && changed == 0 && ply > sub_max + 1) { break; }
        }
        for (auto& x : v) { std::ranges::replace(x, UNKNOWN, DRAW); }
        std::array<std::array<std::uint64_t, 3>, 2> c{};
        for (auto s = 0; s < 2; ++s) { for (auto x : v[s]) { ++c[s][x >= DRAW ? 1 : (x & 1) ? 0 : 2]; } }
        std::cout << std::format("{}: {:.1f}s, longest {} plies\n  white to move W {:10} D {:10} L {:10}\n  black to move W {:10} D {:10} L {:10}\n",
                label, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), last,
                c[0][0], c[0][1], c[0][2], c[1][0], c[1][1], c[1][2]);
}

int max_dtm(std::array<std::vector<std::uint8_t>, 2> const& v)
{
        auto m = 0;
        for (auto const& x : v) { for (auto y : x) { if (y < DRAW) { m = std::max(m, int(y)); } } }
        return m;
}


// Bellman check: every value must equal the best over its children
std::uint64_t verify(int ns, std::array<std::vector<std::uint8_t>, 2> const& v, std::uint64_t size)
{
        auto pref = [](std::uint8_t x) { return x >= DRAW ? 0 : (x & 1) ? 1000 - x : -1000 + x; };
        std::uint64_t errors = 0;
#pragma omp parallel for schedule(dynamic, 1 << 14) reduction(+ : errors)
        for (std::int64_t ii = 0; ii < static_cast<std::int64_t>(size); ++ii) {
                for (auto s = 0; s < 2; ++s) {
                        auto const p = decode(static_cast<std::uint64_t>(ii), ns, s);
                        auto best = DRAW; auto first = true;
                        auto const n = children(p, [&](std::uint8_t c, bool, auto&&...) {
                                auto const b = c >= DRAW ? DRAW : static_cast<std::uint8_t>(c + 1);
                                if (first || pref(b) > pref(best)) { best = b; first = false; }
                        });
                        if (n == 0) { auto q = p; q.stm ^= 1; best = children(q, [](std::uint8_t, bool, auto&&...) {}) > 0 ? std::uint8_t{0} : DRAW; }
                        errors += best != v[s][static_cast<std::uint64_t>(ii)];
                }
        }
        return errors;
}

std::string nm(int s) { return std::format("{}{}", static_cast<char>('a' + fo[s]), ro[s] + 1); }


void diagram(Pos const& p)
{
        auto lake = [](int f, int r) { return (r == 4 || r == 5) && (f == 2 || f == 3 || f == 6 || f == 7); };
        std::cout << "    +" << std::string(2 * W + 1, '-') << "+\n";
        for (auto r = H - 1; r >= 0; --r) {
                std::cout << std::format("{:3} | ", r + 1);
                for (auto f = 0; f < W; ++f) {
                        auto c = lake(f, r) ? '~' : '.';
                        for (auto sq : { p.a, p.b }) { if (fo[sq] == f && ro[sq] == r) { c = 'L'; } }
                        for (auto k = 0; k < p.ns; ++k) { auto const sq = k ? p.s2 : p.s1; if (fo[sq] == f && ro[sq] == r) { c = 'S'; } }
                        std::cout << c << ' ';
                }
                std::cout << "|\n";
        }
        std::cout << "    +" << std::string(2 * W + 1, '-') << "+\n      a b c d e f g h i j\n";
}

void pv(Pos p)
{
        std::cout << "PV:";
        for (auto ply = 0; ply < 300 && p.ns > 0; ++ply) {
                auto const target = value(p);
                auto found = false; Pos next{}; int fr = -1, tt = -1; bool cap = false;
                children(p, [&](std::uint8_t c, bool fin, Pos const& ch, int from, int to) {
                        auto const b = c >= DRAW ? DRAW : static_cast<std::uint8_t>(c + 1);
                        if (found || b != target) { return; }
                        found = true; next = ch; fr = from; tt = to; cap = fin;
                });
                if (!found) { break; }
                if (p.stm == 0) { std::cout << std::format(" {}.", ply / 2 + 1); } else if (ply == 0) { std::cout << " 1..."; }
                std::cout << ' ' << nm(fr) << (cap && p.stm == 0 ? 'x' : '-') << nm(tt);
                if (cap && p.stm == 1) { std::cout << "(sneuvelt)"; }
                p = next;
        }
        std::cout << '\n';
}

} // namespace

int main()
{
        auto lake = [](int f, int r) { return (r == 4 || r == 5) && (f == 2 || f == 3 || f == 6 || f == 7); };
        std::array<std::array<int, H>, W> id{};
        auto n = 0;
        for (auto r = 0; r < H; ++r) { for (auto f = 0; f < W; ++f) { if (lake(f, r)) { id[f][r] = -1; continue; } fo[n] = f; ro[n] = r; id[f][r] = n++; } }
        int const df[4] = { 0, 1, 0, -1 }, dr[4] = { 1, 0, -1, 0 };
        for (auto s = 0; s < N; ++s) { for (auto d = 0; d < 4; ++d) { auto const f = fo[s] + df[d], r = ro[s] + dr[d]; nb[s][d] = (f < 0 || f >= W || r < 0 || r >= H) ? -1 : id[f][r]; } }
        for (auto b = 1; b < N; ++b) { for (auto a = 0; a < b; ++a) { pairs.push_back({ a, b }); } }
        std::cout << std::format("2-1: {} placements, 2-2: {} placements\n", S21, S22);

        solve(1, v21, S21, 1, "2L-1S");
        solve(2, v22, S22, max_dtm(v21), "2L-2S");

        std::cout << std::format("verify 2L-1S: {} errors, 2L-2S: {} errors\n", verify(1, v21, S21), verify(2, v22, S22));

        // longest wins in 2-2
        for (auto s = 0; s < 2; ++s) {
                auto best = 0; std::uint64_t bi = 0;
                for (auto i = std::uint64_t{0}; i < S22; ++i) { auto const x = v22[s][i]; if (x < DRAW && x > best) { best = x; bi = i; } }
                auto const p = decode(bi, 2, s);
                std::cout << std::format("\nlangste met {} aan zet: {} plies, luitenanten {} {}, sergeanten {} {}\n",
                        s == 0 ? "rood" : "blauw", best, nm(p.a), nm(p.b), nm(p.s1), nm(p.s2));
                diagram(p);
                pv(p);
        }
}
