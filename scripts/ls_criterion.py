#          Copyright Rein Halbersma 2026.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

# Check the closed-form criterion for L vs S (states in which nobody is restricted) against the databases:
# red wins  <=>  (1) L has the opposition: red to move and odd distance, or blue to move and even distance, and
#                (2) for no square z of the 12-square rings around the lakes does S arrive at least 5 plies before L
#                    (the side to move reaches a square at distance d after 2d - 1 plies, the other side after 2d).
# Usage: python3 scripts/ls_criterion.py <dump directory> <limit> [<limit> ...]
import collections, os, sys

W, H = 10, 10
lake = lambda f, r: r in (4, 5) and f in (2, 3, 6, 7)
squares = [(f, r) for r in range(H) for f in range(W) if not lake(f, r)]
N = len(squares)
index = {fr: i for i, fr in enumerate(squares)}
adj = [[index[(f + a, r + b)] for a, b in ((0, 1), (1, 0), (0, -1), (-1, 0)) if (f + a, r + b) in index] for f, r in squares]

def bfs(s):
    d = [None] * N
    d[s] = 0
    q = collections.deque([s])
    while q:
        x = q.popleft()
        for y in adj[x]:
            if d[y] is None:
                d[y] = d[x] + 1
                q.append(y)
    return d

D = [bfs(s) for s in range(N)]
ring = [i for i, (f, r) in enumerate(squares) if any(lake(f + a, r + b) for a in (-1, 0, 1) for b in (-1, 0, 1))]

def criterion(l, s, stm):
    if D[l][s] % 2 != (1 if stm == 0 else 0):
        return False
    for z in ring:
        arrive_l = 2 * D[l][z] - (1 if stm == 0 else 0)
        arrive_s = 2 * D[s][z] - (1 if stm == 1 else 0)
        if arrive_l - arrive_s >= 5:
            return False
    return True

directory, limits = sys.argv[1], sys.argv[2:]
for k in limits:
    d = open(os.path.join(directory, f'k{k}_L-S.bin'), 'rb').read()
    for stm in (0, 1):
        wrong = 0
        for idx in range(92 * 91):
            l, c = divmod(idx, 91)
            s = c + (1 if c >= l else 0)
            v = d[2 * idx + stm]
            red_wins = v < 253 and v % 2 == (1 if stm == 0 else 0)
            wrong += red_wins != criterion(l, s, stm)
        print(f'k = {k}, {"red" if stm == 0 else "blue"} to move: {wrong} deviations out of {92 * 91}')
