#          Copyright Rein Halbersma 2026.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

# Red wins with red to move, restricted to positions in which nobody is restricted and red cannot attack at once
# (no red piece orthogonally next to a blue piece). Reads the dumps dump/k<limit>_<key>.bin written by stratego_egdb:
# one byte per (placement, side to move) for the states in which nobody is restricted.
# Usage: python3 scripts/no_capture.py <dump directory> <limit> [<limit> ...]   e.g. dump 1 2 3 4 5 inf
import os, sys

W, H = 10, 10
lake = lambda f, r: r in (4, 5) and f in (2, 3, 6, 7)
squares = [(f, r) for r in range(H) for f in range(W) if not lake(f, r)]
N = len(squares)
index = {fr: i for i, fr in enumerate(squares)}
adj = [set() for _ in range(N)]
for i, (f, r) in enumerate(squares):
    for df, dr in ((0, 1), (1, 0), (0, -1), (-1, 0)):
        j = index.get((f + df, r + dr))
        if j is not None:
            adj[i].add(j)
pairs = [(a, b) for b in range(1, N) for a in range(b)]
binom = lambda n, k: 1 if k == 0 else n if k == 1 else n * (n - 1) // 2

def placements(key):
    """Yield (red squares, blue squares) in index order, mirroring encode()/decode() of stratego_egdb."""
    red, blue = key.split('-')
    groups, free = [], N
    for side, v in ((0, red), (1, blue)):
        k = 0
        while k < len(v):
            j = k
            while j < len(v) and v[j] == v[k]:
                j += 1
            groups.append((side, j - k, binom(free, j - k)))
            free -= j - k
            k = j
    total = 1
    for g in groups:
        total *= g[2]
    for idx in range(total):
        digits, x = [], idx
        for g in reversed(groups):
            digits.append(x % g[2]); x //= g[2]
        digits.reverse()
        used, sq = [], ([], [])
        for (side, count, _), d in zip(groups, digits):
            cs = [d] if count == 1 else list(pairs[d])
            placed = []
            for c in cs:
                for u in sorted(used):
                    if u <= c:
                        c += 1
                placed.append(c)
            used += placed
            sq[side].extend(placed)
        yield sq

keys = ['L-S', 'L-SS', 'LS-L', 'SS-S', 'LL-S', 'LS-S']
directory, limits = sys.argv[1], sys.argv[2:]
quiet = {key: [not any(adj[r] & set(b) for r in rs) for rs, b in placements(key)] for key in keys}
print('| database | ' + ' | '.join('∞' if k == 'inf' else f'k = {k}' for k in limits) + ' |')
print('|---|' + '---|' * len(limits))
for key in keys:
    cells = []
    for k in limits:
        path = os.path.join(directory, f'k{k}_{key}.bin')
        if not os.path.exists(path):
            cells.append('')
            continue
        d = open(path, 'rb').read()
        n = wins = longest = 0
        for i, q in enumerate(quiet[key]):
            if q:
                v = d[2 * i]
                n += 1
                if v < 253 and v % 2 == 1:
                    wins += 1
                    longest = max(longest, v)
        cells.append(f'{longest or "—"} ({100 * wins / n:.1f}%)')
    print(f'| {key} | ' + ' | '.join(cells) + ' |')
