#          Copyright Rein Halbersma 2026.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

# Compare each finite tweeveldenregel limit with the unlimited game on the states in which nobody is restricted:
# red wins that only exist with the rule ("extra") and red wins that disappear with the rule ("weg").
# Usage: python3 scripts/compare.py <dump directory> <limit> [<limit> ...]
import os, sys

def red_wins(d, stm):
    return {i for i in range(len(d) // 2) if d[2 * i + stm] < 253 and (d[2 * i + stm] % 2 == (1 if stm == 0 else 0))}

directory, limits = sys.argv[1], sys.argv[2:]
print('| database | side to move | ' + ' | '.join(f'k = {k}: extra / weg' for k in limits) + ' |')
print('|---|---|' + '---|' * len(limits))
for key in ['L-S', 'L-SS', 'LS-L', 'SS-S', 'LL-S', 'LS-S']:
    base = open(os.path.join(directory, f'kinf_{key}.bin'), 'rb').read()
    for stm in (0, 1):
        b = red_wins(base, stm)
        cells = []
        for k in limits:
            path = os.path.join(directory, f'k{k}_{key}.bin')
            if not os.path.exists(path):
                cells.append(''); continue
            w = red_wins(open(path, 'rb').read(), stm)
            cells.append(f'+{len(w - b)} / −{len(b - w)}')
        print(f'| {key} | {"red" if stm == 0 else "blue"} | ' + ' | '.join(cells) + ' |')
