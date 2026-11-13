#          Copyright Rein Halbersma 2026.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

# Summarise results/k*.log into one Markdown table per database: for each tweeveldenregel limit the share of red
# wins over the states in which nobody is restricted (red / blue to move), the longest win over all states, and the
# longest red win with red to move from such a state.
# Usage: python3 scripts/table.py results/k1.log results/k2.log ... (the limit is read from the first line)
# The first table has one row per database and one column per limit: longest win (red wins % red to move / blue to move).
import re, sys

data, order, limits = {}, [], []
for path in sys.argv[1:]:
    text = open(path).read()
    first = text.splitlines()[0]
    k = 'inf' if 'onbeperkt' in first else re.search(r'max (\d+)', first).group(1)
    limits.append(k)
    for m in re.finditer(r'^(\S+-\S+)\s+\d+ toestanden.*?langste\s+(\d+) plies.*?\n(.*)\n(.*)\n', text, re.M):
        key, longest, rtm, btm = m.groups()
        if key not in order:
            order.append(key)
        pct = lambda line: float(re.search(r'niemand beperkt: rood\s+([\d.]+)%', line).group(1))
        lw = re.search(r'langste rode winst (\d+) plies', rtm)
        data[key, k] = (pct(rtm), pct(btm), int(longest), int(lw.group(1)) if lw else 0)

head = ['k = ' + k if k != 'inf' else '∞' for k in limits]
print('| database | ' + ' | '.join(head) + ' |')
print('|---|' + '---|' * len(limits))
for key in order:
    cells = []
    for k in limits:
        if (key, k) in data:
            a, b, l, w = data[key, k]
            cells.append(f'{l} ({a:.1f} / {b:.1f})')
        else:
            cells.append('')
    print(f'| {key} | ' + ' | '.join(cells) + ' |')

for key in order:
    print(f'\n**{key}**\n')
    print('| limit | red wins, red to move | red wins, blue to move | longest win (plies) | longest red win, red to move |')
    print('|---|---|---|---|---|')
    for k in limits:
        if (key, k) in data:
            a, b, l, w = data[key, k]
            print(f'| {"∞" if k == "inf" else k} | {a:.1f}% | {b:.1f}% | {l} | {w or "—"} |')
