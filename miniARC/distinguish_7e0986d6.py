"""Find a small counterexample distinguishing two real-ARC explanations.

For task 7e0986d6, both the component-level `structural` rule and the
pixel-level `per_cell_local` shortcut fit every official train and test pair.
This script searches a small finite family of grids for a minimal input x such
that

    structural(x) != per_cell_local(x).

The intended output is defined by the independently corroborated structural
rule.  The resulting (x, structural(x)) is therefore a discriminating witness:
it preserves the structural explanation and refutes the local shortcut.

The search is deliberately transparent rather than clever.  We enumerate
small binary foreground patterns with colors 1 (reference) and 2 (nuisance),
require both colors, and order candidates by number of nonzero cells and then
by grid area.  This gives a compact witness suitable for inspection.

Run from repository root:

    python3 miniARC/distinguish_7e0986d6.py
"""

from __future__ import annotations

from itertools import combinations, product

from real_arc_generality_7e0986d6 import structural, per_cell_local

Grid = list[list[int]]


def show(g: Grid) -> str:
    chars = {0: ".", 1: "R", 2: "n"}
    return "\n".join("".join(chars[v] for v in row) for row in g)


def crop(g: Grid) -> Grid:
    pts = [(r, c) for r, row in enumerate(g) for c, v in enumerate(row) if v]
    if not pts:
        return [[0]]
    r0, r1 = min(r for r, _ in pts), max(r for r, _ in pts)
    c0, c1 = min(c for _, c in pts), max(c for _, c in pts)
    return [row[c0 : c1 + 1] for row in g[r0 : r1 + 1]]


def connected_same_color(g: Grid, color: int) -> list[list[tuple[int, int]]]:
    unseen = {(r, c) for r, row in enumerate(g) for c, v in enumerate(row) if v == color}
    out = []
    while unseen:
        seed = unseen.pop()
        todo = [seed]
        comp = [seed]
        while todo:
            r, c = todo.pop()
            for q in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)):
                if q in unseen:
                    unseen.remove(q)
                    todo.append(q)
                    comp.append(q)
        out.append(comp)
    return out


def meaningful(g: Grid) -> bool:
    """Keep witnesses that exercise the intended component distinction.

    Reference color must contain a component of at least 4 cells.  Nuisance
    must contain a multi-cell component smaller than 4; otherwise disagreement
    can be a trivial frequency/tie artefact rather than component-vs-cell.
    Also require reference to be globally more frequent so per_cell_local
    assigns roles as intended.
    """
    nr = sum(v == 1 for row in g for v in row)
    nn = sum(v == 2 for row in g for v in row)
    if not (nr > nn >= 2):
        return False
    refs = connected_same_color(g, 1)
    noise = connected_same_color(g, 2)
    return any(len(c) >= 4 for c in refs) and any(2 <= len(c) < 4 for c in noise)


def enumerate_grids(h: int, w: int, nonzero: int):
    cells = list(range(h * w))
    for occupied in combinations(cells, nonzero):
        # Assign each occupied cell R/n.  Skip monochrome quickly.
        for labels in product((1, 2), repeat=nonzero):
            if 1 not in labels or 2 not in labels:
                continue
            g = [[0] * w for _ in range(h)]
            for i, v in zip(occupied, labels):
                g[i // w][i % w] = v
            if crop(g) != g:
                continue  # canonical translation representative
            yield g


def find_witness() -> tuple[Grid, Grid, Grid] | None:
    # Search by foreground complexity first.  The smallest meaningful case has
    # >=4 reference cells and >=2 nuisance cells, hence starts at 6.
    for nonzero in range(6, 10):
        for area in range(nonzero, 17):
            for h in range(2, 5):
                if area % h:
                    continue
                w = area // h
                if not (2 <= w <= 6):
                    continue
                for g in enumerate_grids(h, w, nonzero):
                    if not meaningful(g):
                        continue
                    s = structural(g)
                    l = per_cell_local(g)
                    if s != l:
                        return g, s, l
    return None


def changed(a: Grid, b: Grid) -> list[tuple[int, int, int, int]]:
    return [
        (r, c, a[r][c], b[r][c])
        for r in range(len(a))
        for c in range(len(a[0]))
        if a[r][c] != b[r][c]
    ]


def main() -> None:
    found = find_witness()
    if found is None:
        raise SystemExit("no witness found in search bounds")
    x, ys, yl = found
    print("minimal distinguishing witness found\n")
    print("input")
    print(show(x))
    print("\nstructural output (intended witness output)")
    print(show(ys))
    print("\nper-cell local output")
    print(show(yl))
    print("\ndisagreement cells:")
    for r, c, a, b in changed(ys, yl):
        print(f"  ({r},{c}): structural={a}, local={b}")
    print("\nAdd this pair as a third demonstration:")
    print("  structural remains consistent by construction")
    print("  per_cell_local is refuted")
    print("\nThis exposes over-generalisation followed by correction at the")
    print("schema level: a pixel-independent rule that survived the official ARC")
    print("task must be refined to preserve nuisance-component identity.")


if __name__ == "__main__":
    main()
