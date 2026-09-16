"""A tiny partial-order generalization experiment for MiniARC.

This deliberately avoids assigning numeric scores to solutions.

Hypotheses have the form

    selector ; action

where selectors are conjunctions of object predicates and actions are simple
shape-preserving transformations.  Evidence removes inconsistent hypotheses.
Among the survivors we prefer a hypothesis only when its selector is a strict
semantic generalization of another survivor's selector.  Incomparable maximal
generalizations remain ambiguous.

The semantic order is computed extensionally over *all* binary 4x4 grids, so
this file is intended as a small laboratory rather than a scalable ARC solver.

Run:

    python miniARC/poset_generalization.py

The built-in examples are generated from Largest;X and are chosen so that
Largest;X survives while conjunctions such as Largest&Leftmost;X are dominated.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import FrozenSet, Iterable, Iterator, Sequence

import numpy as np

Grid = tuple[tuple[int, ...], ...]
Cell = tuple[int, int]
Object = frozenset[Cell]

N = 4
ATOMS = ("Largest", "Smallest", "Leftmost", "Rightmost")
ACTIONS = ("X", "Y", "R2")


def as_grid(a: np.ndarray) -> Grid:
    return tuple(tuple(int(x) for x in row) for row in a.tolist())


def array(g: Grid) -> np.ndarray:
    return np.asarray(g, dtype=np.uint8)


def all_grids(n: int = N) -> Iterator[Grid]:
    """Enumerate the complete finite input universe."""
    for bits in range(1 << (n * n)):
        yield tuple(
            tuple((bits >> (r * n + c)) & 1 for c in range(n))
            for r in range(n)
        )


def components(g: Grid) -> tuple[Object, ...]:
    """Maximal 4-connected foreground components."""
    n, m = len(g), len(g[0])
    unseen = {(r, c) for r in range(n) for c in range(m) if g[r][c]}
    out: list[Object] = []
    while unseen:
        seed = unseen.pop()
        todo = [seed]
        obj = {seed}
        while todo:
            r, c = todo.pop()
            for q in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if q in unseen:
                    unseen.remove(q)
                    obj.add(q)
                    todo.append(q)
        out.append(frozenset(obj))
    return tuple(out)


def bbox(o: Object) -> tuple[int, int, int, int]:
    rs = [r for r, _ in o]
    cs = [c for _, c in o]
    return min(rs), min(cs), max(rs), max(cs)


def selected_by_atom(atom: str, objects: Sequence[Object]) -> set[int]:
    if not objects:
        return set()
    if atom == "Largest":
        v = max(map(len, objects))
        return {i for i, o in enumerate(objects) if len(o) == v}
    if atom == "Smallest":
        v = min(map(len, objects))
        return {i for i, o in enumerate(objects) if len(o) == v}
    if atom == "Leftmost":
        v = min(bbox(o)[1] for o in objects)
        return {i for i, o in enumerate(objects) if bbox(o)[1] == v}
    if atom == "Rightmost":
        v = max(bbox(o)[3] for o in objects)
        return {i for i, o in enumerate(objects) if bbox(o)[3] == v}
    raise ValueError(atom)


@dataclass(frozen=True, order=True)
class Selector:
    # Empty conjunction means Object: every foreground object.
    atoms: FrozenSet[str]

    @property
    def name(self) -> str:
        return "Object" if not self.atoms else "&".join(sorted(self.atoms))

    def select(self, g: Grid) -> tuple[Object, ...]:
        os = components(g)
        if not self.atoms:
            return os
        ids = set(range(len(os)))
        for atom in self.atoms:
            ids &= selected_by_atom(atom, os)
        return tuple(os[i] for i in sorted(ids))


@dataclass(frozen=True, order=True)
class Hypothesis:
    selector: Selector
    action: str

    @property
    def name(self) -> str:
        return f"{self.selector.name};{self.action}"


def transform_object(o: Object, action: str) -> Object:
    """Transform an object inside its original bounding box.

    X/Y/R2 preserve bounding-box dimensions, avoiding an anchoring convention
    for non-square 90-degree rotations in this first experiment.
    """
    r0, c0, r1, c1 = bbox(o)
    h, w = r1 - r0 + 1, c1 - c0 + 1
    local = {(r - r0, c - c0) for r, c in o}
    if action == "X":
        local = {(r, w - 1 - c) for r, c in local}
    elif action == "Y":
        local = {(h - 1 - r, c) for r, c in local}
    elif action == "R2":
        local = {(h - 1 - r, w - 1 - c) for r, c in local}
    else:
        raise ValueError(action)
    return frozenset((r0 + r, c0 + c) for r, c in local)


def apply(h: Hypothesis, g: Grid) -> Grid:
    a = array(g).copy()
    selected = h.selector.select(g)
    # Simultaneous semantics: select from the input, clear selected cells, then
    # write all transformed cells.
    for o in selected:
        for r, c in o:
            a[r, c] = 0
    for o in selected:
        for r, c in transform_object(o, h.action):
            a[r, c] = 1
    return as_grid(a)


def selectors(max_conjuncts: int = 2) -> tuple[Selector, ...]:
    out = [Selector(frozenset())]
    for k in range(1, max_conjuncts + 1):
        out.extend(Selector(frozenset(xs)) for xs in combinations(ATOMS, k))
    return tuple(out)


def hypotheses() -> tuple[Hypothesis, ...]:
    return tuple(Hypothesis(s, a) for s in selectors() for a in ACTIONS)


Example = tuple[Grid, Grid]


def consistent(h: Hypothesis, evidence: Sequence[Example]) -> bool:
    return all(apply(h, x) == y for x, y in evidence)


def version_space(evidence: Sequence[Example]) -> tuple[Hypothesis, ...]:
    return tuple(h for h in hypotheses() if consistent(h, evidence))


def selector_signature(s: Selector, universe: Iterable[Grid]) -> frozenset[tuple[Grid, Object]]:
    """Extension of a selector over a finite universe.

    A selector denotes the set of (grid, object) pairs it selects.  Inclusion
    therefore gives an extensional notion of selector generality.
    """
    return frozenset((g, o) for g in universe for o in s.select(g))


def semantic_generality() -> dict[tuple[Selector, Selector], bool]:
    """Return >= relation on selectors, computed over every 4x4 grid.

    general[(a,b)] means a is at least as general as b: every object selected
    by b is also selected by a.  This is deliberately semantic, not based on
    conjunction length.
    """
    ss = selectors()
    sig: dict[Selector, set[tuple[Grid, Object]]] = {s: set() for s in ss}
    # One pass through the 65,536 grids is substantially cheaper than
    # materialising the universe once per selector.
    for g in all_grids():
        for s in ss:
            sig[s].update((g, o) for o in s.select(g))
    return {(a, b): sig[b] <= sig[a] for a in ss for b in ss}


def preferred(
    evidence: Sequence[Example],
    general: dict[tuple[Selector, Selector], bool],
) -> tuple[Hypothesis, ...]:
    """Undominated consistent hypotheses.

    h dominates k iff they perform the same action and h's selector is a
    strict semantic generalization of k's selector.  No comparison is invented
    between different actions or incomparable selectors.
    """
    vs = version_space(evidence)
    out: list[Hypothesis] = []
    for h in vs:
        dominated = False
        for k in vs:
            if h == k or h.action != k.action:
                continue
            k_ge_h = general[(k.selector, h.selector)]
            h_ge_k = general[(h.selector, k.selector)]
            if k_ge_h and not h_ge_k:
                dominated = True
                break
        if not dominated:
            out.append(h)
    return tuple(out)


def hasse_edges(
    hs: Sequence[Hypothesis],
    general: dict[tuple[Selector, Selector], bool],
) -> tuple[tuple[Hypothesis, Hypothesis], ...]:
    """Cover edges general -> specific among supplied hypotheses."""
    edges: list[tuple[Hypothesis, Hypothesis]] = []
    for a in hs:
        for b in hs:
            if a == b or a.action != b.action:
                continue
            if not general[(a.selector, b.selector)] or general[(b.selector, a.selector)]:
                continue
            between = False
            for c in hs:
                if c in (a, b) or c.action != a.action:
                    continue
                a_ge_c = general[(a.selector, c.selector)] and not general[(c.selector, a.selector)]
                c_ge_b = general[(c.selector, b.selector)] and not general[(b.selector, c.selector)]
                if a_ge_c and c_ge_b:
                    between = True
                    break
            if not between:
                edges.append((a, b))
    return tuple(edges)


def grid(*rows: str) -> Grid:
    assert len(rows) == N and all(len(r) == N for r in rows)
    return tuple(tuple(int(c) for c in r) for r in rows)


def demo() -> None:
    intended = Hypothesis(Selector(frozenset({"Largest"})), "X")

    # Each input has multiple components.  The largest component changes
    # position across examples, breaking accidental Leftmost/Rightmost
    # correlations while keeping the intended selector invariant.
    xs = (
        grid("1100", "1000", "0001", "0000"),
        grid("0001", "0000", "0110", "0100"),
        grid("1000", "0000", "0011", "0001"),
    )
    evidence = tuple((x, apply(intended, x)) for x in xs)

    print(f"selectors: {len(selectors())}; hypotheses: {len(hypotheses())}")
    print("computing semantic selector order over all 4x4 binary grids ...")
    general = semantic_generality()

    vs = version_space(evidence)
    mins = preferred(evidence, general)
    print(f"consistent: {len(vs)}")
    for h in vs:
        print("  ", h.name)
    print(f"undominated: {len(mins)}")
    for h in mins:
        print("  *", h.name)

    print("cover edges among consistent hypotheses (general -> specific):")
    for a, b in hasse_edges(vs, general):
        print(f"  {a.name} -> {b.name}")


if __name__ == "__main__":
    demo()
