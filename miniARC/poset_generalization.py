"""Partial-order generalization experiment for MiniARC.

No numeric score is assigned to a solution. Evidence filters hypotheses; among
survivors, a hypothesis is dominated only by a strict semantic generalization
with the same action. Incomparable undominated hypotheses remain ambiguous.

The selector order is computed extensionally over all 65,536 binary 4x4 grids.
Run with:

    python miniARC/poset_generalization.py
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import FrozenSet, Iterator, Sequence

import numpy as np

Grid = tuple[tuple[int, ...], ...]
Cell = tuple[int, int]
Object = frozenset[Cell]
N = 4
ATOMS = ("Largest", "Smallest", "Leftmost", "Rightmost")
ACTIONS = ("X", "Y", "R2")


def as_grid(a: np.ndarray) -> Grid:
    return tuple(tuple(int(x) for x in row) for row in a.tolist())


def all_grids() -> Iterator[Grid]:
    for bits in range(1 << (N * N)):
        yield tuple(tuple((bits >> (r * N + c)) & 1 for c in range(N)) for r in range(N))


def components(g: Grid) -> tuple[Object, ...]:
    unseen = {(r, c) for r in range(N) for c in range(N) if g[r][c]}
    out: list[Object] = []
    while unseen:
        seed = unseen.pop(); todo = [seed]; obj = {seed}
        while todo:
            r, c = todo.pop()
            for q in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)):
                if q in unseen:
                    unseen.remove(q); obj.add(q); todo.append(q)
        out.append(frozenset(obj))
    return tuple(out)


def bbox(o: Object) -> tuple[int, int, int, int]:
    rs = [r for r, _ in o]; cs = [c for _, c in o]
    return min(rs), min(cs), max(rs), max(cs)


def atom_ids(atom: str, os: Sequence[Object]) -> set[int]:
    if not os: return set()
    if atom == "Largest":
        v = max(map(len, os)); return {i for i,o in enumerate(os) if len(o)==v}
    if atom == "Smallest":
        v = min(map(len, os)); return {i for i,o in enumerate(os) if len(o)==v}
    if atom == "Leftmost":
        v = min(bbox(o)[1] for o in os); return {i for i,o in enumerate(os) if bbox(o)[1]==v}
    if atom == "Rightmost":
        v = max(bbox(o)[3] for o in os); return {i for i,o in enumerate(os) if bbox(o)[3]==v}
    raise ValueError(atom)


@dataclass(frozen=True, order=True)
class Selector:
    atoms: FrozenSet[str]  # empty conjunction = Object
    @property
    def name(self) -> str:
        return "Object" if not self.atoms else "&".join(sorted(self.atoms))

    def ids(self, os: Sequence[Object], cache: dict[str, set[int]] | None = None) -> set[int]:
        if not self.atoms: return set(range(len(os)))
        cache = cache or {a: atom_ids(a, os) for a in ATOMS}
        ids = set(range(len(os)))
        for atom in self.atoms: ids &= cache[atom]
        return ids

    def select(self, g: Grid) -> tuple[Object, ...]:
        os = components(g); ids = self.ids(os)
        return tuple(os[i] for i in sorted(ids))


@dataclass(frozen=True, order=True)
class Hypothesis:
    selector: Selector
    action: str
    @property
    def name(self) -> str: return f"{self.selector.name};{self.action}"


def selectors(max_conjuncts: int = 2) -> tuple[Selector, ...]:
    out = [Selector(frozenset())]
    for k in range(1, max_conjuncts + 1):
        out.extend(Selector(frozenset(xs)) for xs in combinations(ATOMS, k))
    return tuple(out)


def hypotheses() -> tuple[Hypothesis, ...]:
    return tuple(Hypothesis(s,a) for s in selectors() for a in ACTIONS)


def transform_object(o: Object, action: str) -> Object:
    r0,c0,r1,c1 = bbox(o); h,w = r1-r0+1,c1-c0+1
    local = {(r-r0,c-c0) for r,c in o}
    if action == "X": local = {(r,w-1-c) for r,c in local}
    elif action == "Y": local = {(h-1-r,c) for r,c in local}
    elif action == "R2": local = {(h-1-r,w-1-c) for r,c in local}
    else: raise ValueError(action)
    return frozenset((r0+r,c0+c) for r,c in local)


def apply(h: Hypothesis, g: Grid) -> Grid:
    a = np.asarray(g, dtype=np.uint8).copy(); chosen = h.selector.select(g)
    for o in chosen:
        for r,c in o: a[r,c] = 0
    for o in chosen:
        for r,c in transform_object(o, h.action): a[r,c] = 1
    return as_grid(a)


Example = tuple[Grid, Grid]

def version_space(evidence: Sequence[Example]) -> tuple[Hypothesis, ...]:
    return tuple(h for h in hypotheses() if all(apply(h,x)==y for x,y in evidence))


def semantic_generality() -> dict[tuple[Selector,Selector], bool]:
    """general[(a,b)] iff every (grid,object) selected by b is selected by a."""
    ss = selectors()
    general = {(a,b): True for a in ss for b in ss}
    for g in all_grids():
        os = components(g); cache = {a: atom_ids(a,os) for a in ATOMS}
        ids = {s: s.ids(os,cache) for s in ss}
        for a in ss:
            for b in ss:
                if general[(a,b)] and not ids[b] <= ids[a]: general[(a,b)] = False
    return general


def preferred(evidence: Sequence[Example], general: dict[tuple[Selector,Selector],bool]) -> tuple[Hypothesis,...]:
    vs = version_space(evidence); out=[]
    for h in vs:
        dominated = any(
            k != h and k.action == h.action
            and general[(k.selector,h.selector)] and not general[(h.selector,k.selector)]
            for k in vs
        )
        if not dominated: out.append(h)
    return tuple(out)


def hasse_edges(hs: Sequence[Hypothesis], general: dict[tuple[Selector,Selector],bool]):
    edges=[]
    for a in hs:
        for b in hs:
            if a==b or a.action!=b.action: continue
            if not general[(a.selector,b.selector)] or general[(b.selector,a.selector)]: continue
            if not any(
                c not in (a,b) and c.action==a.action
                and general[(a.selector,c.selector)] and not general[(c.selector,a.selector)]
                and general[(c.selector,b.selector)] and not general[(b.selector,c.selector)]
                for c in hs
            ): edges.append((a,b))
    return tuple(edges)


def grid(*rows: str) -> Grid:
    return tuple(tuple(int(c) for c in row) for row in rows)


def show(stage: str, evidence: Sequence[Example], general) -> None:
    vs=version_space(evidence); ps=preferred(evidence,general)
    print(f"\n{stage}: consistent={len(vs)}, undominated={len(ps)}")
    print("consistent:  " + ", ".join(h.name for h in vs))
    print("preferred:   " + ", ".join(h.name for h in ps))
    edges=hasse_edges(vs,general)
    if edges:
        print("covers:      " + ", ".join(f"{a.name}>{b.name}" for a,b in edges))


def demo() -> None:
    intended = Hypothesis(Selector(frozenset({"Largest"})), "X")
    # Found by exhaustive search. After e1, Largest;X and Leftmost;X are
    # incomparable undominated explanations. e2 breaks that ambiguity.
    xs = (
        grid("1101","1011","1000","0000"),
        grid("0111","1001","1100","0000"),
    )
    evidence = tuple((x,apply(intended,x)) for x in xs)
    print(f"selectors={len(selectors())}, hypotheses={len(hypotheses())}, universe={1<<(N*N)}")
    general=semantic_generality()
    print(f"semantic selector-order pairs={sum(general.values())}")
    show("after example 1", evidence[:1], general)
    show("after example 2", evidence, general)


if __name__ == "__main__": demo()
