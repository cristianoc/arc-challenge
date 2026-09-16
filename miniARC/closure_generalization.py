"""Generalization as certified closure of finite evidence.

No scores.  An explanation carries extension operations that it can justify
structurally.  Starting from observed input/output pairs, we close the evidence
under those operations.  Explanations are compared by inclusion of the
resulting prediction sets.

The first finite experiment uses 1-D scenes with a movable object (1) and a
marker (2).  It checks three things:

1. translation equivariance extends one observation to its translation orbit;
2. adding an independently certified gap-change operation strictly enlarges
   the closure of NextToMarker but is *not* valid for Shift(3);
3. translation alone cannot distinguish NextToMarker from Shift(3), preventing
   the closure construction from manufacturing the desired answer.

Run:
    python miniARC/closure_generalization.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

N = 8
Grid = tuple[int, ...]
Example = tuple[Grid, Grid]


def grid(object_pos: int, marker_pos: int) -> Grid:
    assert 0 <= object_pos < N and 0 <= marker_pos < N and object_pos != marker_pos
    xs = [0] * N
    xs[object_pos] = 1
    xs[marker_pos] = 2
    return tuple(xs)


def positions(g: Grid) -> tuple[int, int]:
    return g.index(1), g.index(2)


def move_object(g: Grid, dest: int) -> Grid | None:
    o, m = positions(g)
    if not 0 <= dest < N or dest == m:
        return None
    xs = list(g)
    xs[o] = 0
    xs[dest] = 1
    return tuple(xs)


@dataclass(frozen=True)
class Hypothesis:
    name: str
    run: Callable[[Grid], Grid | None]


def next_to_marker() -> Hypothesis:
    return Hypothesis("NextToMarker", lambda g: move_object(g, positions(g)[1] - 1))


def shift(k: int) -> Hypothesis:
    return Hypothesis(f"Shift({k})", lambda g: move_object(g, positions(g)[0] + k))


def universe() -> tuple[Grid, ...]:
    return tuple(grid(o, m) for o in range(N) for m in range(N) if o != m)


def translate(g: Grid, d: int) -> Grid | None:
    o, m = positions(g)
    o2, m2 = o + d, m + d
    if 0 <= o2 < N and 0 <= m2 < N:
        return grid(o2, m2)
    return None


def move_marker(g: Grid, d: int) -> Grid | None:
    """Change only the marker position; this is NOT automatically licensed."""
    o, m = positions(g)
    m2 = m + d
    if 0 <= m2 < N and m2 != o:
        return grid(o, m2)
    return None


@dataclass(frozen=True)
class Operation:
    name: str
    on_input: Callable[[Grid], Grid | None]
    on_output: Callable[[Grid], Grid | None]


def translations() -> tuple[Operation, ...]:
    return tuple(
        Operation(
            f"Translate({d:+d})",
            lambda g, d=d: translate(g, d),
            lambda g, d=d: translate(g, d),
        )
        for d in range(-(N - 1), N)
        if d != 0
    )


def marker_variations() -> tuple[Operation, ...]:
    """Candidate operations: vary marker in input, and move output object with it.

    If the input marker moves by d, a NextToMarker output should move its object
    by d while keeping the marker at its new location.  The output transform is
    defined explicitly below because simply moving the marker is insufficient.
    """

    def output_change(g: Grid, d: int) -> Grid | None:
        o, m = positions(g)
        m2, o2 = m + d, o + d
        if not (0 <= m2 < N and 0 <= o2 < N) or m2 == o2:
            return None
        return grid(o2, m2)

    return tuple(
        Operation(
            f"VaryMarker({d:+d})",
            lambda g, d=d: move_marker(g, d),
            lambda g, d=d: output_change(g, d),
        )
        for d in range(-(N - 1), N)
        if d != 0
    )


def commutes(h: Hypothesis, op: Operation) -> bool:
    """Check h(op(x)) = op(h(x)) everywhere both sides are meaningful.

    We require equality whenever op(x), h(x), and both composed results exist.
    Cases where the operation cannot be applied are outside its finite domain.
    """
    compared = False
    for x in universe():
        ox = op.on_input(x)
        hx = h.run(x)
        if ox is None or hx is None:
            continue
        lhs = h.run(ox)
        rhs = op.on_output(hx)
        if lhs is None or rhs is None:
            continue
        compared = True
        if lhs != rhs:
            return False
    return compared


def certified(h: Hypothesis, candidates: Iterable[Operation]) -> tuple[Operation, ...]:
    return tuple(op for op in candidates if commutes(h, op))


def close(seed: Iterable[Example], operations: Iterable[Operation]) -> frozenset[Example]:
    """Least finite closure under operations, rejecting inconsistent collisions."""
    known: dict[Grid, Grid] = dict(seed)
    todo = list(seed)
    ops = tuple(operations)
    while todo:
        x, y = todo.pop()
        for op in ops:
            x2, y2 = op.on_input(x), op.on_output(y)
            if x2 is None or y2 is None:
                continue
            old = known.get(x2)
            if old is not None:
                if old != y2:
                    raise ValueError(f"inconsistent closure at {x2}: {old} vs {y2}")
                continue
            known[x2] = y2
            todo.append((x2, y2))
    return frozenset(known.items())


def valid_predictions(h: Hypothesis, facts: Iterable[Example]) -> bool:
    return all(h.run(x) == y for x, y in facts)


def show(g: Grid) -> str:
    return "".join("." if x == 0 else str(x) for x in g)


def describe(name: str, facts: frozenset[Example]) -> None:
    print(f"{name}: {len(facts)} predictions")
    for x, y in sorted(facts, key=lambda p: positions(p[0])):
        print(f"  {show(x)} -> {show(y)}")


def demo() -> None:
    relational = next_to_marker()
    constant = shift(3)

    seed_x = grid(0, 4)
    seed_y = relational.run(seed_x)
    assert seed_y is not None and constant.run(seed_x) == seed_y
    seed = frozenset({(seed_x, seed_y)})

    trans = translations()
    marker = marker_variations()

    rel_t = certified(relational, trans)
    con_t = certified(constant, trans)
    rel_m = certified(relational, marker)
    con_m = certified(constant, marker)

    print("certificates computed extensionally over the complete finite universe:\n")
    print("NextToMarker translations:", [op.name for op in rel_t])
    print("Shift(3) translations:     ", [op.name for op in con_t])
    print("NextToMarker marker-var:  ", [op.name for op in rel_m])
    print("Shift(3) marker-var:       ", [op.name for op in con_m])

    # Translation is independently valid for both explanations.
    c_rel_t = close(seed, rel_t)
    c_con_t = close(seed, con_t)
    assert c_rel_t == c_con_t
    assert valid_predictions(relational, c_rel_t)
    assert valid_predictions(constant, c_con_t)

    print("\nclosure from one observation using translation certificates:")
    describe("NextToMarker", c_rel_t)
    print("Shift(3) has the identical closure.")
    print("=> translation gives no preference, as required.\n")

    # Now include every operation actually certified by each explanation.
    c_rel_all = close(seed, rel_t + rel_m)
    c_con_all = close(seed, con_t + con_m)
    assert valid_predictions(relational, c_rel_all)
    assert valid_predictions(constant, c_con_all)

    describe("NextToMarker with all certificates", c_rel_all)
    describe("Shift(3) with all certificates", c_con_all)

    print("\nclosure comparison:")
    print("  Shift closure subset of relational:", c_con_all <= c_rel_all)
    print("  strict:                            ", c_con_all < c_rel_all)
    print("\nImportant: this dominance is justified only if the marker-variation")
    print("operations are accepted as legitimate structural morphisms of the task.")
    print("Computing that an explanation commutes with a proposed operation does not")
    print("by itself justify proposing that operation.  The experiment separates")
    print("certificate checking from the source of admissible variations.")


if __name__ == "__main__":
    demo()
