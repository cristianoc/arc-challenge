"""A second MiniARC+ experiment: relational vs accidental explanations.

This file does NOT add a second preference rule.  It constructs the first case
where the current semantic-specialization order intentionally has nothing to
say, even though one explanation looks much more ARC-like.

World
-----
A 1x8 row contains:
  1 : a movable object (one cell)
  2 : a marker/target (one cell)
  0 : background

Candidate explanations:
  Shift(k)       move the object k cells right
  NextToMarker   move the object immediately to the left of the marker

Training examples are chosen so that the marker is always exactly four cells
to the right of the object.  Hence Shift(3) and NextToMarker produce identical
outputs on every training example, while absolute positions vary.

On a counterfactual test where the gap changes, they disagree.

The important point is methodological: neither total behavior extends the
other, so G1 (semantic specialization by extension) cannot order them.  The
presence of a varying marker gives us a *visible structural reason* to inspect,
but this program deliberately does not turn that reason into a score or a new
order.  It reports counterfactual dependencies so that G2 can be formulated
only after we understand what is being captured.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

N = 8
Grid = tuple[int, ...]


def grid(object_pos: int, marker_pos: int) -> Grid:
    assert 0 <= object_pos < N
    assert 0 <= marker_pos < N
    assert object_pos != marker_pos
    xs = [0] * N
    xs[object_pos] = 1
    xs[marker_pos] = 2
    return tuple(xs)


def positions(g: Grid) -> tuple[int, int]:
    return g.index(1), g.index(2)


def move_object(g: Grid, dest: int) -> Grid | None:
    obj, marker = positions(g)
    if not (0 <= dest < N) or dest == marker:
        return None
    xs = list(g)
    xs[obj] = 0
    xs[dest] = 1
    return tuple(xs)


@dataclass(frozen=True)
class Hypothesis:
    name: str
    run: Callable[[Grid], Grid | None]


def shift(k: int) -> Hypothesis:
    return Hypothesis(
        f"Shift({k})",
        lambda g: move_object(g, positions(g)[0] + k),
    )


def next_to_marker() -> Hypothesis:
    return Hypothesis(
        "NextToMarker",
        lambda g: move_object(g, positions(g)[1] - 1),
    )


def universe() -> tuple[Grid, ...]:
    return tuple(grid(o, m) for o in range(N) for m in range(N) if o != m)


def extension(h: Hypothesis) -> frozenset[tuple[Grid, Grid]]:
    """Graph of the partial function over the complete finite universe."""
    out = set()
    for g in universe():
        y = h.run(g)
        if y is not None:
            out.add((g, y))
    return frozenset(out)


def strict_extension(more_general: Hypothesis, more_specific: Hypothesis) -> bool:
    a, b = extension(more_general), extension(more_specific)
    return b < a


def agrees(h: Hypothesis, examples: tuple[tuple[Grid, Grid], ...]) -> bool:
    return all(h.run(x) == y for x, y in examples)


def depends_on_marker(h: Hypothesis) -> bool:
    """Counterfactual dependence: hold object fixed, vary only marker.

    True iff changing the marker can change h's predicted destination.
    This is a diagnostic property, NOT a preference rule.
    """
    for o in range(N):
        gs = [grid(o, m) for m in range(N) if m != o]
        ys = {h.run(g) for g in gs}
        if len(ys) > 1:
            return True
    return False


def depends_on_absolute_object_position(h: Hypothesis) -> bool:
    """Diagnostic: hold object-marker displacement fixed and translate both.

    Reports whether the relative output displacement changes under a joint
    translation.  Both hypotheses in the main experiment are translation
    equivariant, so this is expected to be False.
    """
    for gap in range(1, N):
        displacements = set()
        for o in range(N - gap):
            m = o + gap
            y = h.run(grid(o, m))
            if y is not None:
                new_o, _ = positions(y)
                displacements.add(new_o - o)
        if len(displacements) > 1:
            return True
    return False


def show(g: Grid) -> str:
    return "".join("." if x == 0 else str(x) for x in g)


def demo() -> None:
    intended = next_to_marker()
    accidental = shift(3)
    hs = (intended, accidental)

    # marker - object = 4, so destination marker-1 = object+3.
    train_inputs = (grid(0, 4), grid(1, 5), grid(2, 6), grid(3, 7))
    examples = tuple((x, intended.run(x)) for x in train_inputs)
    assert all(y is not None for _, y in examples)
    examples = tuple((x, y) for x, y in examples if y is not None)

    print("training examples:")
    for x, y in examples:
        print(f"  {show(x)} -> {show(y)}")

    print("\nconsistent hypotheses:")
    for h in hs:
        print(f"  {h.name}: {agrees(h, examples)}")

    print("\nG1 semantic-extension comparisons:")
    print(f"  NextToMarker > Shift(3): {strict_extension(intended, accidental)}")
    print(f"  Shift(3) > NextToMarker: {strict_extension(accidental, intended)}")
    print("  result: incomparable")

    print("\nstructural diagnostics (not preferences):")
    for h in hs:
        print(
            f"  {h.name}: marker-dependent={depends_on_marker(h)}, "
            f"absolute-position-dependent={depends_on_absolute_object_position(h)}"
        )

    test = grid(1, 7)  # gap 6 rather than 4
    print("\ncounterfactual test with changed object-marker gap:")
    print(f"  input:        {show(test)}")
    for h in hs:
        y = h.run(test)
        print(f"  {h.name:12}: {show(y) if y is not None else 'undefined'}")


if __name__ == "__main__":
    demo()
