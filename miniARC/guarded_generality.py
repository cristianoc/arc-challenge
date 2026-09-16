"""Generalization by weakening assumptions.

This experiment abstracts the selector-specific order in poset_generalization.py.
An explanation is a guarded partial behavior

    guard => body

and one explanation is at least as general as another when:

  1. its guard is weaker (it applies on a superset of worlds), and
  2. on every world where the more specific explanation applies, the two
     bodies predict the same result.

Everything is checked extensionally over a finite universe.  There is no score.

This subsumes examples such as

    Largest;X  >  Largest&Leftmost;X

but also permits different bodies to be related when they coincide on the
narrower's domain.

The relational-vs-constant example is included as an important negative case:
if both NextToMarker and Shift(3) claim to be unconditional rules, neither is
more general.  If Shift(3) is honestly stated only under the assumption Gap=4,
then unconditional NextToMarker dominates it.  The order therefore compares
*explanations including their assumptions*; it does not infer hidden guards in
order to make a desired answer win.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, Iterable, TypeVar

W = TypeVar("W")
Y = TypeVar("Y")


@dataclass(frozen=True)
class Explanation(Generic[W, Y]):
    name: str
    guard: Callable[[W], bool]
    body: Callable[[W], Y | None]

    def applies(self, w: W) -> bool:
        return self.guard(w) and self.body(w) is not None


def at_least_as_general(
    general: Explanation[W, Y],
    specific: Explanation[W, Y],
    worlds: Iterable[W],
) -> bool:
    """Semantic generality over a finite universe.

    Every world covered by `specific` must be covered by `general`, and the
    predictions must agree there.
    """
    for w in worlds:
        if specific.applies(w):
            if not general.applies(w):
                return False
            if general.body(w) != specific.body(w):
                return False
    return True


def strictly_more_general(
    general: Explanation[W, Y],
    specific: Explanation[W, Y],
    worlds: tuple[W, ...],
) -> bool:
    return at_least_as_general(general, specific, worlds) and not at_least_as_general(
        specific, general, worlds
    )


# ---------------------------------------------------------------------------
# Tiny relational world from relational_ambiguity.py
# ---------------------------------------------------------------------------

N = 8
Grid = tuple[int, ...]


def grid(object_pos: int, marker_pos: int) -> Grid:
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


def universe() -> tuple[Grid, ...]:
    return tuple(grid(o, m) for o in range(N) for m in range(N) if o != m)


def next_to_marker_body(g: Grid) -> Grid | None:
    return move_object(g, positions(g)[1] - 1)


def shift3_body(g: Grid) -> Grid | None:
    return move_object(g, positions(g)[0] + 3)


def always(_: Grid) -> bool:
    return True


def gap4(g: Grid) -> bool:
    o, m = positions(g)
    return m - o == 4


def demo() -> None:
    worlds = universe()

    relational = Explanation("True => NextToMarker", always, next_to_marker_body)
    constant = Explanation("True => Shift(3)", always, shift3_body)
    guarded_constant = Explanation("Gap=4 => Shift(3)", gap4, shift3_body)

    print("unconditional candidate explanations:")
    print(
        "  NextToMarker > Shift(3):",
        strictly_more_general(relational, constant, worlds),
    )
    print(
        "  Shift(3) > NextToMarker:",
        strictly_more_general(constant, relational, worlds),
    )
    print("  result: incomparable")

    print("\nwhen the constant rule states the assumption under which it was derived:")
    print(
        "  NextToMarker > (Gap=4 => Shift(3)):",
        strictly_more_general(relational, guarded_constant, worlds),
    )
    print(
        "  (Gap=4 => Shift(3)) > NextToMarker:",
        strictly_more_general(guarded_constant, relational, worlds),
    )

    # Sanity check: on Gap=4 the bodies really coincide.
    assert all(
        next_to_marker_body(w) == shift3_body(w)
        for w in worlds
        if gap4(w) and next_to_marker_body(w) is not None and shift3_body(w) is not None
    )


if __name__ == "__main__":
    demo()
