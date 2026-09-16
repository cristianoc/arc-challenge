"""Representations as closure operators on ground explanations.

This is the next abstraction above `representation_generality.py`.

Let G be a finite set of ground explanatory instances.  A representation R
provides a family of concepts (sets of ground instances).  Given observed
instances S, its generalization is the least representable concept containing S:

    cl_R(S) = intersection { C in R | S subset C }.

When the concept family is a Moore family (contains G and is closed under
intersection), cl_R is a closure operator: extensive, monotone, idempotent.

The experiment compares three representations of the same ground universe:

* pixel: no cross-size concept, so observed FlipX instances stay unrelated;
* object: adds the parametric concept FlipX(any size), so examples at sizes
  2,3,4 close to the unseen size-5 instance;
* indiscriminate: closes any nonempty evidence to every ground transformation.
  It generalizes more, but incorrectly includes FlipY.  This demonstrates that
  'larger closure' means more generalizing, not better.

Run:
    python miniARC/representation_closure.py
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import chain, combinations

Ground = str
Concept = frozenset[Ground]

FLIP_X = tuple(f"FlipX({n})" for n in (2, 3, 4, 5))
FLIP_Y = tuple(f"FlipY({n})" for n in (2, 3, 4, 5))
G: Concept = frozenset(FLIP_X + FLIP_Y)


def powerset(xs: tuple[Ground, ...]) -> set[Concept]:
    return {
        frozenset(c)
        for k in range(len(xs) + 1)
        for c in combinations(xs, k)
    }


@dataclass(frozen=True)
class Representation:
    name: str
    concepts: frozenset[Concept]

    def close(self, observed: Concept) -> Concept:
        supersets = [c for c in self.concepts if observed <= c]
        if not supersets:
            raise ValueError(f"{self.name}: no concept contains {sorted(observed)}")
        return frozenset.intersection(*supersets)

    def check_moore(self) -> None:
        assert G in self.concepts
        cs = tuple(self.concepts)
        for a in cs:
            for b in cs:
                assert a & b in self.concepts, (self.name, a, b, a & b)

    def check_closure_axioms(self) -> None:
        """Exhaustive check over P(G): extensive, monotone, idempotent."""
        all_sets = tuple(powerset(tuple(G)))
        for s in all_sets:
            cs = self.close(s)
            assert s <= cs
            assert self.close(cs) == cs
        for a in all_sets:
            for b in all_sets:
                if a <= b:
                    assert self.close(a) <= self.close(b)


def moore_closure(generators: set[Concept]) -> frozenset[Concept]:
    """Smallest intersection-closed family containing generators and G."""
    family = set(generators) | {G}
    changed = True
    while changed:
        changed = False
        for a in tuple(family):
            for b in tuple(family):
                c = a & b
                if c not in family:
                    family.add(c)
                    changed = True
    return frozenset(family)


def pixel_representation() -> Representation:
    """Ground instances are representable, but no cross-size FlipX schema is."""
    # Every set of ground instances is a concept, so each observation is its own
    # least representable superset and cl(S) = S: the instances stay unrelated.
    # Singletons alone would not do, since a Moore family is closed only under
    # intersection, leaving G as the sole superset of a multi-instance set.
    return Representation("pixel", frozenset(powerset(tuple(G))))


def object_representation() -> Representation:
    """Adds parametric FlipX and FlipY concepts across all sizes."""
    gens = {frozenset({g}) for g in G}
    gens |= {frozenset(FLIP_X), frozenset(FLIP_Y)}
    return Representation("object", moore_closure(gens))


def indiscriminate_representation() -> Representation:
    """Only empty/top concepts: any nonempty observation closes to all of G."""
    return Representation("indiscriminate", frozenset({frozenset(), G}))


def pointwise_leq(a: Representation, b: Representation) -> bool:
    """a <= b means b generalizes at least as far as a on every evidence set."""
    return all(a.close(s) <= b.close(s) for s in powerset(tuple(G)))


def classify(closure: Concept, target_family: Concept) -> str:
    if closure == target_family:
        return "exact"
    if closure < target_family:
        return "undergeneralizes"
    if target_family < closure:
        return "overgeneralizes"
    return "mixed/incomparable"


def demo() -> None:
    pixel = pixel_representation()
    obj = object_representation()
    bad = indiscriminate_representation()
    reps = (pixel, obj, bad)

    for r in reps:
        r.check_moore()
        r.check_closure_axioms()

    observed = frozenset(FLIP_X[:3])  # sizes 2,3,4
    target = frozenset(FLIP_X)        # intended parametric family, includes size 5

    print("ground universe:")
    print(" ", sorted(G))
    print("observed instances:")
    print(" ", sorted(observed))
    print("intended family:")
    print(" ", sorted(target))

    print("\nclosures:")
    for r in reps:
        c = r.close(observed)
        print(f"  {r.name:14} -> {sorted(c)}")
        print(f"                   {classify(c, target)}")

    cp = pixel.close(observed)
    co = obj.close(observed)
    cb = bad.close(observed)
    assert cp == observed
    assert co == target
    assert cb == G
    assert cp < co < cb

    print("\nfor this evidence:")
    print("  pixel closure < object closure < indiscriminate closure")
    print("  but only the middle closure equals the intended family")

    print("\npointwise closure-order between representations:")
    for a in reps:
        for b in reps:
            if a is not b and pointwise_leq(a, b):
                print(f"  {a.name} <= {b.name}")

    print("\nInterpretation:")
    print("  A representation induces a closure operator, hence a principled")
    print("  generalization operation.  More closure is not automatically better:")
    print("  the indiscriminate representation is maximally aggressive and wrong.")
    print("  Relative to a known target family T, the desirable condition is")
    print("  cl_R(E) = T; subset is undergeneralization, superset overgeneralization.")


if __name__ == "__main__":
    demo()
