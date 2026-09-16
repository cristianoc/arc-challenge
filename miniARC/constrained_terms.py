"""Constrained terms: a minimal order-theoretic model of generalization.

An explanation is a pair (term, constraint).

    term        parametric explanatory schema
    constraint  finite set of parameter valuations on which it claims to apply

Within one term, weaker constraints are more general.  In this finite model a
constraint is represented extensionally as a set of valuations, so weakening
is simply superset inclusion.

    (t, Phi1) >= (t, Phi2)  iff  [[Phi1]] superset [[Phi2]]

Different terms are deliberately incomparable.  Evidence only filters
explanations: a candidate must cover every observed valuation and produce every
observed output.  Preferred explanations are the undominated survivors.

The file exercises three phenomena:

1. predicate weakening: Largest&Leftmost -> Largest;
2. parameter freeing: FlipX(size in {2,3,4}) -> FlipX(any size);
3. ambiguity across terms: FlipX(any size) || FlipY(any size).

It then shows the important limitation: the maximally weak constraint `True`
is preferred whenever the term fits the observations, even though finite data
do not justify extrapolating the term.  Thus the order is a clean account of
*relative generality once a parametric term is admitted*, not a complete theory
of which terms may be inferred from evidence.

Run:
    python miniARC/constrained_terms.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Hashable, Iterable

Valuation = Hashable
Output = Hashable


@dataclass(frozen=True)
class Term:
    name: str
    run: Callable[[Valuation], Output]

    # Identity is semantic/schema identity for this tiny experiment.  Callable
    # fields are excluded from comparison by giving Term an explicit equality
    # key through name below.
    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Term) and self.name == other.name


@dataclass(frozen=True)
class Explanation:
    term: Term
    domain: frozenset[Valuation]

    @property
    def name(self) -> str:
        return f"{self.term.name} @ {format_domain(self.domain)}"


@dataclass(frozen=True)
class Problem:
    universe: frozenset[Valuation]
    evidence: tuple[tuple[Valuation, Output], ...]


def format_domain(xs: frozenset[Valuation]) -> str:
    if not xs:
        return "False"
    return "{" + ", ".join(sorted(map(str, xs))) + "}"


def covers(problem: Problem, e: Explanation) -> bool:
    return all(v in e.domain and e.term.run(v) == y for v, y in problem.evidence)


def at_least_as_general(a: Explanation, b: Explanation) -> bool:
    """Constraint weakening within the same explanatory term."""
    return a.term == b.term and a.domain >= b.domain


def strictly_more_general(a: Explanation, b: Explanation) -> bool:
    return at_least_as_general(a, b) and not at_least_as_general(b, a)


def preferred(problem: Problem, candidates: Iterable[Explanation]) -> tuple[Explanation, ...]:
    vs = tuple(e for e in candidates if covers(problem, e))
    return tuple(e for e in vs if not any(strictly_more_general(k, e) for k in vs))


def powerset_nonempty(xs: tuple[Valuation, ...]) -> tuple[frozenset[Valuation], ...]:
    out: list[frozenset[Valuation]] = []
    n = len(xs)
    for mask in range(1, 1 << n):
        out.append(frozenset(xs[i] for i in range(n) if mask & (1 << i)))
    return tuple(out)


def all_constraints(term: Term, universe: tuple[Valuation, ...]) -> tuple[Explanation, ...]:
    return tuple(Explanation(term, d) for d in powerset_nonempty(universe))


def experiment_predicate_weakening() -> None:
    """Model selector predicates extensionally as sets of object situations."""
    # Situations record which atomic predicates an object satisfies.
    worlds: tuple[Valuation, ...] = (
        frozenset({"Largest"}),
        frozenset({"Leftmost"}),
        frozenset({"Largest", "Leftmost"}),
        frozenset(),
    )
    term = Term("FlipX(selected-object)", lambda w: "X")

    largest_domain = frozenset(w for w in worlds if "Largest" in w)
    both_domain = frozenset(w for w in worlds if {"Largest", "Leftmost"} <= w)
    largest = Explanation(term, largest_domain)
    both = Explanation(term, both_domain)

    # Observe only the intersection case: both explanations fit.
    observed = frozenset({"Largest", "Leftmost"})
    p = Problem(frozenset(worlds), ((observed, "X"),))
    best = preferred(p, (largest, both))

    print("EXPERIMENT 1: predicate weakening")
    print("  Largest >= Largest&Leftmost:", strictly_more_general(largest, both))
    print("  preferred:", [e.name for e in best])
    assert best == (largest,)


def experiment_parameter_freeing() -> None:
    sizes: tuple[Valuation, ...] = (2, 3, 4, 5)
    flip_x = Term("FlipX(size)", lambda n: f"flip-x-{n}")
    candidates = all_constraints(flip_x, sizes)
    p = Problem(
        frozenset(sizes),
        tuple((n, f"flip-x-{n}") for n in (2, 3, 4)),
    )
    best = preferred(p, candidates)

    print("\nEXPERIMENT 2: parameter freeing")
    print("  observed valuations: {2,3,4}")
    print("  preferred:", [e.name for e in best])
    assert len(best) == 1 and best[0].domain == frozenset(sizes)
    print("  note: the order prefers the unconstrained instance including unseen 5")


def experiment_incomparable_terms() -> None:
    sizes: tuple[Valuation, ...] = (2, 3, 4, 5)
    # The observed output is deliberately symmetric/identity-like so both term
    # schemas agree on the one observed valuation.
    tx = Term("FlipX(size)", lambda n: "same" if n == 4 else f"x-{n}")
    ty = Term("FlipY(size)", lambda n: "same" if n == 4 else f"y-{n}")
    candidates = all_constraints(tx, sizes) + all_constraints(ty, sizes)
    p = Problem(frozenset(sizes), ((4, "same"),))
    best = preferred(p, candidates)

    print("\nEXPERIMENT 3: incomparable terms")
    print("  preferred:", [e.name for e in best])
    assert len(best) == 2
    assert {e.term.name for e in best} == {"FlipX(size)", "FlipY(size)"}
    assert all(e.domain == frozenset(sizes) for e in best)
    print("  result: maximal constraint weakening does not order different terms")


def experiment_limitation() -> None:
    """Expose, rather than hide, the remaining induction problem."""
    universe: tuple[Valuation, ...] = (0, 1, 2, 3, 4)
    # Two terms agree on observed 0,1,2 but extrapolate differently.
    linear = Term("n+1", lambda n: int(n) + 1)
    lookupish = Term("observed-then-zero", lambda n: int(n) + 1 if int(n) <= 2 else 0)
    candidates = all_constraints(linear, universe) + all_constraints(lookupish, universe)
    p = Problem(
        frozenset(universe),
        tuple((n, n + 1) for n in (0, 1, 2)),
    )
    best = preferred(p, candidates)

    print("\nEXPERIMENT 4: the remaining induction problem")
    print("  both terms fit observations 0,1,2")
    print("  preferred:", [e.name for e in best])
    print("  predictions at 3:", {e.term.name: e.term.run(3) for e in best})
    assert len(best) == 2
    print("  result: constraint weakening alone correctly leaves the terms incomparable")
    print("  but it promotes each admitted term to the weakest constraint (full universe).")
    print("  Therefore term admission / term generalization is the next unresolved layer.")


def demo() -> None:
    experiment_predicate_weakening()
    experiment_parameter_freeing()
    experiment_incomparable_terms()
    experiment_limitation()


if __name__ == "__main__":
    demo()
