"""End-to-end toy representation learning by anti-unification.

Unlike the previous experiments, the useful parametric abstraction is not
pre-installed.  The learner starts with ground terms, anti-unifies them, and
uses the resulting term to induce a closure over a finite ground universe.

Then a counterexample demonstrates the limit: over-general anti-unification can
abstract a dimension that actually matters.  A split/refinement operation
restores the relevant distinction while retaining the useful size parameter.

No numeric ranking is used.

Run:
    python miniARC/antiunify_learning.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Union

SIZES = (2, 3, 4, 5)
ACTIONS = ("X", "Y")


@dataclass(frozen=True, order=True)
class Var:
    name: str

    def __str__(self) -> str:
        return self.name


Term = Union[str, int, Var, tuple]


def pretty(t: Term) -> str:
    if isinstance(t, Var):
        return t.name
    if isinstance(t, tuple):
        head, *args = t
        return f"{head}(" + ",".join(pretty(a) for a in args) + ")"
    return str(t)


def ground(action: str, size: int) -> Term:
    return ("Flip", action, size)


GROUND = frozenset(ground(a, n) for a in ACTIONS for n in SIZES)


class Fresh:
    def __init__(self) -> None:
        self.i = 0

    def var(self) -> Var:
        self.i += 1
        return Var(f"v{self.i}")


def anti_unify_pair(a: Term, b: Term, fresh: Fresh, memo: dict[tuple[Term, Term], Var]) -> Term:
    """First-order syntactic anti-unification with disagreement sharing.

    Repeated identical disagreement pairs receive the same variable.  This is
    enough for the finite examples here and exposes the substitution structure
    explicitly.
    """
    if a == b:
        return a
    if isinstance(a, tuple) and isinstance(b, tuple) and len(a) == len(b) and a[0] == b[0]:
        return tuple([a[0]] + [anti_unify_pair(x, y, fresh, memo) for x, y in zip(a[1:], b[1:])])
    key = (a, b)
    rev = (b, a)
    if key in memo:
        return memo[key]
    if rev in memo:
        return memo[rev]
    v = fresh.var()
    memo[key] = v
    return v


def anti_unify(terms: Iterable[Term]) -> Term:
    xs = list(terms)
    assert xs
    out = xs[0]
    fresh = Fresh()
    for x in xs[1:]:
        out = anti_unify_pair(out, x, fresh, {})
    return out


def matches(pattern: Term, value: Term, env: dict[Var, Term] | None = None) -> bool:
    if env is None:
        env = {}
    if isinstance(pattern, Var):
        old = env.get(pattern)
        if old is None:
            env[pattern] = value
            return True
        return old == value
    if isinstance(pattern, tuple):
        return (
            isinstance(value, tuple)
            and len(pattern) == len(value)
            and all(matches(p, v, env) for p, v in zip(pattern, value))
        )
    return pattern == value


def closure(pattern: Term) -> frozenset[Term]:
    return frozenset(g for g in GROUND if matches(pattern, g))


def show(xs: Iterable[Term]) -> str:
    return "{" + ", ".join(sorted(pretty(x) for x in xs)) + "}"


def learn_pattern(evidence: frozenset[Term]) -> Term:
    return anti_unify(sorted(evidence, key=pretty))


def split_on_argument(pattern: Term, argument_index: int, values: Iterable[Term]) -> tuple[Term, ...]:
    """Refine a tuple pattern by reinstating a relevant argument distinction."""
    assert isinstance(pattern, tuple)
    out = []
    for value in values:
        xs = list(pattern)
        xs[argument_index] = value
        out.append(tuple(xs))
    return tuple(out)


def experiment_size_abstraction() -> Term:
    evidence = frozenset({ground("X", 2), ground("X", 3), ground("X", 4)})
    p = learn_pattern(evidence)
    c = closure(p)
    target = frozenset(ground("X", n) for n in SIZES)

    print("EXPERIMENT 1: construct size abstraction\n")
    print("ground evidence:", show(evidence))
    print("anti-unifier:   ", pretty(p))
    print("induced closure:", show(c))
    print("target:         ", show(target))
    print("exact:          ", c == target)
    print()

    assert isinstance(p, tuple)
    assert p[0] == "Flip" and p[1] == "X" and isinstance(p[2], Var)
    assert c == target
    return p


def experiment_overgeneralize_then_split() -> None:
    # Symmetric situations can make X/Y ground explanations observationally
    # indistinguishable.  If we anti-unify across both action and size, we get
    # Flip(action,size), whose closure is the whole universe.
    evidence = frozenset({
        ground("X", 2),
        ground("Y", 2),
        ground("X", 3),
        ground("Y", 3),
        ground("X", 4),
        ground("Y", 4),
    })
    p = learn_pattern(evidence)
    c = closure(p)

    print("EXPERIMENT 2: over-general anti-unification\n")
    print("ground evidence:", show(evidence))
    print("anti-unifier:   ", pretty(p))
    print("induced closure:", show(c))
    print()

    assert isinstance(p, tuple)
    assert isinstance(p[1], Var) and isinstance(p[2], Var)
    assert c == GROUND

    # An asymmetric witness establishes that action is semantically relevant.
    # We model the repair structurally: split the action variable while leaving
    # the size variable free.  This creates two parametric concepts rather than
    # reverting to ground cases.
    refined = split_on_argument(p, 1, ACTIONS)
    print("counterexample says: action distinction matters")
    print("split concepts:")
    for q in refined:
        print(" ", pretty(q), "=>", show(closure(q)))

    x_pattern = next(q for q in refined if isinstance(q, tuple) and q[1] == "X")
    x_target = frozenset(ground("X", n) for n in SIZES)
    assert closure(x_pattern) == x_target
    print("\nsize abstraction retained after split:", pretty(x_pattern))
    print("X closure exact:", closure(x_pattern) == x_target)


def demo() -> None:
    experiment_size_abstraction()
    experiment_overgeneralize_then_split()


if __name__ == "__main__":
    demo()
