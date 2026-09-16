"""Test curriculum/order effects on an ARC-derived refinement sequence.

Uses three demonstrations:
  A, B : the two official training pairs of ARC task 7e0986d6
  W    : the six-cell distinguishing witness

        RRR       RRR
        Rnn  -->  RRR

The learner is intentionally incremental: it does NOT recompute a batch version
space after every example.  Its state records which abstractions have been
constructed/refuted.

We model three independently discoverable commitments:

  color_parametric
      fixed color identities have been refuted by seeing two examples whose
      nonzero color pairs differ.

  component_coupling
      per-cell independence has been refuted by an example on which the local
      and component rules disagree and the component rule is correct.

  frequency_assumption
      a frequency-based role assignment remains viable until an example
      violates it.  None of A/B/W currently does, so this commitment survives
      unless dominated structurally once the component schema is available.

The learner starts with the first example's fixed colors and a per-cell schema.
On later examples it performs local repairs only when forced.  Therefore the
trajectory can depend on order even if the final semantic state is confluent.

Run:
    python3 miniARC/confluence_7e0986d6.py
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import permutations
import json
from pathlib import Path

from real_arc_generality_7e0986d6 import (
    TASK,
    structural,
    per_cell_local,
    nonzero_frequency_roles,
)

Grid = list[list[int]]
Pair = tuple[Grid, Grid]

W_IN: Grid = [
    [1, 1, 1],
    [1, 2, 2],
]
W_OUT: Grid = structural(W_IN)
assert W_OUT == [[1, 1, 1], [1, 1, 1]]
assert per_cell_local(W_IN) != W_OUT


@dataclass(frozen=True)
class State:
    # None before first example; otherwise the concrete role pair initially
    # hypothesized from that first example.
    fixed_roles: tuple[int, int] | None = None
    color_parametric: bool = False
    component_coupling: bool = False
    frequency_assumption: bool = True
    seen: tuple[str, ...] = ()

    def representation(self) -> str:
        color = "parametric-colors" if self.color_parametric else f"fixed-colors={self.fixed_roles}"
        structure = "components" if self.component_coupling else "per-cell"
        freq = "+freq-role" if self.frequency_assumption else ""
        return f"{structure}/{color}{freq}"

    def semantic_key(self) -> tuple[bool, bool, bool, tuple[int, int] | None]:
        # Once colors are parametric, the initial fixed role pair is semantically
        # irrelevant and should not make two final states different.
        roles = None if self.color_parametric else self.fixed_roles
        return (self.color_parametric, self.component_coupling, self.frequency_assumption, roles)


def pair_roles(x: Grid) -> tuple[int, int] | None:
    return nonzero_frequency_roles(x)


def local_fits(pair: Pair) -> bool:
    x, y = pair
    return per_cell_local(x) == y


def structural_fits(pair: Pair) -> bool:
    x, y = pair
    return structural(x) == y


def step(state: State, label: str, pair: Pair) -> tuple[State, list[str]]:
    """Incrementally repair the current representation only when evidence forces it."""
    x, y = pair
    events: list[str] = []
    s = state
    roles = pair_roles(x)

    if s.fixed_roles is None:
        s = replace(s, fixed_roles=roles)
        events.append(f"initialize color roles to {roles}")
    elif not s.color_parametric and roles != s.fixed_roles:
        s = replace(s, color_parametric=True)
        events.append(f"anti-unify color roles: {s.fixed_roles} and {roles} -> (N,R)")

    # Current structural commitment starts per-cell.  A counterexample to that
    # schema triggers the component abstraction.  We require the component rule
    # itself to fit, otherwise this finite learner has no repair operation.
    if not s.component_coupling and not local_fits(pair):
        if not structural_fits(pair):
            raise AssertionError(f"{label}: neither local nor structural schema fits")
        s = replace(s, component_coupling=True)
        events.append("refine per-cell -> connected nuisance component")

    # Frequency roles are an extra assumption inside either schema.  If an
    # example has exactly two colors but the intended output contradicts the
    # frequency-based solver, it would be refuted here.  For A/B/W it remains
    # empirically viable.  Once component structure is discovered, however,
    # the frequency-constrained component explanation is dominated by the same
    # component rule without that assumption, so discharge it structurally.
    if s.component_coupling and s.frequency_assumption:
        s = replace(s, frequency_assumption=False)
        events.append("discharge frequency-role assumption (dominated by component rule)")

    s = replace(s, seen=s.seen + (label,))
    return s, events


def load_examples() -> dict[str, Pair]:
    task = json.loads(Path(TASK).read_text())
    assert len(task["train"]) == 2
    return {
        "A": (task["train"][0]["input"], task["train"][0]["output"]),
        "B": (task["train"][1]["input"], task["train"][1]["output"]),
        "W": (W_IN, W_OUT),
    }


def run_order(order: tuple[str, ...], examples: dict[str, Pair]) -> tuple[State, list[str]]:
    state = State()
    trace = ["start: " + state.representation()]
    for label in order:
        state, events = step(state, label, examples[label])
        trace.append(f"after {label}: {state.representation()}")
        trace.extend("    " + e for e in events)
    return state, trace


def main() -> None:
    examples = load_examples()
    results = []
    print("ARC 7e0986d6 curriculum/confluence experiment\n")
    print("W witness:")
    print("  RRR    -> RRR")
    print("  Rnn       RRR\n")

    for order in permutations(("A", "B", "W")):
        final, trace = run_order(order, examples)
        results.append((order, final, trace))
        print("ORDER", "->".join(order))
        for line in trace:
            print(" ", line)
        print("  final semantic key:", final.semantic_key())
        print()

    keys = {final.semantic_key() for _, final, _ in results}
    reps = {final.representation() for _, final, _ in results}

    print("SUMMARY")
    print("  number of curricula:            ", len(results))
    print("  distinct final representations: ", len(reps))
    print("  distinct final semantic states: ", len(keys))
    print("  strongly confluent:              ", len(reps) == 1)
    print("  semantically confluent:          ", len(keys) == 1)

    # The designed finite learner should converge on the same endpoint: colors
    # parametric, component coupling retained, frequency assumption discharged.
    expected = (True, True, False, None)
    assert keys == {expected}, keys

    # Trajectories should nevertheless differ: if W arrives first, component
    # structure is discovered immediately; if W arrives last, the local schema
    # survives both large ARC demonstrations before being refuted by six cells.
    first_events = {order: trace for order, _, trace in results}
    assert "components" in first_events[("W", "A", "B")][1]
    assert "per-cell" in first_events[("A", "B", "W")][2]

    print("\nInterpretation:")
    print("  endpoint is order-independent in this learner, but discovery time is not.")
    print("  The six-cell witness can reveal component structure at step 1 or step 3")
    print("  depending on curriculum, while the final inductive semantics is unchanged.")


if __name__ == "__main__":
    main()
