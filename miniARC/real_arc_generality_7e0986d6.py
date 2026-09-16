"""Real ARC generality experiment on training task 7e0986d6.

This is the first experiment in this directory that uses an actual ARC task
rather than a synthetic finite world.

It compares several *training-consistent candidate explanations* without
assigning them numeric scores.  Candidate explanations expose assumptions; the
only preference edges are justified assumption weakenings.

The reference structural rule is independently corroborated by the executable
ARC-AGI solution atlas: same-colored 4-connected components smaller than four
cells are noise; components of at least four cells are references.  Noise
adjacent to at least two cells of a reference is restored to that reference's
color, otherwise erased to background.

Candidates:

  structural
      The component rule above.  No fixed foreground colors and no global
      frequency ordering are assumed.

  frequency_shortcut
      First assumes that the nuisance color is globally the least frequent
      nonzero color and the reference color is the most frequent one, then
      performs the same local restoration.  If it fits, it is an explicitly
      more constrained explanation, not merely a "longer" or "shorter" one.

  fixed_first_pair
      Specializes the color roles to the first training pair.  It is expected
      to fail the second pair and is included to demonstrate that the task
      itself forces color-parametricity.

  per_cell_local
      A deliberately different schema: infer colors by frequency and decide
      each nuisance cell independently from immediate neighbors.  It is not
      ordered against the component schema.  If it fits training but fails
      test, that is useful evidence that our current preorder needs no extra
      edge: ordinary evidence eliminates it.  If it fits test too, the two
      explanations remain genuinely incomparable under the current theory.

Run from the repository root:

    python miniARC/real_arc_generality_7e0986d6.py
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from collections import Counter
from typing import Callable

Grid = list[list[int]]
TASK = Path(__file__).parents[1] / "data" / "training" / "7e0986d6.json"


def components(grid: Grid) -> list[list[tuple[int, int]]]:
    unseen = {(r, c) for r, row in enumerate(grid) for c, v in enumerate(row) if v != 0}
    out: list[list[tuple[int, int]]] = []
    while unseen:
        seed = unseen.pop()
        color = grid[seed[0]][seed[1]]
        todo = [seed]
        comp = [seed]
        while todo:
            r, c = todo.pop()
            for q in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if q in unseen and 0 <= q[0] < len(grid) and 0 <= q[1] < len(grid[0]) and grid[q[0]][q[1]] == color:
                    unseen.remove(q)
                    todo.append(q)
                    comp.append(q)
        out.append(comp)
    return out


def structural(grid: Grid) -> Grid:
    """Color-parametric component explanation."""
    groups = components(grid)
    refs = [g for g in groups if len(g) >= 4]
    out = [row[:] for row in grid]
    for noise in (g for g in groups if len(g) < 4):
        neighbors = {
            (r + dr, c + dc)
            for r, c in noise
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
        }
        replacement = 0
        for ref in refs:
            if len(neighbors.intersection(ref)) >= 2:
                rr, cc = ref[0]
                replacement = grid[rr][cc]
                break
        for r, c in noise:
            out[r][c] = replacement
    return out


def nonzero_frequency_roles(grid: Grid) -> tuple[int, int] | None:
    counts = Counter(v for row in grid for v in row if v)
    if len(counts) != 2:
        return None
    ordered = sorted(counts, key=lambda x: (counts[x], x))
    return ordered[0], ordered[-1]  # nuisance, reference


def frequency_shortcut(grid: Grid) -> Grid:
    """Structural repair plus the extra assumption nuisance=freq-min."""
    roles = nonzero_frequency_roles(grid)
    if roles is None:
        return [row[:] for row in grid]
    nuisance, reference = roles
    # Enforce the extra role assumption by rejecting components whose colors do
    # not have the expected global role.
    groups = components(grid)
    refs = [g for g in groups if len(g) >= 4 and grid[g[0][0]][g[0][1]] == reference]
    out = [row[:] for row in grid]
    for g in groups:
        color = grid[g[0][0]][g[0][1]]
        if color != nuisance:
            continue
        neighbors = {
            (r + dr, c + dc)
            for r, c in g
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
        }
        replacement = 0
        for ref in refs:
            if len(neighbors.intersection(ref)) >= 2:
                replacement = reference
                break
        for r, c in g:
            out[r][c] = replacement
    return out


def make_fixed_color_rule(nuisance: int, reference: int) -> Callable[[Grid], Grid]:
    def solve(grid: Grid) -> Grid:
        groups = components(grid)
        refs = [g for g in groups if len(g) >= 4 and grid[g[0][0]][g[0][1]] == reference]
        out = [row[:] for row in grid]
        for g in groups:
            if grid[g[0][0]][g[0][1]] != nuisance:
                continue
            neighbors = {
                (r + dr, c + dc)
                for r, c in g
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
            }
            replacement = reference if any(len(neighbors.intersection(ref)) >= 2 for ref in refs) else 0
            for r, c in g:
                out[r][c] = replacement
        return out
    return solve


def per_cell_local(grid: Grid) -> Grid:
    """Different schema: each nuisance cell is classified independently."""
    roles = nonzero_frequency_roles(grid)
    if roles is None:
        return [row[:] for row in grid]
    nuisance, reference = roles
    h, w = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    for r in range(h):
        for c in range(w):
            if grid[r][c] != nuisance:
                continue
            k = sum(
                0 <= rr < h and 0 <= cc < w and grid[rr][cc] == reference
                for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
            )
            out[r][c] = reference if k >= 2 else 0
    return out


@dataclass(frozen=True)
class Candidate:
    name: str
    solve: Callable[[Grid], Grid]
    schema: str
    assumptions: frozenset[str]


def agrees(solve: Callable[[Grid], Grid], pairs: list[dict]) -> bool:
    return all(solve(p["input"]) == p["output"] for p in pairs)


def strict_assumption_weakening(a: Candidate, b: Candidate) -> bool:
    """a dominates b only inside the same schema and with fewer assumptions."""
    return a.schema == b.schema and a.assumptions < b.assumptions


def main() -> None:
    task = json.loads(TASK.read_text())
    first_roles = nonzero_frequency_roles(task["train"][0]["input"])
    assert first_roles is not None
    fixed = make_fixed_color_rule(*first_roles)

    candidates = (
        Candidate("structural", structural, "component-repair", frozenset()),
        Candidate(
            "frequency_shortcut",
            frequency_shortcut,
            "component-repair",
            frozenset({"nuisance-is-global-frequency-min", "reference-is-global-frequency-max"}),
        ),
        Candidate(
            "fixed_first_pair",
            fixed,
            "component-repair",
            frozenset({f"nuisance-color={first_roles[0]}", f"reference-color={first_roles[1]}"}),
        ),
        Candidate(
            "per_cell_local",
            per_cell_local,
            "per-cell-local",
            frozenset({"nuisance-is-global-frequency-min", "reference-is-global-frequency-max"}),
        ),
    )

    train_ok = {c.name: agrees(c.solve, task["train"]) for c in candidates}
    test_ok = {c.name: agrees(c.solve, task["test"]) for c in candidates}

    print("ARC task 7e0986d6\n")
    print("candidate                    train   test   schema")
    print("------------------------------------------------------------")
    for c in candidates:
        print(f"{c.name:28} {str(train_ok[c.name]):5}   {str(test_ok[c.name]):5}  {c.schema}")

    survivors = tuple(c for c in candidates if train_ok[c.name])
    edges = tuple((a, b) for a in survivors for b in survivors if strict_assumption_weakening(a, b))
    maximal = tuple(c for c in survivors if not any(a == other and b == c for a, b in edges for other in (a,)))

    print("\njustified preference edges among training-consistent candidates")
    if not edges:
        print("  (none)")
    for a, b in edges:
        print(f"  {a.name}  >  {b.name}    [assumption weakening]")

    print("\nundominated training-consistent candidates")
    for c in maximal:
        print(" *", c.name)

    print("\nobservations")
    if train_ok["structural"]:
        print(" - structural rule fits all ARC demonstrations")
    if test_ok["structural"]:
        print(" - structural rule also reproduces the official ARC test output")
    if not train_ok["fixed_first_pair"]:
        print(" - fixed colors are refuted by the demonstrations themselves: color identity must be parametric")
    if train_ok["frequency_shortcut"]:
        print(" - frequency shortcut fits training but is dominated by structural under the declared component schema")
    if train_ok["per_cell_local"] and not test_ok["per_cell_local"]:
        print(" - local shortcut is eliminated by the real ARC test, without adding a preference edge")
    elif train_ok["per_cell_local"] and test_ok["per_cell_local"]:
        print(" - local and component explanations remain extensionally compatible with this task; current preorder leaves them incomparable")

    # Reference solver should be exact on this public task.
    assert train_ok["structural"]
    assert test_ok["structural"]


if __name__ == "__main__":
    main()
