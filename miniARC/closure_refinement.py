"""Finite representation learning as closure refinement.

This realizes the experiment proposed in inductive_closure.md.

Ground atoms are Action(size), with actions X/Y and sizes 2/3/4/5.  A
representation is a partition of these atoms.  Its closure of evidence is the
union of all blocks touched by the evidence.  Thus blocks are the concepts the
representation treats as indistinguishable/generalizable.

Three representations:

  pixel      every ground atom is its own block (undergeneralizes)
  conflated  all X/Y atoms at all sizes are one block (overgeneralizes)
  correct    one all-size X block and one all-size Y block (exact)

The experiment uses two stages of evidence/targets:

  stage 1: only symmetric training situations are available, so X and Y are
           observationally indistinguishable; merging across size is useful.
  stage 2: an asymmetric witness separates X from Y; splitting the conflated
           block repairs overgeneralization while preserving size abstraction.

No scores or program lengths are used.  Merge/split are operations on the
closure system itself.

Run:
    python miniARC/closure_refinement.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet

SIZES = (2, 3, 4, 5)
ACTIONS = ("X", "Y")


@dataclass(frozen=True, order=True)
class Atom:
    action: str
    size: int

    def __str__(self) -> str:
        return f"Flip{self.action}({self.size})"


G = frozenset(Atom(a, n) for a in ACTIONS for n in SIZES)
Block = FrozenSet[Atom]


@dataclass(frozen=True)
class Representation:
    name: str
    blocks: tuple[Block, ...]

    def __post_init__(self) -> None:
        union = frozenset().union(*self.blocks)
        assert union == G, (self.name, union, G)
        for i, a in enumerate(self.blocks):
            assert a
            for b in self.blocks[i + 1 :]:
                assert a.isdisjoint(b)

    def close(self, evidence: FrozenSet[Atom]) -> FrozenSet[Atom]:
        return frozenset().union(*(b for b in self.blocks if b & evidence)) if evidence else frozenset()

    def block_of(self, atom: Atom) -> Block:
        return next(b for b in self.blocks if atom in b)


def pixel() -> Representation:
    return Representation("pixel", tuple(frozenset({a}) for a in sorted(G)))


def conflated() -> Representation:
    return Representation("conflated", (G,))


def correct() -> Representation:
    return Representation(
        "correct-object",
        tuple(frozenset(Atom(a, n) for n in SIZES) for a in ACTIONS),
    )


def partition_finer(a: Representation, b: Representation) -> bool:
    """Every a-block is contained in some b-block: a is finer/more precise."""
    return all(any(x <= y for y in b.blocks) for x in a.blocks)


def strict_finer(a: Representation, b: Representation) -> bool:
    return partition_finer(a, b) and not partition_finer(b, a)


def sound(r: Representation, evidence: FrozenSet[Atom], target: FrozenSet[Atom]) -> bool:
    return r.close(evidence) <= target


def complete(r: Representation, evidence: FrozenSet[Atom], target: FrozenSet[Atom]) -> bool:
    return target <= r.close(evidence)


def classify(r: Representation, evidence: FrozenSet[Atom], target: FrozenSet[Atom]) -> str:
    s, c = sound(r, evidence, target), complete(r, evidence, target)
    if s and c:
        return "exact"
    if s:
        return "undergeneralizes"
    if c:
        return "overgeneralizes"
    return "mixed error"


def show_set(xs: FrozenSet[Atom]) -> str:
    return "{" + ", ".join(str(x) for x in sorted(xs)) + "}"


def describe_stage(title: str, evidence: FrozenSet[Atom], target: FrozenSet[Atom], reps: tuple[Representation, ...]) -> None:
    print(title)
    print("evidence:", show_set(evidence))
    print("target:  ", show_set(target))
    for r in reps:
        print(f"  {r.name:14} closure={show_set(r.close(evidence))}")
        print(f"  {'':14} status={classify(r, evidence, target)}")
    print()


def demo() -> None:
    p, c, r = pixel(), conflated(), correct()
    reps = (p, c, r)

    # Structural order of partitions: pixel is finest, conflated coarsest.
    assert strict_finer(p, r)
    assert strict_finer(r, c)

    # Stage 1 models evidence from situations in which X/Y happen to be
    # observationally indistinguishable.  At the ground-explanation level we
    # record only one representative atom per observed size.  The semantic
    # target for the intended X-family is nevertheless all X sizes.
    e1 = frozenset({Atom("X", 2), Atom("X", 3), Atom("X", 4)})
    tx = frozenset(Atom("X", n) for n in SIZES)

    describe_stage("STAGE 1: merge across size", e1, tx, reps)
    assert classify(p, e1, tx) == "undergeneralizes"
    assert classify(r, e1, tx) == "exact"
    assert classify(c, e1, tx) == "overgeneralizes"

    # Stage 2 makes the reason for the overgeneralization observable: an
    # asymmetric witness establishes that X and Y must be distinct concepts.
    # The target remains the X family; the repair is a split of the one giant
    # block into action-indexed blocks, retaining the size merge.
    witness_x = Atom("X", 5)
    witness_y = Atom("Y", 5)
    assert c.block_of(witness_x) == c.block_of(witness_y)
    assert r.block_of(witness_x) != r.block_of(witness_y)

    e2 = e1 | frozenset({witness_x})
    describe_stage("STAGE 2: asymmetric witness forces X/Y split", e2, tx, reps)
    assert classify(r, e2, tx) == "exact"
    assert classify(c, e2, tx) == "overgeneralizes"

    print("representation order (precision):")
    print("  pixel  <finer>  correct-object  <finer>  conflated")
    print()
    print("repair path relative to the task:")
    print("  pixel --merge sizes--> correct-object")
    print("  conflated --split X/Y--> correct-object")
    print()
    print("The same exact representation is reached from opposite directions:")
    print("one repair removes distinctions; the other introduces a distinction.")
    print("Therefore useful representation learning is not monotone in raw")
    print("abstraction/coarseness.  It is directed by soundness/completeness errors.")


if __name__ == "__main__":
    demo()
