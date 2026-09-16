# Generalization as assumption discharge

This note develops the partial-order account one step further. The aim is still to avoid assigning numeric scores to solutions.

## 1. Start from what is actually justified

A finite training set does not justify an unconditional rule. It justifies a rule on the observed situations.

So instead of treating a candidate solution as a bare function `h`, treat the initial explanatory object as a guarded claim

```text
A => h
```

where `A` describes the situations on which the claim is currently supported.

Generalization is then a move from a stronger guard to a weaker guard:

```text
A => h
|
v
B => h
```

where every situation satisfying `A` also satisfies `B`.

The endpoint `True => h` is an unconditional generalization.

The preference relation is therefore not based on a number. For the same explanatory behavior, weaker assumptions are preferred:

```text
True => h
    >
A => h
    >
A & B => h
```

This subsumes the first MiniARC+ relation. For example,

```text
Largest => X
        >
Largest & Leftmost => X
```

is just assumption weakening.

## 2. The hard part is not the order

Once guarded explanations are given, the order is straightforward. The hard question is:

> Which assumptions may be discharged?

This is where the inductive content lives.

For example, from observations in which the marker is always four cells to the right of the object, both of these guarded explanations are valid:

```text
Gap=4 => destination = object+3
Gap=4 => destination = marker-1
```

Both can be weakened syntactically to unconditional claims:

```text
True => destination = object+3
True => destination = marker-1
```

but the observations alone do not license either weakening. On the observed subspace the two formulas are equal.

Therefore the theory must distinguish:

1. **ordering explanations once their assumptions are known**, from
2. **licensing a discharge step**.

The first is semantic inclusion. The second is the generalization problem.

## 3. Do not discharge assumptions merely because a hypothesis ignores them

A tempting rule is:

> If the right-hand side of an explanation does not mention an assumption, discharge it.

This is unsound as an inductive principle.

From

```text
Gap=4 => destination = object+3
```

`Gap=4` does not occur syntactically in `object+3`, but dropping it is exactly the unsupported extrapolation we are trying to understand.

Likewise, a constant-output rule may not mention color, object count, position, or shape. That does not make it valid under arbitrary changes to all of them.

Syntactic non-use is not evidence of independence.

## 4. Variation can license some discharges

There is, however, one very conservative operation available directly from evidence.

Suppose examples vary a dimension while the same explanatory relation continues to hold. Then the conjunction describing the observations need not fix that dimension to one value.

Example:

```text
example 1: object-x = 0, marker-x = 4
example 2: object-x = 1, marker-x = 5
example 3: object-x = 2, marker-x = 6
```

Absolute position varies. A guard saying

```text
object-x = 0
```

cannot describe all examples, whereas a translation-invariant guard can.

This is not yet arbitrary extrapolation. It is ordinary abstraction of observed variation.

A useful distinction is therefore:

- **interpolation discharge**: remove distinctions already varied by the evidence;
- **extrapolation discharge**: remove a restriction outside the range demonstrated by the evidence.

The first can often be justified extensionally from the examples. The second requires an inductive principle.

## 5. A finite abstract state space

To make this concrete, represent each world by a finite tuple of abstract features.

For the relational toy problem:

```text
(object_position, marker_position, gap, displacement)
```

For object MiniARC:

```text
(number_of_objects,
 selected_size_rank,
 selected_x_rank,
 selected_y_rank,
 action)
```

An assumption is simply a set of abstract worlds. A guarded explanation is a partial behavior on such a set.

This gives us exact operations:

```text
strengthen(A, p) = A intersect worlds(p)
weaken(A, p)     = remove constraint p when representable
```

and an exact semantic order by set inclusion.

No numeric complexity is needed.

## 6. Evidence gives a lower bound, not a unique guard

Let `Obs(E)` be the set of abstract worlds represented by the examples.

Any admissible guard for an explanation must contain the observed worlds on which that explanation is claimed:

```text
Obs(E) subset A
```

But there are generally many such `A`.

At one extreme:

```text
A = Obs(E)
```

which makes no extrapolation.

At the other:

```text
A = Universe
```

which makes the rule unconditional.

Generalization is movement upward through this family of guards.

This makes ambiguity unavoidable and explicit: there can be many incomparable ways to enlarge `Obs(E)`.

## 7. The object to compute is a frontier

For a fixed explanatory body `h`, define

```text
Admissible(E,h) = guards A containing Obs(E)
                  for which A => h is currently licensed
```

The maximally weak licensed guards are the current generalizations of `h`.

Across different explanatory bodies, retain the undominated guarded explanations.

The result is a frontier, not a winner.

A unique intended solution appears only when the available discharge principles leave one undominated explanation relevant to the test.

## 8. Candidate discharge principles should be local operations

Rather than define a global score, add small discharge rules one at a time.

A candidate rule has the form

```text
A & p => h
-------------  D
A => h
```

where `D` states when removing `p` is licensed.

The research program is to discover a small set of defensible `D`s.

Potential sources include:

- observed variation;
- renaming invariance;
- translation/rotation/color symmetries already demonstrated by examples;
- object permutation;
- parametricity/uniformity of a derivation;
- relational structure preserved across examples.

These are hypotheses to test, not axioms to assume wholesale.

## 9. A first rule we can defend: observed-variation discharge

Suppose a guard contains `p = c`, but the examples supporting the same explanatory body contain at least two different values of `p` while preserving the body relation.

Then `p = c` cannot be a common assumption of those examples. The explanation must already have been generalized over `p` enough to cover the observed values.

In a finite feature space this can be expressed without a score: replace singleton constraints by the smallest available feature-set containing all observed values.

For example:

```text
object-x in {0,1,2,3}
```

is justified by four observed positions, while

```text
object-x in ALL_POSITIONS
```

is an additional extrapolation if other positions were not observed.

This gives a clean boundary between what the examples force and what an inductive principle adds.

## 10. Why this is useful for ARC

An ARC training set is sparse. Many candidate programs agree on it.

Instead of asking which agreeing program has the lowest scalar complexity, we can ask:

1. what assumptions make each explanation valid on the examples?
2. which assumptions are already eliminated by observed variation?
3. which additional assumptions can be discharged by a small collection of general principles?
4. after these discharge steps, which explanations dominate others?
5. which incomparable explanations remain genuinely ambiguous?

This turns "natural generalization" into a sequence of explicit, inspectable operations.

## 11. Relation to learning experiments

This also reconnects to the original epiplexity discussion.

A learning process can be viewed as discovering discharge steps:

```text
very specific explanation
        |
        v
fewer assumptions
        |
        v
more reusable explanation
```

A training curve may reveal when these transitions become accessible, but the transitions themselves can be defined independently of neural training.

This suggests a possible abstract object behind the experiment: the partially ordered space of explanations reachable by licensed assumption discharge under bounded computation.

That is deliberately richer than a single scalar.

## 12. Next concrete test

The next implementation should use a finite feature universe and enumerate guards explicitly.

We should construct examples where:

- one assumption can be discharged purely because the examples vary it;
- one assumption cannot yet be discharged;
- two different discharge paths produce incomparable explanations;
- an additional example licenses one path but not the other.

Then we can ask whether actual small ARC puzzles can be represented as the same kind of discharge process before introducing any stronger principle.