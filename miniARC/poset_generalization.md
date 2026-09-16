# MiniARC+: generalization as a partial order

## Goal

This experiment asks whether ARC-style generalization can be described without assigning a numeric complexity score to every solution.

The original MiniARC has 32 normalized whole-grid transformations. Examples eliminate inconsistent transformations and the shortest surviving transformation is selected; if several transformations of the same minimum length remain, the puzzle is ambiguous.

Here we deliberately remove the numeric selection rule. Instead, hypotheses carry only a **partial order of generality**. Evidence removes inconsistent hypotheses. Among the survivors, hypotheses that are strictly dominated by a more general consistent hypothesis are discarded. Incomparable hypotheses remain ambiguous.

The guiding principle is:

> Do not decide between two solutions unless we have a justified relation saying that one is a generalization of the other.

There is no requirement for a total order, a score, or even a lattice.

Implementation: [`poset_generalization.py`](poset_generalization.py).

## Minimal world

The experiment uses binary 4x4 grids. This is small enough that all

\[
2^{16}=65,536
\]

possible inputs can be enumerated exactly.

Foreground cells form maximal 4-connected objects.

A hypothesis has the form

\[
h = s;a
\]

where `s` is a selector and `a` is an action.

The atomic selectors are:

- `Largest`
- `Smallest`
- `Leftmost`
- `Rightmost`

The empty selector, `Object`, selects every foreground object. Selectors may also contain conjunctions of up to two atomic predicates, for example

```
Largest&Leftmost
```

The initial actions are:

- `X`: horizontal reflection inside the selected object's bounding box;
- `Y`: vertical reflection;
- `R2`: 180-degree rotation.

This gives a small, completely enumerable hypothesis language. The point is not to approximate full ARC yet, but to introduce one important feature absent from original MiniARC: a transformation may apply to an object selected by a property of the input.

## Evidence

For a set of input/output examples \(E\), the version space is simply

\[
V(E)=\{h\mid h\text{ agrees with every example in }E\}.
\]

Evidence has no other role. In particular, examples do not assign scores or modify the generality relation.

## The generality relation

A selector denotes the set of `(grid, object)` pairs that it selects over the complete 4x4 universe.

For selectors \(s_1,s_2\), define

\[
s_1 \sqsupseteq s_2
\]

when every object selected by \(s_2\) is also selected by \(s_1\):

\[
\llbracket s_2\rrbracket\subseteq\llbracket s_1\rrbracket.
\]

Thus `Largest` is more general than `Largest&Leftmost`, because every object satisfying both predicates satisfies `Largest`, while the converse is false.

This relation is computed **extensionally**, by enumerating all 65,536 binary 4x4 grids. It is not inferred from conjunction length or another syntactic score.

For hypotheses with the same action,

\[
s_1;a \sqsupset s_2;a
\]

when \(s_1\) is a strict semantic generalization of \(s_2\).

We currently make no comparison between different actions or between selectors whose denotations are incomparable.

## Preferred solutions

The preferred solutions are the undominated members of the version space:

\[
G(E)=\operatorname{Max}_{\sqsupseteq} V(E).
\]

Equivalently, \(h\in G(E)\) exactly when

1. \(h\) explains every example; and
2. there is no strictly more general hypothesis \(h'\) that also explains every example.

There may be several such hypotheses. That is intentional.

For example, suppose these all explain the observations:

```
Largest;X
Leftmost;X
Largest&Leftmost;X
```

The order contains

```
Largest;X  --------->  Largest&Leftmost;X
Leftmost;X --------->  Largest&Leftmost;X
```

where arrows point from general to specific.

But `Largest;X` and `Leftmost;X` are incomparable. Therefore

\[
G(E)=\{Largest;X,Leftmost;X\}.
\]

The theory removes the gratuitous conjunction but does not manufacture a preference between `Largest` and `Leftmost`.

## Staged experiment

The executable demo constructs an intended rule

\[
Largest;X.
\]

It searches the finite grid universe for examples with controlled ambiguity.

### Stage 1: genuine ambiguity

The first example is chosen so that the version space is exactly

```
Largest;X
Leftmost;X
Largest&Leftmost;X
```

The partial order eliminates only the conjunction. The result is

\[
G(E_1)=\{Largest;X,Leftmost;X\}.
\]

This is the desired behavior: the observation has not established whether size or horizontal position is the relevant selector.

### Stage 2: evidence resolves it

A second example is found in which `Largest;X` remains correct but `Leftmost;X` does not.

Then

\[
V(E_1,E_2)=G(E_1,E_2)=\{Largest;X\}.
\]

The important separation is:

- the **order** says that adding an unnecessary condition is a specialization;
- the **evidence** decides which incomparable generalizations survive.

The partial order does not resolve an ambiguity that should instead be resolved by evidence.

## A useful failure found while building the experiment

The first hand-written demo was intended to establish `Largest;X`, but the other objects happened to be horizontally symmetric. Consequently `Object;X` also fit every example.

Since `Object` is genuinely more general than `Largest`, the partial-order rule preferred

```
Object;X
```

over

```
Largest;X.
```

This was not a failure of the ordering rule. It exposed a defect in the examples: they did not demonstrate that only the largest object should be transformed.

The demo was therefore changed to search automatically for discriminating examples.

This illustrates an attractive property of the approach: a surprising preferred hypothesis can expose information that the examples failed to provide, rather than being hidden by an arbitrary numerical tie-breaker.

## What has been established so far

Very little is assumed.

We do **not** assume:

- shortest programs are preferred;
- a numeric cost exists;
- all hypotheses are comparable;
- every ambiguity has a unique intended solution;
- the hypothesis space forms a lattice.

The only current preference principle is:

> If one consistent hypothesis is obtained from another by imposing a genuine semantic restriction on the objects to which the same action applies, prefer the less restricted hypothesis.

Everything else remains incomparable.

This is intentionally weak. The aim is to earn additional ordering relations from examples rather than postulate a general-purpose complexity score.

## Next experiment: discover the second relation

The next step is to construct an ARC-like case with:

1. an intended hypothesis \(h_i\);
2. a spurious hypothesis \(h_s\);
3. both hypotheses consistent with all training examples;
4. \(h_i\) and \(h_s\) incomparable under the current semantic-extension order;
5. a strong human intuition that \(h_i\) is nevertheless the intended generalization.

That pair is the useful object of study.

Rather than immediately adding another rule, we should inspect what structural fact supports the preference. If it can be expressed as a robust pairwise relation, it becomes a candidate second generator of the preorder.

The process is therefore incremental:

\[
\preceq_1\;\subseteq\;\preceq_2\;\subseteq\;\cdots
\]

where each extension adds only pairwise preferences for which we have an independently defensible reason.

The intended research question is not yet "what score measures generalization?" It is:

> **What is the weakest structured preference relation on explanations that accounts for the generalizations intended in ARC?**

If this succeeds, a scalar measure could later be derived from the structure if useful. It is not part of the definition.