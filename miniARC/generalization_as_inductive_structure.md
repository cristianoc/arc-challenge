# Generalization as Inductive Structure

## A compact account motivated by ARC and epiplexity

### 1. Setup

Let `X` be a space of inputs, `Y` a space of outputs, and `H` a class of explanations. An explanation `h` denotes a (possibly partial) map from `X` to `Y`.

For a finite set of examples

$$E=\{(x_i,y_i)\}_{i=1}^n,$$

define the version space

$$V(E)=\{h\in H \mid h(x_i)=y_i\text{ for all }i\}.$$

Interpolation asks only for some element of `V(E)`. Generalization requires additional structure on `H`.

We assume a preorder

$$\succeq\;\subseteq H\times H,$$

where

$$h\succeq g$$

means that `h` is at least as general as `g`. No total order or numerical score is assumed.

The preferred explanations are the maximal consistent elements

$$G(E)=\{h\in V(E)\mid \nexists g\in V(E).\ g\succ h\}.$$

Thus `G(E)` may contain several incomparable explanations.

This is intentional: ambiguity is preserved unless the theory provides a reason to remove it.

---

### 2. Structural sources of the preorder

#### Constraint weakening

Represent an explanation as

$$(t,\Phi),$$

where `t` is a parametric term and `Phi` is a constraint on its parameters or applicability.

For a fixed term `t`, define

$$(t,\Phi_1)\succeq(t,\Phi_2)$$

whenever

$$\Phi_2\Rightarrow\Phi_1.$$

So a weaker assumption gives a more general explanation.

Examples:

```text
Largest & Leftmost        <  Largest
FlipX(size in {2,3,4})    <  FlipX(any size)
component + frequency     <  component
```

No assumptions are counted; logical implication supplies the order.

#### Anti-unification

Constraint weakening presupposes a common parametric term. Such a term may itself be learned.

From ground explanations

$$t_1,\ldots,t_n,$$

anti-unification seeks a term `T` and substitutions `sigma_i` such that

$$t_i=T\sigma_i.$$

For example,

```text
Flip(X,2)
Flip(X,3)      --->      Flip(X,n)
Flip(X,4)
```

Once `Flip(X,n)` exists, the three observations are no longer unrelated facts; they are instances of one schema.

A counterexample may later show that a variable was too permissive, forcing a split such as

```text
Flip(a,n)  --->  Flip(X,n), Flip(Y,n).
```

Hence representation learning has two dual moves:

> merge distinctions that do not matter; split distinctions that do matter.

---

### 3. Consequence semantics of a representation

A representation `R` determines which unseen instances follow from which observations. Extensionally, this can often be represented by a closure operator

$$cl_R : P(G)\to P(G)$$

on a ground universe `G`, satisfying

$$S\subseteq cl_R(S),$$

$$S\subseteq T\Rightarrow cl_R(S)\subseteq cl_R(T),$$

and

$$cl_R(cl_R(S))=cl_R(S).$$

The interpretation is simple:

$$cl_R(E)=\text{the consequences licensed by representation }R\text{ from }E.$$

The important distinction is:

> **Closure is not learning. Closure is the semantics of a fixed inductive representation. Learning changes the representation.**

A learning trajectory therefore has the form

$$R_0\to R_1\to\cdots\to R_k,$$

with corresponding consequence structures

$$cl_{R_0},cl_{R_1},\ldots,cl_{R_k}.$$

Predictions may disappear across this trajectory because the representation itself can change.

If an intended concept `T subset G` is known, define

$$cl_R(E)\subseteq T$$

as inductive soundness,

$$T\subseteq cl_R(E)$$

as inductive completeness, and

$$cl_R(E)=T$$

as exact generalization.

Thus undergeneralization and overgeneralization are distinct inclusion failures. A useful refinement may enlarge or shrink closure; “more general” is not the same as “better”.

---

### 4. Representation changes generalization

The same ground behaviors can induce different generality structures in different representations.

At pixel level we may have unrelated atoms

```text
Flip2   Flip3   Flip4
```

while an object representation exposes

```text
          FlipX(n)
         /   |   \
  FlipX(2) FlipX(3) FlipX(4)
```

The observations have not changed. The representation has introduced substitution structure that was previously absent.

This gives a precise sense in which abstraction can improve generalization:

> **an abstraction is useful when it exposes valid generality relations between concrete explanations.**

This is logically independent of whether it also reduces search cost.

---

### 5. A real ARC example

Consider ARC task `7e0986d6`. Its official demonstrations contain large regions of one foreground color disturbed by small components of another. The output restores nuisance cells that belong to a nearby large component and removes the remaining nuisance cells.

A crop from the actual ARC interface is shown below.

<p align="center">
  <img src="figures/arc_7e0986d6_screenshot_crop.png" width="650" alt="ARC task 7e0986d6 example 1 input and output">
</p>

We compared four executable explanations:

| explanation | train | test | status |
|---|:---:|:---:|---|
| connected-component repair | yes | yes | undominated |
| same repair + frequency assumptions | yes | yes | dominated |
| first-example colors fixed | no | no | refuted |
| per-cell local repair | yes | yes | incomparable |

The first nontrivial structural relation is

$$h_{component}\succ h_{frequency},$$

because the second is the same component rule with additional assumptions about global color frequencies.

The fixed-color explanation is eliminated by evidence: the second demonstration changes the foreground colors, forcing color identity to become a parameter.

However,

$$h_{component}\parallel h_{local}.$$

Both fit every official training and test pair, and the current preorder does not compare them.

So even interpolation plus the official held-out test does not uniquely determine the explanatory schema.

#### A minimal discriminating witness

An exhaustive small-grid search found the following witness:

![Minimal distinguishing witness](images/7e0986d6_witness.svg)

The two nuisance cells form one connected component. The component rule transforms them together; the per-cell rule treats them independently and erases the rightmost cell.

Hence, for the added example `w`,

$$h_{component}\models w,$$

while

$$h_{local}\not\models w.$$

This is evidence elimination, not an order relation between the two schemas.

The salient fact is that a six-cell example distinguishes explanations that the much larger official examples do not.

Thus the informational value of an example is state-relative: it depends on which ambiguity it resolves, not on its raw size.

---

### 6. Confluence and learning path

Let the two official demonstrations be `A,B` and the witness be `W`.

We implemented an incremental learner whose state records three commitments:

1. fixed versus parametric colors;
2. per-cell versus connected-component structure;
3. presence or absence of an extra frequency assumption.

The learner updates its current representation incrementally rather than recomputing the batch solution from scratch.

All six permutations of `A,B,W` reached the same final state:

```text
parametric colors
connected nuisance components
no frequency assumption
```

Thus, in this finite experiment, the endpoint is strongly confluent.

But the trajectories differ. Under

```text
A -> B -> W
```

component structure is discovered only at the third example. Under

```text
W -> A -> B
```

it is discovered immediately.

Therefore we have, in one microscopic example,

$$\text{same final inductive structure}$$

but

$$\text{different discovery paths}.$$

This suggests separating two questions:

$$\boxed{\text{What is the generalization?}}$$

and

$$\boxed{\text{How difficult is it to discover?}}$$

The first is semantic and order-theoretic. The second depends on curriculum, computation, and search.

---

### 7. Relation to epiplexity

Epiplexity operationalizes learnable structure through a learning experiment. The analysis above suggests a possible factorization of that idea.

The structural object is an inductive representation `R`, or equivalently its consequence relation / closure semantics. Learning constructs a path

$$R_0\to R_1\to\cdots\to R_*.$$

An epiplexity-like scalar may then be interpreted as measuring some aspect of the difficulty of reaching `R_*`: for example the evidence, computation, or trajectory required to discover it.

On this view, the learning experiment remains useful, but it is no longer the definition of structure itself.

The candidate conceptual separation is:

> **Generalization is the induced consequence structure. Epiplexity measures something about the cost of discovering that structure.**

This also explains why curriculum can affect a learning curve even when the final inductive endpoint is invariant.

---

### 8. Open problem

The central unresolved question is the choice of admissible representations and refinement operations.

Given sparse evidence `E`, which anti-unifications, parameter domains, abstractions, and splits are available to the learner?

Equivalently: which preorder on explanations is justified independently of knowing the intended answer?

A useful empirical program is therefore:

1. enumerate multiple explanations for real ARC tasks;
2. derive structural order relations by substitution and constraint weakening;
3. retain incomparable survivors rather than force a total ranking;
4. automatically generate discriminating witnesses;
5. study whether the resulting representation refinements are confluent across curricula.

The hypothesis is that a substantial part of ARC generalization can be described by such structured refinement, without assigning every solution a scalar complexity.