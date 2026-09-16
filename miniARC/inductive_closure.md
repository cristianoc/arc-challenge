# Inductive closure: soundness, completeness, and refinement

This note develops the closure view of generalization far enough to state the
basic correctness questions without assigning numeric scores to hypotheses or
representations.

## 1. Ground universe and intended concept

Let `G` be a finite universe of ground explanations/situations. A puzzle family
has an intended concept

$$T \subseteq G.$$

The observed training instances are

$$E \subseteq T.$$

A representation `R` determines a closure operator

$$cl_R : P(G) \to P(G)$$

with the usual properties:

$$S \subseteq cl_R(S),$$

$$S \subseteq U \Rightarrow cl_R(S) \subseteq cl_R(U),$$

and

$$cl_R(cl_R(S)) = cl_R(S).$$

The prediction/generalization induced by `R` from evidence `E` is simply

$$cl_R(E).$$

The intended target is not "the largest closure". It is `T`.

## 2. Inductive soundness and completeness

For a fixed task `(E,T)`, call `R` **inductively sound** when

$$cl_R(E) \subseteq T.$$

It makes no generalization outside the intended concept.

Call `R` **inductively complete** when

$$T \subseteq cl_R(E).$$

It reaches every intended instance.

Call it **exact** when both hold:

$$cl_R(E)=T.$$

This separates the two characteristic errors:

- undergeneralization: `cl_R(E)` is a strict subset of `T`;
- overgeneralization: `T` is a strict subset of `cl_R(E)` or the two sets overlap without inclusion.

The pixel/object/bad experiment from `representation_closure.py` is then:

```text
pixel:   E = cl_pixel(E)  proper-subset  T
object:      cl_object(E) = T
bad:         T  proper-subset  cl_bad(E)
```

So object abstraction is not "better because it generalizes more". It is
better for this task because it repairs undergeneralization without crossing
the target boundary.

## 3. Representation order is not a quality order

Closure operators have a natural pointwise order. Write

$$R_1 \preceq R_2$$

when

$$cl_{R_1}(S) \subseteq cl_{R_2}(S)$$

for every `S`.

This means only that `R_2` is a more aggressive/coarser inductive
representation: it identifies at least everything `R_1` identifies.

It does **not** mean `R_2` is better.

For the toy example:

$$R_{pixel} \preceq R_{object} \preceq R_{bad},$$

while `R_object` is the exact one.

This is analogous to ordering abstract domains by precision: the order is
structural; correctness relative to a semantic target is a separate property.

## 4. Refinement relative to a task

Suppose `R` undergeneralizes:

$$cl_R(E) \subset T.$$

A task-directed **expansive refinement** from `R` to `R'` should satisfy

$$cl_R(E) \subseteq cl_{R'}(E) \subseteq T.$$

It adds justified generalizations but remains sound.

If `R` overgeneralizes, a **restrictive refinement** should satisfy

$$T \subseteq cl_{R'}(E) \subseteq cl_R(E).$$

It removes false generalizations while retaining completeness.

Thus "refinement" need not always mean more or less abstract. Its direction is
relative to the error being repaired.

For a finite task, exactness is reached when neither error remains.

## 5. The target is unavailable during real learning

The definitions above use `T`, which is known in synthetic experiments but not
to the learner. This is intentional: soundness/completeness are semantic
specifications used to evaluate a learning/generalization procedure, not rules
that the learner can invoke directly.

This mirrors ordinary correctness definitions. A program verifier need not
know a proof automatically merely because semantic validity is well-defined.

The learning problem is therefore:

> From `E` and available representational operations, discover a representation
> `R` for which `cl_R(E)` approaches the unknown intended concept `T`.

The theory of what counts as correct can be clean even when discovering the
correct representation is computationally difficult.

## 6. Why adding examples can change generalization

This framework makes the dependence on evidence explicit:

$$E \mapsto cl_R(E).$$

For fixed `R`, monotonicity gives

$$E_1 \subseteq E_2 \Rightarrow cl_R(E_1) \subseteq cl_R(E_2).$$

But in a learner that can change representation, new examples may also cause

$$R_1 \longrightarrow R_2.$$

So the actual transition is

$$cl_{R_1}(E_1) \longrightarrow cl_{R_2}(E_2).$$

The second effect can be much larger than the first. A new example may reveal
that several previously unrelated cases are instances of one parametric term,
causing a representation change and therefore a large jump in closure.

This gives a precise version of the idea that examples can do more than
eliminate hypotheses: they can expose a new generality structure.

## 7. Representation discovery as closure discovery

Let a learner at stage `i` have representation `R_i`. Its acquired inductive
structure can be identified extensionally with `cl_{R_i}`.

Learning may then be viewed as a sequence

$$R_0 \to R_1 \to \cdots \to R_k$$

or extensionally

$$cl_0 \to cl_1 \to \cdots \to cl_k.$$

A structural learning event occurs when the new closure operator relates ground
instances that the old one did not.

For example:

```text
before:
    FlipX(2)   FlipX(3)   FlipX(4)
    unrelated ground atoms

after:
              FlipX(N)
             /   |   \
       FlipX(2) FlipX(3) FlipX(4)
```

The acquired object/size abstraction has changed which finite sets have
nontrivial closure.

## 8. A task family, not one task

A single `(E,T)` cannot determine whether a representation is generally useful.
A representation could accidentally be exact on one puzzle.

Let a task family be

$$F = \{(E_j,T_j)\}_j.$$

`R` is sound for the family when

$$cl_R(E_j) \subseteq T_j$$

for every task `j`, and complete when

$$T_j \subseteq cl_R(E_j)$$

for every `j`.

This is the first place where robustness enters naturally. A representation is
not validated because it extrapolates one puzzle correctly, but because the
same closure structure remains sound across a family of independently varying
problems.

No averaging or score is necessary. Failure on one task is simply a failed
property.

## 9. Relative comparison without scalarization

For two representations `R1` and `R2`, relative to a task family `F`, useful
pairwise relations include:

- `R2` repairs an undergeneralization of `R1` if it strictly enlarges closure on
  at least one task while remaining inside every target;
- `R2` repairs an overgeneralization of `R1` if it strictly removes predictions
  on at least one task while still covering every target;
- they are incomparable when one improves one dimension/task while worsening
  another.

This yields a partial order of justified improvements rather than a total
ranking.

## 10. Relation to abstraction refinement

In an ARC solver with representations

$$A \to A^+ \to G,$$

we can separate two effects of moving from `A` to `A+`:

1. **search effect:** the new representation changes the cost of finding
   candidate programs;
2. **inductive effect:** the new representation changes the closure operator
   over concrete explanations.

The second effect is logically independent of the first.

An abstraction can therefore help generalization even if search were free:
previously unrelated concrete explanations may become substitution instances
of one abstract term.

Conversely, an abstraction can make search dramatically cheaper while inducing
an incorrect closure. Search efficiency alone does not establish good
inductive bias.

## 11. Connection back to learning experiments

This gives a candidate answer to the question that motivated the investigation.

A learning experiment can reveal two different things:

- **predictive facts:** additional ground instances become predictable;
- **inductive structure:** the learner acquires a closure rule under which many
  ground instances are related as instances of one concept.

The second is more interesting. It can cause a large number of new predictions
without those predictions being learned independently.

Thus the structural content acquired by learning may be better represented by
a change

$$cl_{before} \longrightarrow cl_{after}$$

than by the number of additional correct predictions alone.

This does not yet define epiplexity or another scalar measure. It gives a
candidate object that a scalar measure might summarize.

## 12. Next falsification experiment

The next experiment should contain a family of finite tasks and at least three
representations:

1. one that systematically undergeneralizes;
2. one whose abstraction gives exact closure across the family;
3. one that looks plausible on the first few tasks but overgeneralizes on a
   held-out structural variation.

Then perform representation refinement using only failures exposed by new
examples. The important observation is whether the repair can be expressed as
an order-theoretic change in the closure system rather than by tuning a numeric
penalty.

A useful concrete candidate is an object transformation family varying
independently in size, position, and action. A representation that abstracts
size but accidentally conflates `FlipX` and `FlipY` should initially appear
successful on symmetric objects and then be refuted by an asymmetric object.
The repair should split one closed concept into two while retaining size
parametricity.

That would demonstrate both directions of refinement in one tiny world:

```text
pixel  --expand-->  conflated object abstraction  --split-->  correct object abstraction
```

The first step repairs undergeneralization; the second repairs
overgeneralization.