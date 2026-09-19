# Operation-relative abstraction and exact conditioning

This is accepted core mathematics, not an experiment ledger. The checked source
is [SymArc/Learning.lean](SymArc/Learning.lean), imported by the ordinary
`lake build` target. It reuses `Fits` from [Theory.lean](SymArc/Theory.lean).
The earlier closure/equivariance account remains in [MATH.md](MATH.md).
No entropy objective, selector, default preference, confidence estimate or
empirical success rate is promoted to a theorem here.

## 1. What an abstraction shares

Let U be inputs (possibly a grid together with a position), Y outputs, O available
operations, and E : O -> U -> Option Y their interpreter. For a representation
`a : U -> Z` and selector `h : Z -> O`, define

```math
p(u)=E(h(a(u)),u).
```

Equality of a-values shares an operation, not necessarily its resulting value.
The interpreter's input access is part of the declared language. An unrestricted
interpreter could hide the whole task and make a vacuous abstraction look useful.
The theorem makes no computational-compression claim about a or E.

For labelled data D define

```math
V_z=\{o:\forall(u,y)\in D,\ a(u)=z\Rightarrow E(o,u)=y\}.
```

**Coherence theorem.** There exists a selector whose decoded program fits D if
and only if every V_z is nonempty. This is `coherent_iff_fitting`. The proof uses
classical choice for arbitrary types; finite algorithms can enumerate the choices.
The formulation includes unused keys, whose compatibility condition is vacuous;
a nonempty operation alphabet, as in every present application, handles them.
This is a semantics of total selectors. A closed finite lookup table additionally
restricts its input domain to observed keys; missing entries are not synthesized
by the theorem.

An explicit recoding `a=q o b` transports a fitting selector on a to one on the
finer representation b (`fitting_refinement`). It establishes representability,
not improved prediction or better identification of the selector.

## 2. Conflicts require distinctions, but no canonical best quotient follows

For a labelled subrelation K of D, suppose no operation explains every member.
Then no fitting representation can give every member of K the same key. This
is `conflict_not_constant`, independent of any enumeration or scoring algorithm.

With a finite feature family F, a constant key on K means every selected feature
is constant there. Thus a necessary refinement clause asks for at least one
feature that varies on K. It is only a necessary clause; all resulting classes
must be checked again. Conflicts can require more than two observations.

**A smallest example with no feasible common coarsening.** Three observations
have compatible-operation sets `{b,c}`, `{a,c}`, `{a,b}`. Any pair can share an
operation; the triple cannot. The partition `{0}|{1,2}` and the partition
`{0,1}|{2}` are both coherent, but any common coarsening merges all three and
is not coherent. The checked witnesses and obstruction are
`ThreePoint.first_fits`, `ThreePoint.last_fits`, and
`ThreePoint.no_common_coherent_coarsening`.

Consequently there need not be a greatest coherent quotient. Compatibility does
not itself select a unique abstraction. Changing the operation language can also
change which quotients are coherent. Neither fact prevents studying useful
construction procedures under explicit assumptions.

## 3. Nonvacuous prediction and query-domain constraints

For a family H of partial programs and required inputs Q, let

```math
H_Q=\{p\in H:Q\subseteq\operatorname{dom}(p)\}.
```

`Condition` represents this family and `TotalOn` its domain requirement.
`Predicts H u y` requires BOTH a member of H and agreement that every member
returns y at u. It is intentionally stronger than the older, potentially
vacuous `Theory.Determined` predicate; that definition is preserved for its
existing uses rather than silently changed.

The checked statements establish:

- A member satisfying the domain condition remains a member
  (`condition_preserves_member`).
- Passing to a nonempty subfamily preserves an already determined answer
  (`predicts_of_subfamily`).
- If the intended program is a member and is query-total, a conditioned unanimous
  prediction agrees with it (`conditioned_prediction_correct`).

The last statement needs target membership. It does not infer membership from
fitting, totality, consensus, branch coverage or a validation score. It also says
nothing about inputs outside the stated domain.

## 4. Exact independent-key and coupled-family conditioning

For an independent selector family satisfying `h(z) in V_z`, define

```math
T_z(Q)=\{o:\forall u\in Q,\ a(u)=z\Rightarrow E(o,u)\text{ is defined}\}.
```

`independent_totality_iff` proves that a query-total selector exists precisely
when every intersection `V_z intersect T_z(Q)` is nonempty.
`attainable_operation_iff` additionally proves that, assuming global feasibility,
this intersection is the EXACT set of operations occurring at z in surviving
concrete programs. One operation must cover every query occurrence of its key.

This factorization is not valid for arbitrary coupled choices. For example,
if the family contains only `(a,a)` and `(b,b)`, projecting first yields `{a,b}`
at each key. Separate domain requirements may leave `(a,b)`, although that
program never existed. `projection_can_invent_program` checks this obstruction.

For families represented as a union of coupled branches, the correct identity is

```math
\operatorname{Condition}(\bigcup_i H_i,Q)
=\bigcup_i\operatorname{Condition}(H_i,Q).
```

This is `condition_union`: condition each WHOLE branch, then project. It applies
to shared defaults without making any particular default-selection cost an
accepted principle. No theorem here verifies the existing Python conditioner;
its independent finite tests remain empirical implementation checks.

## 5. A domain guard cannot label its unobserved branch

Let `orElse(p,q)` execute p where defined and q otherwise. If p fits D, then
`orElse(p,q)` fits D for every q (`fits_orElse`). At a new u where p is undefined,
choosing q to be constant y produces y for any requested colour (`unseen_fallback`).
These fitting, totalized alternatives need not agree. A domain test supplies an
applicability condition, not an output label. This does not invalidate conditional
synthesis; it states the evidence boundary that a construction procedure must
respect.

## Checking and scope

```sh
lake build
lake env lean tests/learning_axioms.lean
```

The core target includes all 15 statements above. The persistent
`.github/workflows/symarc-math.yml` compiles the pinned Lean version, prints their
axiom dependencies and rejects `sorryAx`; it also checks the unchanged Rust
solver. The proofs introduce no custom axioms or admitted propositions. Standard
classical choice and extensionality, where used, are foundational dependencies,
not an executable synthesis algorithm.

The integrated mathematics does NOT prove that a constructed feature is useful,
that the finite grammar contains a target, that the experimental search is
complete, or that held-out accuracy estimates unseen performance. Those are
separate claims with separate evidence. The construction experiments instantiate
coherence with a finite language and retain explicit costs and failure witnesses.
