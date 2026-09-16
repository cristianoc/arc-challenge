# Generalization and the discovery of inductive structure

*Research note — 16 September 2026*

**Abstract.** We study generalization through an order on explanations and a
consequence operation determined by a representation. Evidence removes
inconsistent explanations; abstraction changes which explanations can be
related. Finite experiments separate these operations and exhibit both
undergeneralization and overgeneralization. An ARC case study supplies two
incomparable explanations consistent with every official example, a six-cell
witness distinguishing them, and an incremental learner whose final
representation is independent of example order although its discovery path is
not. The account specifies a structural object that learning discovers; it
does not yet define a measure of discovery difficulty.

## 1. Explanations and generality

Let an example be an input–output pair, and let $E$ be a finite set of
examples. Fix a representation $R$, a finite set $H_R$ of admissible
explanations, and a preorder $\preceq_R$, oriented so that
$h\preceq_R k$ means that $k$ is at least as general as $h$.
Identifying mutually comparable explanations gives a partial order. Define

$$
V_R(E)=\{h\in H_R:h\text{ agrees with every example in }E\},
\qquad
F_R(E)=\operatorname{Max}_{\preceq_R}V_R(E).
$$

The **generalization frontier** $F_R(E)$ can contain several explanations.
For example, selecting the largest object and selecting the leftmost object
may both explain the data. A rule requiring both properties specializes each;
removing that rule leaves the ambiguity between the other two intact.

A useful explanation form is $(t,\Phi)$: a parametric term together with a
constraint on its parameters or applicability. Within a common schema,

$$
(t,\Phi_1)\preceq_R(t,\Phi_2)
\quad\text{if}\quad
\Phi_1\Rightarrow\Phi_2.
$$

This is assumption weakening. It orders candidate claims but does not prove
them: validity under $\Phi_1$ does not imply validity under $\Phi_2$.
The choice of admissible weakenings is an inductive commitment. Observed
variation can refute a fixed parameter value; extrapolating to unobserved
values requires additional structure. In particular, an expression's failure
to mention a feature does not establish independence from that feature.

## 2. Constructing a representation

Generality can also arise through substitution. A term $T$ generalizes
$t_1,\ldots,t_m$ when substitutions $\sigma_i$ satisfy

$$
t_i=T\sigma_i \qquad(1\leq i\leq m).
$$

Anti-unification constructs such a shared term. For instance,

$$
\operatorname{Flip}(X,2),\quad
\operatorname{Flip}(X,3),\quad
\operatorname{Flip}(X,4)
\quad\leadsto\quad
\operatorname{Flip}(X,n).
$$

The resulting variable needs a domain. In our finite experiment that domain
is explicitly $\{2,3,4,5\}$; predicting size 5 uses this choice as well as
the discovered schema. Anti-unification alone does not justify unrestricted
substitution.

A representation can also erase a relevant distinction. Abstracting the
action gives $\operatorname{Flip}(a,n)$; separating it into
$\operatorname{Flip}(X,n)$ and $\operatorname{Flip}(Y,n)$ restores the
action distinction while retaining size abstraction. The toy implementation
models this split explicitly. Thus representation learning comprises both
merging instances into schemas and splitting overly broad schemas.

## 3. Consequences of a fixed representation

To give a precise extensional model, fix a finite universe $U$ of ground
explanatory instances. These are distinct from the input–output examples of
Section 1: interpreting examples as such instances is itself part of the
representation. Let $\mathcal C_R\subseteq\mathcal P(U)$ contain $U$
and be closed under intersections. For observed instances $S\subseteq U$,
define

$$
\operatorname{cl}_R(S)
=\bigcap\{C\in\mathcal C_R:S\subseteq C\}.
$$

**Proposition.** This operation is extensive, monotone, and idempotent:

$$
S\subseteq\operatorname{cl}_R(S),\qquad
S\subseteq S'\Rightarrow
\operatorname{cl}_R(S)\subseteq\operatorname{cl}_R(S'),\qquad
\operatorname{cl}_R(\operatorname{cl}_R(S))=\operatorname{cl}_R(S).
$$

*Proof.* Every intersected set contains $S$. Enlarging $S$ reduces the
family being intersected. Finally, its intersection belongs to
$\mathcal C_R$, so closing it again leaves it unchanged. ∎

For an intended concept $T\subseteq U$ and observations $S\subseteq T$,
inductive soundness means $\operatorname{cl}_R(S)\subseteq T$, completeness
means $T\subseteq\operatorname{cl}_R(S)$, and exactness means equality.
The target is known in the synthetic experiment, not generally to a learner.

Take $U=\{\operatorname{Flip}(a,n):a\in\{X,Y\},\ n\in\{2,3,4,5\}\}$,
with $S$ the three displayed $X$-instances and $T$ all four
$X$-instances. The implemented representations give

$$
\operatorname{cl}_{\rm ground}(S)=S
\subsetneq
\operatorname{cl}_{\rm object}(S)=T
\subsetneq
\operatorname{cl}_{\rm indiscriminate}(S)=U.
$$

More closure is therefore not a quality criterion. Moreover, adding a
concept to an intersection-based representation can only shrink its closure;
inducing stronger generalization may require removing finer concepts. The
object model does precisely this relative to $\mathcal P(U)$.

Closure describes consequences within a fixed representation. Learning
changes the representation, producing transitions

$$
(R_i,S_i)\longrightarrow(R_{i+1},S_{i+1}).
$$

Even when $S_i\subseteq S_{i+1}$, the corresponding closures need not grow:
a representation change can retract a previous generalization. This closure
model and the frontier model describe complementary aspects of induction;
we have not established a general construction identifying them.

## 4. A distinguishing ARC example

ARC task `7e0986d6` has two official training pairs and one test pair. Consider
two rules, evaluated using orthogonal adjacency in the original input:

* **Component rule $s$.** Partition nonzero cells into monochromatic
  connected components. Components of size at least four are references.
  Recolor each smaller component to the color of a reference touching at
  least two distinct reference cells; otherwise erase it to background.
* **Local rule $\ell$.** With two foreground colors, take the more frequent
  as reference. Recolor each cell of the other color independently when it
  has at least two reference-colored neighbors; otherwise erase it.

The cases discussed here have unambiguous reference choices. The component
implementation selects the first qualifying reference if several exist.
The measured results are:

| Explanation | Both training pairs | Official test |
|---|:---:|:---:|
| Component rule | Fits | Fits |
| Component repair with frequency-role assumptions | Fits | Fits |
| Repair with colors fixed to the first training pair | Fails | Fails |
| Local rule | Fits | Fits |

Under the declared schema order, the frequency-constrained explanation is
dominated by component repair. This comparison concerns the guarded schema,
not an inclusion order on arbitrary total programs. Component repair and the
local rule remain incomparable.

An enumerated witness separates them. Here `R` and `n` are distinct nonzero
colors, and `.` is background:

```text
input       s(input)    ℓ(input)
RRR         RRR         RRR
Rnn         RRR         RR.
```

The nuisance component touches three reference cells collectively; its
rightmost cell touches only one individually. Four reference cells and two
connected nuisance cells are required by the search's admissibility
conditions. This witness attains that six-cell lower bound and the minimum
possible area. Minimality is relative to those conditions.

Labeling the witness with $s$'s output refutes $\ell$. The label is
constructed from $s$, so this demonstrates discrimination between the
rules, not independent evidence that $s$ is the uniquely intended rule.

## 5. Endpoint and discovery path

Let $A,B$ be the official training pairs and $W$ the labeled witness.
The incremental learner records three commitments: fixed or parametric color
roles, individual cells or connected components, and retention or removal of
the frequency assumption. Differing color roles trigger parameterization;
failure of the local rule, provided the component rule fits, triggers
component structure; that structure then triggers frequency-assumption
removal by the declared preference rule. The repair operations are supplied
to the learner.

All six permutations of $(A,B,W)$ reach the same represented explanation:
parametric colors, connected components, and no frequency assumption.
Historical fields still differ. This is order independence of the final
representation and its semantic key for this experiment, not a general
confluence theorem. Component structure is discovered exactly when $W$
arrives, at step 1, 2, or 3.

For a fixed vocabulary of representation changes, let
$\operatorname{Rev}_I(e)$ denote the changes triggered by example $e$
in learner state $I$. A possible comparison of examples is

$$
e\preceq_I e'
\quad\Longleftrightarrow\quad
\operatorname{Rev}_I(e)\subseteq\operatorname{Rev}_I(e').
$$

After $A$, example $B$ triggers color parameterization, whereas $W$
also triggers component structure and assumption removal. This comparison is
relative to the learner, its state, and the chosen change vocabulary.

The work separates three questions: which explanations evidence permits,
which consequences a representation supports, and how that representation
is discovered. An eventual connection to epiplexity would need to relate a
measure of learnable information under computational constraints to these
discovery paths. No such measure or equivalence has been established here.

## Reproduction

From the repository root, run the following dependency-free experiments.
Their assertions check the finite closure model, toy schema construction,
ARC fit, witness, and six curricula respectively.

```sh
python3 miniARC/representation_closure.py
python3 miniARC/antiunify_learning.py
python3 miniARC/real_arc_generality_7e0986d6.py
python3 miniARC/distinguish_7e0986d6.py
python3 miniARC/confluence_7e0986d6.py
```
