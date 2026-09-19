# 013 — Minimal sufficient distinctions

## Question

When two programs fit all observed examples but disagree elsewhere, can we
characterise the difference as an unjustified identification made by one
program's abstraction, and can training data supply positive evidence for which
identifications are safe?

This experiment starts from the three 012 mechanism cases. It does **not** use
their official test outputs for discovery or scoring. The goal is to turn the
retrospective "information loss" observation into a prospective criterion that
can later be evaluated on untouched tasks.

## Mathematical object

Let (X) be inputs and (Y) outputs. An abstraction is a map

[
a : X \to A.
]

It induces a kernel equivalence

[
x \equiv_a x' \quad\Longleftrightarrow\quad a(x)=a(x').
]

A predictor (p:X\to Y) **factors through** (a) when there is some
(h:A\to Y) such that

[
p = h\circ a.
]

Equivalently,

[
x\equiv_a x' \implies p(x)=p(x').
]

Thus the kernel of (a) is exactly the set of distinctions that every program
factoring through (a) is forced to forget.

For a training specification (D\subseteq X\times Y), call (a)
**training-sufficient** when no two labelled training inputs collide with
different outputs:

[
(x,y),(x',y')\in D \land a(x)=a(x') \implies y=y'.
]

This is deliberately weak. With sparse examples almost any coarse abstraction
can be training-sufficient.

## Distinguishing witnesses

Let (a) and (b) be two abstractions or decision statistics. A pair
(x,x') is an (a/b) distinguishing witness when

[
a(x)=a(x') \qquad\text{and}\qquad b(x)\ne b(x').
]

If candidate (p_a) factors through (a), then necessarily
(p_a(x)=p_a(x')). A candidate using (b) may distinguish them.

This gives a purely structural test: a witness proves an expressivity boundary
of the abstraction without assigning a ground-truth output to the synthetic
input.

The three 012 cases instantiate it as follows.

| Task | Coarse statistic (a) | Retained distinction (b) |
|---|---|---|
| 7b5033c1 | colour histogram + first-seen order | path traversal sequence |
| 8f215267 | local frame-adjacent patch | global same-colour component collection |
| 97d7923e | positional guards | marker-relative rank relation |

The first two are literal information-loss cases. The third is different: the
raw features are available, but the decision rule uses the wrong relation.
Keep these categories separate.

## The missing condition: justified forgetting

A coarse kernel is useful only if the training evidence supports treating its
members alike. Mere absence of contradiction is not support.

For an intervention family (I), write (x' = i(x)) for an admissible
counterfactual transformation. An observed pair of examples provides positive
support for forgetting the distinction changed by (i) when the examples
instantiate the same transformation pattern and the outputs transform as the
candidate law predicts.

The critical separation is:

1. **compatibility** — no observed example refutes the identification;
2. **support** — observed examples vary the supposedly irrelevant distinction
   while preserving the relevant relation;
3. **determination** — the supported identifications suffice to force a unique
   prediction.

Experiment 009 measured mostly (1). Experiment 010 searched for failures of a
particular form of (2). The present experiment asks how to represent (2)
locally, below whole-grid symmetry.

## Prospective protocol

### Phase A — freeze a small relation language

Before inspecting any new holdout outputs, define a finite vocabulary of local
relations and interventions:

- object translation within a containing region;
- permutation of peer objects;
- colour renaming within detected roles;
- path-preserving recolouring;
- insertion/removal of remote peer objects;
- permutation of columns/rows that preserves an explicitly detected
  marker-to-object relation.

Each operation must have a syntactic applicability predicate based only on the
input. No task-specific constants or exemptions may be added after evaluation.

### Phase B — infer candidate kernels from programs

For each training-fitting program in a bounded corpus, instrument its decisions
to record the statistic actually consumed at each output decision. Two inputs
that produce the same recorded statistic are provisionally identified.

For hand-audited cases, use the already known statistics only as development
fixtures. The evaluation corpus must infer them automatically or use a frozen
feature language.

### Phase C — search for kernel witnesses

For each provisional identification (x\equiv_a x'), search admissible
interventions that preserve (a) while changing one alternative relation
(b). Record:

- whether a witness exists;
- intervention cost;
- which candidate programs are forced to agree;
- which candidates distinguish the witness;
- whether the changed relation was varied independently anywhere in training.

Synthetic witnesses remain **unlabelled**. Their purpose is to expose what a
candidate cannot represent, not to declare another candidate correct.

### Phase D — measure training support

For every distinction erased by a candidate, count independent training
transports that vary that distinction while preserving the proposed output
relation. Use task-level counts, not multiple cells from one grid, as the
primary support unit.

A candidate is never promoted merely because its kernel is larger. Larger
kernels compress more and also make stronger unsupported claims.

### Phase E — blind evaluation

Freeze the relation/intervention language and selection rule, then evaluate on
tasks whose test outputs were not inspected during development.

Primary comparison:

- ordinary shortest fitting program;
- maximum compatible forgetting;
- maximum **supported** forgetting;
- conservative rule that retains all candidates when support is insufficient.

Report exact test accuracy, abstention/underdetermination, and the frequency of
unsupported kernel identifications.

## Development predictions from 012

These are sanity checks, not evaluation claims.

### 7b5033c1

The histogram program identifies paths with equal colour counts and first-seen
order. The existing witness (1,1,2) versus (1,2,1) lies in one kernel class
of that abstraction but not of the path-sequence abstraction.

Training does not vary repeated-colour path order, so the finer distinction is
not selected by direct contrast. The expected outcome is **underdetermined**,
not "path wins". This is an important control against smuggling the known test
answer into the criterion.

### 8f215267

The local program identifies inputs that agree on the frame-adjacent patch.
Adding a remote same-colour object leaves that statistic unchanged while
changing the global count.

Training contains nine frames whose intended stripe counts agree with global
same-colour object counts. The prospective question is whether there is genuine
independent variation of remote objects sufficient to support global scope, or
whether all nine observations are confounded by layout. Count this explicitly.

### 97d7923e

This should **not** be forced into an information-loss story. Column
permutation preserves marker length and rank structure while breaking the
position guard. The relevant object is a relation over already retained
features. Treat this as rule invariance rather than kernel refinement.

## What would count as a result?

A positive result is not that the known repairs survive more synthetic tests.
It is one of:

1. training-supported kernel evidence rejects fitting programs that later fail
   on blind tests more often than it rejects correct ones;
2. retaining distinctions until there is positive evidence to quotient them
   improves calibration/abstention even when exact accuracy does not improve;
3. the method reliably reports underdetermination on development cases such as
   7b5033c1 where training does not distinguish the competing explanations.

A negative result is equally useful: if almost no ARC task contains independent
variation supporting quotient decisions, then sparse-example ARC does not
supply enough evidence for this notion of generalisation by itself; stronger
priors or cross-task knowledge are necessary.

## Relation to prior work

This resembles abstraction refinement only superficially. In CEGAR and
abstraction-refinement synthesis, a known specification or concrete semantics
identifies a spurious abstract counterexample and tells the algorithm how the
abstraction is too coarse. Here the target function is unknown outside finitely
many examples. The central problem is therefore **evidence for refinement or
quotienting**, not refinement once a counterexample is known.

It is also related in spirit to invariant prediction: variation across
environments or interventions can supply evidence that a predictive relation
is stable. We use the analogy only as motivation; ARC examples do not arrive
with a causal model or intervention semantics.

## Decision rule for this experiment

Do not integrate a new solver heuristic yet. First implement the witness and
support measurements on the three 012 development cases and on a frozen sample
of training-only tasks. If the measurements collapse to compatibility counts or
require task-specific intervention choices, close the experiment as a negative
result.
