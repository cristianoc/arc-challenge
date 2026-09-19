# 008: manually chosen symmetries invalidate seven known incorrect solutions

**What we showed:** after inspecting the tasks and their test answers, we could
manually choose plausible task-specific symmetry requirements that seven of ten
known incorrect programs violate, while their hand-written repairs satisfy them.
The result is conditional on accepting those requirements as properties of the
intended task.

I chose the requirements from a small menu of familiar ARC-inspired
transformations: colour renaming, horizontal reflection and transposition.
These are not universally valid invariants established by ARC's declared priors.
The experiment did not discover them, justify their applicability from training
pairs alone, or derive them from a study of Kaggle methods. Choosing which ones
apply already supplies part of our interpretation of the intended solution.

## What was implemented and run

A Python runner compared ten existing original programs with ten existing
hand-written repairs from audit 006. Both versions fit all 25 supplied training
examples. The runner transformed training inputs and their expected outputs
according to the manually supplied requirements, executed both programs, and
checked exact agreement. It performed 854 colour-swap checks and seven geometry
checks per candidate family. It did not synthesise or repair programs.

**Results:** colour swaps reject five originals; geometry checks reject two
additional originals. All ten repairs pass. Three originals also pass everything:
histogram instead of path order, local instead of global counting, and
position-specific guards instead of rank selection. This is a retrospective
7/10 result within selected pairs, not a blind repair or generalisation score.

## Exactly which assumptions were supplied

A common menu was used, but **the same invariants were not applied to every
task**. These choices were made manually during the earlier retrospective audit:

| Task(s) | Colours held fixed during swaps | Geometry checked |
|---|---|---|
| `1ae2feb7` | 0 | Horizontal reflection |
| `135a2760` | 0 | Transposition |
| `221dfab4` | 0, 3, 4 | Transposition |
| `e3721c99` | 0, 5 | None |
| `6ffbe589`, `7b5033c1`, `8f215267`, `97d7923e`, `a251c730`, `dbff022c` | 0 | None |

Every pair of colours outside the fixed set was swapped. The rationale was that
some colour names appeared interchangeable, while background or designated
marker/target colours had specific roles; geometry was selected where the
proposed interpretation appeared orientation-independent. These are task-specific
judgments informed by inspected answers, not consequences established by the
training data. A repair passing a chosen requirement does not justify that
requirement: using it as justification would be circular.

For a chosen transformation T, a training pair (x,y) generated the expected pair
(T(x),T(y)). Calling a mismatch an error on that new pair assumes the intended
rule respects T. Removing test outputs from the runner does not undo their
influence on the prior manual selection of repairs and requirements. Likewise,
registering the execution protocol before running it did not make these choices
blind.

The remaining research problem is **how to select and justify an applicable
transformation law without already knowing the intended solution**. This
experiment tested the consequences of supplied laws, not that discovery problem.
Neither anti-unification nor MDL was implemented or evaluated.

[Direct report](report.md) · [per-task counts and counterexamples](results.json) ·
[manifest](run.json) · [original registered protocol](protocol.md).
The report and protocol are retained unchanged as execution records; this note
clarifies their interpretation.

## Execution details

Implementation and protocol were committed at
`b7c96c9dae9e686be42eb648e01ac706446cf4a9` before execution. Both candidates fit
all 25 original training pairs. Each was then checked on 854 single colour swaps
and seven geometry transformations. Including the 25 identity cases, there were
886 probes per candidate family, representing 671 distinct labelled pairs across
tasks; 215 nonidentity operations act identically. No exception or timeout
occurred. Twelve processes completed the functional diagnostic in 1.93 seconds;
this is not a solver timing comparison. Transform controls passed. Sources and
full inputs were checked against 006's hashes, then test pairs were discarded
before workers executed. Core, object library and accepted mathematics unchanged.

Candidates, task selection and symmetry assumptions came from 006, whose test
answers were inspected. Repaired candidates had already passed related random
symmetry checks. Exhaustive single-swap testing and testing the originals are
new here; successful repaired-program symmetry is not fresh independent evidence.
No candidate was discovered or edited during this study. Single swaps generate
the permutation group, but checking them only on the original examples does not
check every composition on every transformed example.

| Task | Extra constraint rejecting original | What the surviving repair adds |
|---|---|---|
| `1ae2feb7` | Colour renaming and horizontal reflection | Barrier role and source-side binding |
| `135a2760` | Transposition | Orientation and complete frame interior |
| `221dfab4` | Transposition | Marker-relative phase and orientation |
| `dbff022c` | Colour renaming | Input-derived directed legend |
| `e3721c99` | Colour renaming | Hole-count legend and different connectivity |
| `a251c730` | Colour renaming | Templates extracted from the input |
| `6ffbe589` | Colour renaming | Input marker counts interpreted as rotations |

Rejecting an original does **not** establish that every part of its repair is
necessary or discovered by that constraint. For example, rejecting row-only
processing by transposition does not determine the correct frame extent.
006 already documented that orientation normalisation alone left two test cells
wrong for `135a2760`. Similarly, a colour-renaming counterexample does not itself
infer a legend reader, its direction, connectivity, or a template algorithm.
Many untested alternatives could survive the same constraints.

Concrete rejection witnesses are retained with transformed input, expected
output, actual output and operation. For example, swapping colours 1 and 2 in
training example 0 of `1ae2feb7` makes the original wrong on 20 cells. Transposing
example 0 of `221dfab4` makes it wrong on 117 cells. These expected outputs come
from the assumed transformation law, not from another independently labelled
example. Accepting that law is the substantive premise.

## The unresolved cases identify different missing evidence

| Task | Why these probes fail to separate the pair | Candidate distinguishing intervention (not executed here) |
|---|---|---|
| `7b5033c1` | Both histogram and path rules are colour-equivariant; renaming never introduces noncontiguous repeats | Supply a labelled path in which a colour recurs after another colour. 007's synthetic check verifies this mechanism. |
| `8f215267` | Renaming preserves the association between local patches and global counts in the supplied examples | Move a separately detected instruction object away from a frame's local patch, preserving global per-colour counts. Require the output to remain unchanged. |
| `97d7923e` | Renaming changes neither absolute positions nor the correlation of guard conditions with ranks | Reorder separated bars while preserving cap-colour groups, heights and marker lengths; require the chosen kth bar and output positions to follow the permutation. |

The last two interventions require scene extraction and admissibility conditions:
objects must stay separate and in the instruction region; bars must not merge,
markers must remain distinct, and tied heights need a policy. They are research
proposals, not verified new witnesses. Merely naming their intended invariants
would assume part of the abstraction we want to discover.

The partial symmetry task `0934a4d8` and unresolved assembly task `5dbc8537` from
006 are excluded from the ten-pair measurement because there is no complete
verified repaired candidate. Their uncertainty remains: existing reflection
constraints leave eight cells unresolved in the former; object extraction and
placement rules remain unspecified in the latter. They are not silently counted
as successes or failures of this selector.

## Precise interpretation

Let D be the supplied labelled pairs and C an explicitly assumed collection of
input/output transformation pairs (T_in, T_out). Define

    D_C = D ∪ {(T_in(x), T_out(y)) : (x,y) ∈ D, T ∈ C}
    V_C = {P : for every (x,y) ∈ D_C, P(x) is defined and equals y}.

The experiment computes membership in V_C for two fixed candidates per task.
The ordinary training version space contains both. Additional contracts shrink
that space, sometimes to one of these two, sometimes leaving both. Shrinking it
is useful only insofar as the contracts are valid for the intended task. The
labels in D_C encode those assumptions; they do not add an independent oracle.

For a full group action G, if two programs are equivariant on the relevant
orbits and agree on D, they agree on G·D: for each g and training input x,

    P(g·x) = g·P(x) = g·Q(x) = Q(g·x).

Thus orbit augmentation cannot distinguish them. It can expose a violation of
the proposed invariance, but cannot reveal a distinction that stays outside
those orbits. This is an elementary conditional observation, not a new theorem.
Our finite probes do not prove global equivariance. Histogram versus path gives
a concrete obstruction: colour renaming preserves both counting and traversal,
and cannot change which positions carry equal colours. A noncontiguous repeat
changes that equality pattern, hence reaches evidence beyond mere renaming.

This also clarifies two different uses of “generalisation”: syntactic
anti-unification enlarges a substitution-instance family, whereas these
constraints reduce the set of admissible closed programs. Neither operation
alone guarantees intended test behaviour. They can be complementary: propose a
parameterised family, then constrain how its parameters depend on the input.
No anti-unifier was run here.

MDL can rank the remaining candidates only relative to a specified coding
language and prior. No MDL code was supplied or tested. If several programs
survive the contracts, a preference for one remains a preference—not a logical
consequence of those constraints. This study locates that residual ambiguity
instead of declaring the shortest, most symmetric, or most confident program
necessarily correct.

## Disposition and reproduction

Closed diagnostic; no integration. Completed runner removed, runnable history
retained at `b7c96c9`. On that revision, fetch 006's pinned inputs, then from repo
root run:

```sh
ARC_REPAIR_CORPUS=/absolute/path/to/corpus.json timeout 600 python3 \
  symarc/experiments/008-repair-constraints/run.py \
  symarc/out/experiments/008-repair-constraints
```

Requires Linux/POSIX alarms and Python; no extra packages. The manifest records
candidate/input hashes, interpreter, revision and clean working tree. A rerun's
wall time and environment will vary. The next bounded study should construct
and validate the proposed structural interventions on the three surviving pairs,
then test original and repaired programs without treating repaired outputs as
independent truth. That is a targeted next question, not a claim that general
abstraction discovery is solved.
