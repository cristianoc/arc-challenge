# 021 — A shared default with explicit exceptions

Baseline: PR #32 at `11466555c84f0b8096f2f8fcf41c4122385a4c2e`.
The current executable composition from 019 remains unchanged. This experiment
compares a new operation-selection prior, not new features or output operations.

## Question

020 minimized the number of distinct operations and retained a literal-versus-
copy tie. Does representing a partial rule as one default plus exceptions help
operation identification without conflating its guard contexts? Does any gain
survive controls requiring literal outputs and the actual ARC1 task comparison?

## Frozen method

Use 018's 29 features, 19 operations, all four feature arms, minimal sufficient
feature subsets of size at most three, and unchanged internal selector ranking.
For a fitting context table, let V_z be the allowed operations at key z.
For every potential default d compute

$$c(d)=\#\{z:d\notin V_z\}.$$

Retain ALL defaults attaining minimum c. For each such default, a context it
fits uses that default; another observed context is an explicit exception whose
operation can be any member of V_z. Predict a cell only if every retained
program has a defined, identical result there. The implementation computes these
per-cell projections exactly, without choosing an arbitrary optimal program.

The default is guarded by membership in the observed key domain. Unseen keys
still abstain; undefined surviving operations still cause abstention. Therefore
there is no new automatic output for an unseen key. This isolates operation
sharing from extrapolating an unqualified fallback.

Every operation, including each literal colour, has equal cost. Each distinct
observed context contributes one exception cost, regardless of its cell count.
Duplicate observations do not add votes; splitting a context can alter this
prior. That representation dependence is a limitation, not hidden invariance.
This is a finite default/exception encoding preference, not Shannon entropy,
measured universal MDL, prior-free generalisation, or a new architecture for
conditional synthesis. The guard key remains an exact feature tuple.

Fit existence and candidate feature pools are unchanged. Only the retained
operations and consequently validation scores/model choices can change.
Relearn this operation preference inside every internal fold and rebuild search
and selection inside every outer fold. No labels outside the supplied fitting
indices may enter a table. The wrapper temporarily substitutes the learner
class in the existing experimental driver inside one worker process; this is
not thread-safe. Parallelism uses separate processes, not concurrent threads.
No stable code or prior experiment is modified.

## Controlled comparison

Keep 018's nearer/farther endpoint teachers and their exact train/test grids.
Each has demonstrations at widths 7 and 11, height 5, endpoint colours (1,2) and
(3,4). Queries have widths 5,9,13,17,21,25,29, each with (6,8) at height 5 and
(7,9) at height 7. Add two prespecified variants for EACH teacher:

- Literal fill: strictly left/right-of-middle positions use fixed colours 6/8,
  with the near/far relation swapping their assignment; ties remain 5. Endpoints
  and background copy. The output need not equal the observed endpoint colour.
- Literal background: keep input-dependent interior filling, but every row
  except the active middle row outputs constant 6, including its border cells.

There are six teachers and 84 query grids. They are researcher-constructed
labels, not new ARC annotations or independent tasks. Old two teachers are
known development controls; new literal variants explicitly test the limits of
a preference that might otherwise privilege copying. Query grids and outputs
are stored separately; write predictions before the scoring command reads them.
Compare unmodified 018 and default/exception learning. Retain all candidates,
selected guards, validation/outer results and full default certificates. No
teacher, feature, action, cost, or tie-break change after seeing outcomes.

## ARC1 comparison, independent of the controlled outcome

Run on the same 400 training and 400 evaluation tasks as 019, using its pinned
projected input and separate answer files. No score-dependent stop or tuning.
Use the retained, verified 019 stable Rust predictions and 018 predictions as
matched controls; the stable solver need not be rebuilt for a table-only study.
A fresh base-mode run must reproduce all 800 relational predictions byte-for-byte.

Report both standalone relational results and the same predeclared no-fit
fallback from 019: preserve stable when its program fits the demonstrations;
otherwise substitute a COMPLETE relational prediction, else preserve stable.
The primary comparison is existing composition versus new composition, not only
relational-alone accuracy. Report exact tasks, grids, complete wrong answers,
paired gains/losses, strict outer-gate counts, all changed task IDs, and excluding
018's three known fixtures as a separate cohort. Nothing uses test correctness
to choose which answer to emit. These are reused public datasets with known
ARC2 overlap, not an independent holdout. No ARC2 or hidden-test gain is assumed.

Twelve workers per arm, serial arms and deterministic ordering. Task-independent
batches are allowed to fit tool windows; no task may be dropped or search budget
changed. Persist and hash complete prediction files before reading answer files.
Runtimes are provenance, not speed comparisons. Preserve existing core and math.

## Checks and interpretation

Twelve synthetic tests pass before corpus or teacher execution. They include
300 comparisons with enumeration of all tiny default-and-exception programs,
literal-default identification, copy identification, multiple optima, exception
ambiguity, unknown keys, undefined survivors, duplicates, fold-label isolation,
and an explicit context-splitting dependence control.

Frozen SHA-256:
- run.py: `00e524797730b2068126004b01b5f5e7aecb1eba2050ce6c482bdd1b48d1645d`
- test_run.py: `187dd972bbdf0018e20bc2fd877ae3c0a73be88b1c87d335c64c239a17327c92`

An independent audit must verify default optimality and all per-key projections,
that each retained set is a nonempty subset of its original set, that candidate
feature pools are unchanged, and that original predicted cells never change for
a FIXED model (only abstentions can become predictions). Selected models may
change and can lose previous answers; quantify this rather than assume dominance.
Independently recompute all task/grid scores and hybrid choices. Repeating table
fits is not an independent reimplementation of the entire outer selector.

A positive outcome on old controls is not blind discovery; a gain on public
ARC1 is a measured candidate, not benchmark competitiveness. No integration
occurs automatically. If the guard error remains or the new method loses useful
answers, retain the limitation and do not tune the prior after scoring.

Conditional synthesis precedent: Alur, Radhakrishna and Udupa, Scaling Enumerative
Program Synthesis via Divide and Conquer (TACAS 2017), separately constructs
partial expressions and predicates before combining them. This study tests only
the explicitly defined default/exception preference, not that broader method.
https://www.microsoft.com/en-us/research/publication/scaling-enumerative-program-synthesis-via-divide-and-conquer/
