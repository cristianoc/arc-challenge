# 025 — Can a guarded operation repair a domain failure?

Baseline: `cf09c2e633ff76f2389140a3540301e522d0e332`, following 024.
Status lives in the experiment ledger. No stable-core or accepted-math change.

## Question and limitation identified before running

024 eliminated partial operations which cannot cover every query occurrence of
a context. Instead, construct a two-leaf expression

```
OrElse(a,b)(u) = a(u) if a(u) is defined, otherwise b(u).
```

The guard is derived from a declared operation's domain, not a guessed output.
A guard by itself supplies no label for its other branch. In particular, if a
fits every training occurrence of a key, it was defined at each of them. Every
OrElse(a,constant c) then fits those occurrences, for every colour c. At a query
where a is undefined those programs are all total but disagree. Totality alone
cannot identify the unobserved fallback. This is a finite-language obstruction,
not a claim that all program synthesis or generalisation is impossible.

## Fixed grammar and factorial

Keep the 29 input features and the <=3-feature context tuples retained by 023's
BASE learner. For each full-data or outer fold use that fold's own old candidate
pool. Do not regenerate feature search under the enlarged operation grammar:
this is a bounded repair of existing representations, not an all-expressivity
comparison. Equivalent tuples across old arms are aliases, never extra votes.

There are 19 primitive operations from 018. Eight can be partial: four adjacent
copies and four copies beyond a same-colour directional run. Construct every
OrElse(a,b) with a one of these eight, b any of the 19 primitives, and a != b.
This gives 144 new terms, 163 operations in total. No nesting, colour-specific
guards, majority rule, learned coordinate threshold or new feature constructor.
Both operand domains and values are evaluated from the input; colour 0 is a
valid result, and -1 alone denotes undefinedness.

Three operation-inference arms use the same context tuples:

- **base**: the original 19 primitives and all consistent choices;
- **unguarded-evidence control (all)**: primitives plus all fitting OrElse terms,
  whether or not each branch appeared in demonstrations;
- **observed**: retain an OrElse term only if BOTH its primary and fallback
  branches execute on at least one labelled training occurrence of that key.
  Keep every consistent primitive. Branch witnesses can be in the same grid;
  this is empirical branch coverage, not causal or independent support.

Each arm intersects the outputs of all retained operations with the labels,
keeping every surviving choice. No default prior or arbitrary preferred action.
The observed-branch restriction is an explicit additional hypothesis restriction,
not a theorem that unobserved branches are invalid. It may remove the target.

## Selection and finite-domain constraint

Recompute each representation's internal withheld-demonstration score using the
arm's operation learner. Refit action sets AND branch-support tests inside each
fold. As before, rank by complete held-out grids, sum of rational per-grid
correct-known fractions, feature count, supplied feature cost, old arm and syntax.
Inner validation uses unconditioned conservative predictions, matching 018/024.

For final and outer queries, impose 024's exact query-totality condition on each
independent per-key operation family. One operation is used at ALL occurrences
of a key. Unknown keys reject that family. Empty families give no predictions;
nonempty but disagreeing families abstain. No per-cell opportunistic fallbacks.
Select the originally best-ranked feasible model in each of base/all/observed.
Also compare **union-observed**: retain base and observed candidates, rank by the
same evidence and feature costs, then prefer base before old-arm/syntax ties.
This guards against silently deleting old candidates; it is not a no-loss theorem.
No additional selection policy or post-score tuning is permitted.

The outer test reconstructs all operation evidence and internal scores using
only remaining labelled demonstrations and the corresponding cached feature
pool. Held-out inputs alone supply domain constraints. Never use query targets,
outer labels, or full-data branch-support counts to select an outer model.

## Corpora, baseline, and outputs

All 800 projected ARC1 problems, archived stable predictions, and all 24 controlled
specifications from 023/024. The latter have 36 crossed and 14 overlapping legacy
queries; report banks separately. This is reused public and inspected controlled
data, not untouched validation. No new ARC2 result or invented ARC annotation.
Keep 400+400 primary denominators; report the established three development
exclusions separately. No-fit composition is unchanged from 019.

Persist all prediction choices and source/data hashes before separately scoring
query answers. Use 12 worker processes, deterministic order, serial batches if
needed. A batch interruption changes neither grammar, tasks nor bounds. Measure
raw and hybrid task/grid correctness, wrong complete answers, abstentions, gains
and losses, and outer-gate outcomes. Separate fixed-feature action changes from
representation reselection. Score the known f76d97a5 failure honestly; do not
supply its inspected output to fitting. Record branch witnesses for selected
conditional operations and whether they actually execute at query points.

Reuse base 023 pools, but independently compare recomputed base scores,
query-total feasibility and selected predictions with saved 024/base results.
A supplementary AFTER-SCORE oracle may check whether any concrete guarded
program matches all query labels; it must not change predictions or selections.
It distinguishes missing programs from agreement/selection bottlenecks.

## Required controls and disposition

Before scientific execution: synthetic positive cases requiring a domain guard,
literal fallback cases, an unobserved branch retaining all colour alternatives,
zero-versus-undefined, one operation shared across repeated keys, empty/unknown
contexts, duplicate grouping, and removal of held-out labels from branch evidence.
Check fitting/branch coverage and query projections against explicit enumeration
on deterministic small action tables. Compare guards with their equivalent
piecewise definition, not just a self-consistency test of the learner.

Independent audit reconstructs extended operation sets, branch witnesses,
internal scores, totality, and full/outer ranking from labelled subsets. It may
reuse the already-audited primitive feature/action evaluator but must not call
the new fit/condition/selector implementation. Repeat predictions and compare
scientific outputs exactly. Disclose reused search/extraction and finite-pool
bounds. Do not compare runtimes with full feature-search runs.

Retain as a candidate only if the measured repair is useful with an interpretable
error tradeoff. A failed repair still yields the branch-evidence obstruction and
exact counterexamples. In either case, no silent stable integration.

Conditional synthesis has established precedents. See Alur, Radhakrishna and
Udupa, TACAS 2017, *Scaling Enumerative Program Synthesis via Divide and Conquer*
(https://www.microsoft.com/en-us/research/publication/scaling-enumerative-program-synthesis-via-divide-and-conquer/).
No novelty is claimed for conditional expressions or finite hypothesis filtering.
