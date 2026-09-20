# 004-competition: Can evidence or entropy choose among competing interpretations?

Baseline `45afda6`; stable core unchanged, hash
`e76a7031af456aeba4348a32b08e32017f44641eb17256e1b01e09a5985a7f5c`.
Import the fixed 003 object grammar via a shared library; no new primitives,
segmentation rules, or copied solver. Exposing the existing inference API and
sharing the run harness must preserve 003 results.

## Opportunity before policy conclusions

Nine families: core grid DSL at depth 2, followed by the eight registered object
segmentations in 003 order. Keep all fitting syntax; each family proposes its
FIRST surviving program, preserving original enumeration order. The only
experimental decision is which family to select, not within-family search.

Run two conditions on all 400 public training tasks (12 workers, serial runs):

- **full:** discovery uses all training pairs except the last; the last pair is
  the new evidence; official task test outputs score only.
- **two-pair:** discovery uses the first training pair, the second is evidence.
  Additional available training outputs are deliberately withheld and scored
  separately. This is a controlled limited-data study, not a production solver
  evaluation using all available information.

Freeze each condition's program vocabulary before revealing the evidence output.
Pool parameters use discovery pairs plus evidence/remaining-training/test INPUTS;
no evidence output or scoring output enters the palette. This can differ from
003, which regenerates the palette on full refit. Enumerate discovery fits once,
predict evidence and scoring inputs, then retain programs matching the evidence.
No test labels or withheld training labels enter any policy.

A label-blind competitive cohort has >=2 surviving families whose representative
programs disagree at official test inputs (undefined is an explicit answer).
Report this cohort and all tasks; never select tasks for correctness or modify
the DSL to manufacture gains. Before interpreting policies, report:

- Baseline fixed-order accuracy.
- Family-choice oracle: some family's representative is correct on ALL test
  outputs. This is the actual ceiling shared by every compared policy.
- Baseline failures recoverable by family choice (opportunity), plus tasks on
  which switching could harm a correct baseline.
- Full-pool oracle separately: correctness possible by changing programs too.

Fewer than five recoverable task errors means too little improvement opportunity
for a broad policy claim; report descriptive cases and an inconclusive verdict.
Even above that threshold, this is exploratory training-corpus evidence.

## Frozen policies and entropy

1. Fixed order: grid first, then segmentations in registered order.
2. Predictive evidence: largest conditional probability of the evidence answer,
   i.e. surviving count / discovery-fit count within each family. Equal family
   priors conditional on discovery fits; ties (1e-12) use fixed order.
3. Minimum answer entropy: lowest mean per-input predictive entropy on official
   test inputs among surviving families, uniform over their surviving programs.
   Ties (1e-12) use fixed order. This explicitly tests confidence as a selector.

All three use exactly the same within-family first program and final fit pools.
Undefined predictions contribute to entropy but do not score correct. Empty
families cannot be selected. No surviving family means abstention.

Report initial/final family counts, disagreement counts, and family entropy
before/after the observed evidence. Initial mass is uniform over discovery-viable
families; posterior family mass is proportional to evidence probability. Entropy
can rise for a particular observation; do not equate its decrease with truth.
Eight object family labels need not denote eight distinct observed partitions.
Equal family prior is explicit, not representation-independent.

A fixed sensitivity check changes the grid program prior only to normalized
length mass q(l) ∝ 2^-l, uniform within K^l programs. Report evidence-policy
choice changes; no extra policy tuning. AST priors remain uniform.

## Cost, checks, reporting and disposition

24 hash-selected pilot tasks, followed by full 400 if <120s and <4GiB peak RSS;
600s limit per process. Exhaustive classes, no search fallback, symmetry or caps.
Both conditions/policies share their enumerated pools. The run emits its report
directly, including per-task choices, opportunity, correctness, and witnesses
for every recoverable error. Save compact JSONL family diagnostics/predictions,
not another copy of core or object programs.

Tests: shared 003 tests and byte-for-byte pilot pool comparison; evidence
conditioning keeps only original fits; labels used only in scoring; fixed tie
handling; undefined entropy; synthetic examples where changing families can
help or harm, ensuring the ceiling differs from the full-program oracle.

Retain a selector as an integration candidate only if meaningful opportunity
exists and it yields more wins than losses in the full-data competitive cohort.
Otherwise close without integration; keep the 003 language candidate separate.

Commands from `symarc/`:

```
cargo test --release --manifest-path experiments/004-competition/Cargo.toml
python3 experiments/004-competition/run.py pilot
python3 experiments/004-competition/run.py full
```
