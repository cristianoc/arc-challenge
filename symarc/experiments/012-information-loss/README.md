# 012 — Information discarded by three wrong programs

Question: for the three 008 symmetry survivors, which input relationships does
006's hand-written repair use, and what evidence for those relationships is
already present in training?

Baseline: `8a5b8a9`; original sources and labelled grids pinned by 006/inputs.json.
This is a retrospective analysis of three known repairs, not blind discovery.

Before running: inspect all training pairs for path runs versus histogram order
(7b5033c1), per-colour global component counts versus local patch lookup
(8f215267), and marker length/rank versus positional guards (97d7923e).
Record each measurable relationship, original/repair agreement, and labelled
accuracy separately for train and test. Test outputs were inspected in 006.

Construct one unlabelled disagreement witness per case using fixed searches:
path: short straight paths with repeated colours; counts: add one isolated
pixel of a frame colour outside that frame's row interval in a training input;
rank: permute pairs of complete columns in a training input. Retain the actual
grids and both predictions. These predictions are not new ground-truth labels.
Where possible demonstrate an information collision: same original decision
statistic, different repaired decision. Report failure to find one honestly.
Search deterministic lexicographic order; no tuning using test outputs.

Primary outcome: three auditable mechanism case studies, with counts of training
relationships and explicit premises needed to extrapolate them. No solver
integration criterion; no MDL score, anti-unifier, law learner or novelty claim.
Use a 12-worker pool across the three independent cases; timing is provenance,
not a performance comparison. Check provenance hashes and witness properties.
