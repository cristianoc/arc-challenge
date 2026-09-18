# Task subsets

Two sets:

* **full**: all 800 public ARC tasks, `full.txt` (400 training + 400
  evaluation). Use `THREADS=12 ./bench.sh full` for a full-pipeline run.
* **quick**: 40 tasks, `quick.txt`, a stratified sample of the full set.
  Runs in about 7 seconds with one worker on this laptop.

Each line is `id split stratum`. Run with

```sh
mkdir -p out
./target/release/symarc --root ../data --tasks-file subsets/quick.txt > out/quick.txt
python3 subsets/estimate.py out/quick.txt                                  # full-set estimates
python3 subsets/estimate.py out/quick.txt out/full.txt   # against an actual full run
```

## Why stratified

The work is about the symmetry filters. Their metrics (rejected generators,
equivariance at the test input, determination, symmetry-only answers) are
only defined or only interesting on the tasks where some program fits the
data, which is 77 of 800. A uniform sample of 40 would contain about four of
those. So tasks are sampled by stratum and full-set counts are estimated by
reweighting: a task in stratum `s` counts `size(s) / sample(s)` times.

| stratum | meaning | size | in quick | weight |
|---|---|---|---|---|
| A1_fitted_edge | a program fits the data and either is wrong on a test or the survivors disagree at a test input | 9 | 9 | 1 |
| A2_fitted | a program fits the data, no edge behaviour | 68 | 13 | 5.2 |
| B_unfit_symreach | no program fits; a test input lies inside the closure | 26 | 5 | 5.2 |
| C_unfit_funcrejected | no program fits; functionality rejected at least one candidate generator | 562 | 9 | 62.4 |
| D_unfit_nothing | no program fits; functionality rejected nothing | 135 | 4 | 33.8 |

Within A2 the sample covers each program family (geometric, tiling, crop,
colour, split, painting) in both splits. Strata are in `strata.json`, which
`estimate.py` reads.

## Interpretation

The strata describe the default depth-2 solver, with 77 fitted tasks among
800. Zero variance on a variable used to define a stratum is by construction,
not evidence of out-of-sample predictive accuracy. Use this sample to compare
changes, then confirm promising effects on the full corpus.

Changes within the fitted strata carry modest weights (1 and about 5).
Changes in the unfit strata can carry weights as high as 62 and have high
variance. A quick-set improvement is not a precise full-set solve-rate claim.
Repeated selection on this fixed sample can overfit it.

A2 is balanced by program family rather than sampled uniformly. The reported
standard errors use a stratified random-sampling formula and are approximate
diagnostics, not calibrated confidence intervals. The estimator requires the
complete quick set and rejects partial runs.

## When to rebuild it

The strata depend on which tasks a program fits, which depends on the DSL and
the enumeration depth. If either changes substantially, rerun the full set once
and rebuild the strata with `python3 subsets/build.py out/full.txt` (deterministic, seed 7). Until then, results on the quick set for a changed DSL
should be read as "within the tasks the old DSL could fit".
