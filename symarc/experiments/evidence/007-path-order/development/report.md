# 007 path-order: development

1 tasks; 12 workers; 0.038s including depth-3 audit.

Training-only enumeration; test labels used only for scoring. Public development data; known ARC-AGI-2 witness is not a fresh holdout.

| Arm | Fitting tasks | Correct tasks | Correct test grids | Defined test grids | Oracle tasks |
|---|---:|---:|---:|---:|---:|
| grid d2 | 0 | 0 | 0 | 0 | 0 |
| 003 objects | 0 | 0 | 0 | 0 | 0 |
| histogram | 1 | 0 | 0 | 1 | 0 |
| path | 1 | 1 | 1 | 1 | 1 |
| control: grid/objects/histogram | 1 | 0 | 0 | 1 | 0 |
| treatment: control/path | 1 | 0 | 0 | 1 | 1 |
| secondary: path/control | 1 | 1 | 1 | 1 | 1 |

Treatment wins/losses vs control: 0/0. Path-first wins/losses: 1/0. New oracle tasks: 1.

## Path-fitting tasks

| Task | Histogram fits | Path fits | Distinct path predictions | Path selected correct | Path oracle | Control oracle | First path | Correct path witness |
|---|---:|---:|---:|---|---|---|---|---|
| 7b5033c1 | 20 | 20 | 2 | true | true | false | modal-4-mono / All / Path4 / first / column | modal-4-mono / All / Path4 / first / column |

## Depth-3 audit

| Task | d3 fits | d3 oracle | Path oracle |
|---|---:|---|---|

Histogram+path candidate ASTs: 1440; actual training example checks: 1484. Summed task seconds (overlapping workers): 0.036. Counts are not equal-cost primitive-operation budgets.

No post-run grammar tuning or test-label-driven selection. A gain means coverage within these bounded classes, not automatic discovery of graph concepts or an all-depth separation.
