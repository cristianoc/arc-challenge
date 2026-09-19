# 007 path-order: full

400 tasks; 12 workers; 3.029s including depth-3 audit.

Training-only enumeration; test labels used only for scoring. Public development data; known ARC-AGI-2 witness is not a fresh holdout.

| Arm | Fitting tasks | Correct tasks | Correct test grids | Defined test grids | Oracle tasks |
|---|---:|---:|---:|---:|---:|
| grid d2 | 55 | 53 | 57 | 59 | 54 |
| 003 objects | 17 | 16 | 16 | 17 | 16 |
| histogram | 1 | 1 | 1 | 1 | 1 |
| path | 0 | 0 | 0 | 0 | 0 |
| control: grid/objects/histogram | 63 | 60 | 64 | 67 | 61 |
| treatment: control/path | 63 | 60 | 64 | 67 | 61 |
| secondary: path/control | 63 | 60 | 64 | 67 | 61 |

Treatment wins/losses vs control: 0/0. Path-first wins/losses: 0/0. New oracle tasks: 0.

## Path-fitting tasks

| Task | Histogram fits | Path fits | Distinct path predictions | Path selected correct | Path oracle | Control oracle | First path | Correct path witness |
|---|---:|---:|---:|---|---|---|---|---|

## Depth-3 audit

| Task | d3 fits | d3 oracle | Path oracle |
|---|---:|---|---|

Histogram+path candidate ASTs: 575520; actual training example checks: 576192. Summed task seconds (overlapping workers): 34.062. Counts are not equal-cost primitive-operation budgets.

No post-run grammar tuning or test-label-driven selection. A gain means coverage within these bounded classes, not automatic discovery of graph concepts or an all-depth separation.
