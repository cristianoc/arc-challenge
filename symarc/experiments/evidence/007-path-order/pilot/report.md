# 007 path-order: pilot

24 tasks; 12 workers; 0.417s including depth-3 audit.

Training-only enumeration; test labels used only for scoring. Public development data; known ARC-AGI-2 witness is not a fresh holdout.

| Arm | Fitting tasks | Correct tasks | Correct test grids | Defined test grids | Oracle tasks |
|---|---:|---:|---:|---:|---:|
| grid d2 | 3 | 2 | 2 | 3 | 3 |
| 003 objects | 1 | 0 | 0 | 1 | 0 |
| histogram | 0 | 0 | 0 | 0 | 0 |
| path | 0 | 0 | 0 | 0 | 0 |
| control: grid/objects/histogram | 4 | 2 | 2 | 4 | 3 |
| treatment: control/path | 4 | 2 | 2 | 4 | 3 |
| secondary: path/control | 4 | 2 | 2 | 4 | 3 |

Treatment wins/losses vs control: 0/0. Path-first wins/losses: 0/0. New oracle tasks: 0.

## Path-fitting tasks

| Task | Histogram fits | Path fits | Distinct path predictions | Path selected correct | Path oracle | Control oracle | First path | Correct path witness |
|---|---:|---:|---:|---|---|---|---|---|

## Depth-3 audit

| Task | d3 fits | d3 oracle | Path oracle |
|---|---:|---|---|

Histogram+path candidate ASTs: 34464; actual training example checks: 34464. Summed task seconds (overlapping workers): 2.627. Counts are not equal-cost primitive-operation budgets.

No post-run grammar tuning or test-label-driven selection. A gain means coverage within these bounded classes, not automatic discovery of graph concepts or an all-depth separation.
