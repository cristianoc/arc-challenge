# 011: unchanged SymArc configurations on ARC-AGI-2

Registered before execution. Reproduce ARC-AGI-1 counts, then report ARC-AGI-2
without tuning. Baseline `2383df6`: stable library and retained 003 object API.
Complete solver uses Config::default(), seed 0, as the earlier 800-task stable
baseline. Separately compare exhaustive grid depth 2, 003 objects, and grid-first
object fallback exactly as in previous expressivity studies. These are distinct
configurations: object fallback is not integrated into the complete solver.
No new primitives, selection rules or symmetry assumptions.

Datasets: repository ARC1 data (400 training + 400 evaluation); official ARC2 at
`f3283f727488ad98fe575ea6a5ac981e4a188e49` (1000 training + 120 evaluation).
ARC2 training includes old tasks, so report split counts and data overlap rather
than treating it as an independent test set. Public tasks inspected in previous
research are not a clean holdout. The 120 external task-specific Python solvers
from 006 were already evaluated on ARC2; they are not SymArc.

One standalone Rust adapter imports both existing libraries; no solver copying.
Fit each task independently using its training pairs and test inputs; test labels
score predictions only. Record training fit, whole-task and grid exact match;
for exhaustive pools also oracle correctness (one fitting program correct on
all test pairs) and selected/correct witness names. Complete solver has no
exhaustive oracle claim. Twelve workers; one serial benchmark process; 600s cap.
No runtime comparison between arms, since each task runs all arms sequentially.
Pilot: 24 hash-selected tasks per dataset; full run if <120s. Full ARC1 checks
must reproduce complete solver 57/400 training, 23/400 evaluation and grid/object
fallback 60/400 training. Resolve discrepancies before interpreting transfer.

Close after measurement, retaining evidence and runnable source history. No
integration, tuning, private-set evaluation or accepted-math changes.
