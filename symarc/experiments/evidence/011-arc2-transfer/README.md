# 011: the unchanged solver scores zero on ARC-AGI-2 public evaluation

**Implemented and run:** a small Rust benchmark adapter imports the existing
stable solver and 003 object library. It reruns the complete solver with its
unchanged default configuration (seed 0), and separately runs grid depth 2,
objects, and grid-first object fallback. No new search rules or primitives were
added. Object fallback remains a separate experimental configuration, not part
of the complete solver.

## Results: tasks with every test output exactly correct

| Dataset | Complete solver | Grid depth 2 | Objects alone | Grid then objects |
|---|---:|---:|---:|---:|
| ARC1 training | 57/400 (14.25%) | 53/400 | 16/400 | 60/400 (15.00%) |
| ARC1 evaluation | 23/400 (5.75%) | 20/400 | 1/400 | 21/400 (5.25%) |
| ARC2 training | 79/1000 (7.90%) | 72/1000 | 17/1000 | 80/1000 (8.00%) |
| ARC2 evaluation | **0/120** | **0/120** | **0/120** | **0/120** |

ARC1 complete-solver and training fallback counts reproduce the earlier reports
exactly. On ARC2 evaluation all configurations also have **zero training-perfect
fits** and get **0/167 test grids** correct. This is a coverage failure under the
existing language/search limits, not merely failure to choose among fitting
candidates. It does not rule out fitting programs outside those limits.

Earlier references to 60/400 concerned the ARC1 training grid/object experiment.
The earlier complete-solver baseline concerned all 800 ARC1 public tasks. In
contrast, 006's external collection of 120 task-specific Python programs was
already evaluated on ARC2; its 27/120 result belongs to those supplied programs,
not SymArc. The distinction matters when comparing figures across the notes.

## Training overlap and limits

ARC2 training is not an independent set of 1,000 new tasks. Against this repo's
ARC1 snapshot, 767 task IDs overlap. Of those, 766 have identical labelled pairs
within each split when pair order is ignored; `ac0c5833` has changed content.
Example ordering differs, which can affect these deterministic search pipelines.
Another six ARC1 IDs occur in ARC2 evaluation. See [overlap details](overlap.json).

| ARC2 training cohort | Complete solver | Grid then objects |
|---|---:|---:|
| IDs already present in local ARC1 | 78/767 | 79/767 |
| IDs absent from local ARC1 | **1/233** | **1/233** |

The one solved new-ID task is `5751f35e`. ID overlap is a descriptive partition,
not a claim that differently named tasks are semantically independent. All data
are public and have been used in this research; this is not a private benchmark
or an untouched-holdout result. ARC2 source is pinned at
`f3283f727488ad98fe575ea6a5ac981e4a188e49`; known annotation anomalies were not edited.

## Evidence and reproduction

Protocol and adapter were committed before running at
`f40ffd2013d7136b2e58faaa7e46e7ace85819fb`. The 48-task pilot (24 hash-selected
per dataset) completed in 2.42s, passing the registered gate. The full 1,920-task
comparison completed in 65.87s with 12 workers. Arms run sequentially within each
worker; these times do not support an arm-by-arm speed comparison. No tuning
followed the pilot. Clean run manifests record sources, binaries, data hashes,
commands and environments. No stable solver, 003 library or accepted mathematics
changed. The historical core has had a documented allocation optimization since
the oldest baseline; the correctness checks reproduce that baseline's counts.

- [Full direct report](full/report.md), [per-task predictions and witnesses](full/tasks.tsv), [manifest](full/run.json).
- [Pilot report](pilot/report.md), [manifest](pilot/run.json).
- [Registered protocol](protocol.md).

The TSV's oracle column refers to the chosen family, not the union of grid and
object pools. It is unavailable for the complete heuristic solver. Selected
program and oracle-witness names identify programs; grid arrays can be regenerated
at the recorded revision. Test outputs are used for scoring, not search decisions.

Closed measurement. The runner is preserved in Git history at `f40ffd2` and
removed from the live experiment tree. To rerun at that revision, from repo root:

```sh
python3 symarc/experiments/011-arc2-transfer/run.py pilot /path/to/ARC-AGI-2
python3 symarc/experiments/011-arc2-transfer/run.py full /path/to/ARC-AGI-2
```

This establishes the requested transfer baseline. Improving selection alone
cannot resolve the zero-fit evaluation result; a later study would need broader
expressivity/search and a separately specified evaluation protocol.

## Direct task viewers from the preceding audit

- [31f7f899 on ARC Prize](https://arcprize.org/tasks/31f7f899): the one-cell symmetry exception.
- [a416fc5b on ARC Prize](https://arcprize.org/play?task=a416fc5b): the anomalous third training output. Also available as our [complete task board](../010-test-surprises/figures/a416fc5b.svg).
- [67e490f4 on ARC Prize](https://arcprize.org/tasks/67e490f4): square training outputs versus a rectangular test template.

The `a416fc5b` viewer link is advertised by [upstream PR #18](https://github.com/arcprize/ARC-AGI-2/pull/18), which already reports the missing input/blank output and proposes a correction. That PR was open when checked; its correction is not in the pinned dataset. The direct viewer could not be loaded by our web retrieval tool, so the task board and upstream before/after images provide alternatives.
