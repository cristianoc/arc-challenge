# Relational objects and informative evidence

Public training tasks only; full; 400 real tasks, 64 controlled cases; 12 workers. Core and 003 object language unchanged. Primary real-task search and saving: 2.26s; total including depth-3 audit and constructed cases: 2.27s. Resource measurements are in stderr.txt.

## Real-task coverage

| Arm | Training-fit tasks | Task exact match | Test-grid exact match | Prediction coverage (grids) | Accuracy given fit | Oracle tasks |
|---|---:|---:|---:|---:|---:|---:|
| grid + 003 objects | 63/400 (15.75%) | 60/400 (15.00%) | 64/416 (15.38%) | 67/416 (16.11%) | 60/63 (95.24%) | 61/400 (15.25%) |
| relations only | 0/400 (0.00%) | 0/400 (0.00%) | 0/416 (0.00%) | 0/416 (0.00%) | n/a | 0/400 (0.00%) |
| control then relations fallback | 63/400 (15.75%) | 60/400 (15.00%) | 64/416 (15.38%) | 67/416 (16.11%) | 60/63 (95.24%) | 61/400 (15.25%) |

Control prefers a fitting grid program, then a fitting 003 object program. The new fallback uses relations only when neither control family fits. Within a language, first fitting program in the fixed order predicts. Oracle columns count any correct program across the listed arm's pools, including unselected families, and require all test outputs correct. These are enumeration experiments, not the complete production solver with repair/symmetry.

Relations add 0 training-fitting tasks and 0 oracle-correct tasks beyond the control. Candidate ASTs checked: 6617592; actual training program/example checks: 6617616. Candidate counts are syntax, not unique semantics or equivalent operation costs.

## Targeted depth-3 audit

Selected by newly fitting training data, before considering test correctness; at most 16 tasks.

| Task | Grid d3 fits | Grid d3 oracle | Relational oracle | Relation selected correct | Audit task seconds |
|---|---:|---|---|---|---:|

0 audited tasks have a relational correct program absent from both the control and the grid depth-3 pool. This is a targeted audit, not full-corpus depth-3 accuracy.

## Controlled abstraction identification

64 constructed cases: 32 scene seeds × two equally represented rules. Both models fit the initial example and disagree at test. The hypothesis prior is 1/2 per rule (initial entropy 1 bit). One of four offered example inputs distinguishes them; three do not. Both models are defined on every input.

| Query policy | Expected correct / 64 | Accuracy | Rule identified | Mean true-rule posterior mass | Mean posterior H (bits) | Mean Brier loss |
|---|---:|---:|---:|---:|---:|---:|
| no extra evidence | 32.0/64 | 50.00% | 0.0/64 | 0.500 | 1.000 | 0.500 |
| uninformative example | 32.0/64 | 50.00% | 0.0/64 | 0.500 | 1.000 | 0.500 |
| informative example (oracle query) | 64.0/64 | 100.00% | 64.0/64 | 1.000 | 0.000 | 0.000 |
| first menu example | 40.0/64 | 62.50% | 16.0/64 | 0.625 | 0.750 | 0.375 |
| random query (exact expectation) | 40.0/64 | 62.50% | 16.0/64 | 0.625 | 0.750 | 0.375 |
| maximum expected information gain | 64.0/64 | 100.00% | 64.0/64 | 1.000 | 0.000 | 0.000 |

Every individual deterministic hypothesis has zero answer entropy, so choosing the most certain hypothesis cannot separate these rules. Information gain instead compares their disagreement on candidate examples, without seeing the answers. One bit of discriminating evidence resolves one bit of uncertainty.

These outcomes follow from the deliberately constructed two-model setting. They verify the mechanism and demonstrate identifiable versus unidentifiable evidence; they do NOT establish automatic representation discovery, a realistic prior, a generally optimal ARC strategy, or access to additional labelled examples in ARC. Random-query results are exact expectations, not sampled estimates.

## Real tasks with relational fits

Correct witnesses are obtained by post-hoc scoring, never used by search or first-program selection.

| Task | Control / relational fits | Control / relational selected correct | Control / relational oracle | Relational first program | Correct relational witness |
|---|---:|---|---|---|---|
