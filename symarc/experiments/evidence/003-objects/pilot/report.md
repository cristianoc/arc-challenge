# Object abstraction run report

Public training only; pilot; 24 tasks; 12 workers. Core depth 2 versus a finite object AST: segment → select → act → render. No symmetry, repair, or fitted-pool cap.

Primary search, reserved-example validation, and pool-writing wall time: 0.45s. Including depth-3 audit: 0.45s. Resource measurements are in stderr.txt.

## Predicting unseen test answers after fitting all training examples

| Arm | Training-fit tasks | Task exact match | Grid exact match | Prediction coverage (grids) | Accuracy given fit | Oracle tasks | Oracle-minus-selected |
|---|---:|---:|---:|---:|---:|---:|---:|
| grid | 3/24 (12.50%) | 2/24 (8.33%) | 2/25 (8.00%) | 3/25 (12.00%) | 2/3 (66.67%) | 3/24 (12.50%) | 1 |
| objects | 1/24 (4.17%) | 0/24 (0.00%) | 0/25 (0.00%) | 1/25 (4.00%) | 0/1 (0.00%) | 0/24 (0.00%) | 0 |
| fallback | 4/24 (16.67%) | 2/24 (8.33%) | 2/25 (8.00%) | 4/25 (16.00%) | 2/4 (50.00%) | 3/24 (12.50%) | 1 |
| validation gate | 4/24 (16.67%) | 2/24 (8.33%) | 2/25 (8.00%) | 4/25 (16.00%) | 2/4 (50.00%) | 3/24 (12.50%) | 1 |

Union oracle: 3/24 (12.50%). Objects add 1 training-fitting tasks and 0 oracle-correct tasks beyond core depth 2. Oracle means at least one SINGLE program gives every test output correctly; it is not a selection method. Grid selects first shortest; objects select first AST in the registered order. Fallback prefers any fitting core program; validation gate prefers objects only when reserved-training-answer probability is strictly higher (tolerance 1e-12), otherwise core. The oracle columns for fallback/gate concern the chosen family, not the union.

## Is an object interpretation predictively useful?

For each task with ≥2 training examples, fit all except the last; predict the last input. Its output is excluded from inference and palette construction. Test outputs are never used for this decision. Refit all training pairs for the test report above. These are within-task validation examples, not the public evaluation split.

| Family | Eligible tasks | Discovery fit | First-program validation accuracy | Correct-answer support | Mean correct-answer probability | Mean predictive entropy (bits) | Mean multiclass Brier loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| grid | 24 | 3/24 (12.50%) | 3/24 (12.50%) | 3/24 (12.50%) | 0.096 | 0.074 | 1.778 |
| objects | 24 | 1/24 (4.17%) | 1/24 (4.17%) | 1/24 (4.17%) | 0.042 | -0.000 | 1.917 |

Objects assign more probability to the reserved answer on 1 tasks; grid on 3; ties on 20. Changing ONLY the grid prior to normalized length mass q(l) ∝ 2^-l, distributed uniformly over K^l sequences at length l, changes the object-preference decision on 0 tasks.

Primary probabilities use a uniform prior over each family's syntax conditioned on discovery fits. Empty pools abstain (unit mass on undefined), giving zero correct-answer support, entropy zero, and Brier loss 2. Low entropy can therefore mean failure, and must be read alongside support and accuracy. Brier is the sum of squared probability errors (0 best, 2 worst). No calibrated probability that a puzzle is intrinsically about objects is claimed; these are comparisons under explicit languages and priors.

| Validation preference | Tasks | Core test correct | Object test correct | Object-only wins | Core-only wins |
|---|---:|---:|---:|---:|---:|
| objects higher | 1 | 0 | 0 | 0 | 0 |
| core higher or tie | 23 | 2 | 0 | 0 | 2 |

## Depth-3 audit of newly fitting tasks

Selected by training fit and hash order, never test correctness; at most 16 tasks.

| Task | Core d3 fits | Core d3 oracle | Object oracle | Core d3 selected correct | Object selected correct | Audit task seconds |
|---|---:|---|---|---|---|---:|

This is a targeted audit, not full-corpus depth-3 accuracy. 0 audited tasks have an object oracle-correct solution absent from the enumerated core depth-3 pool.

## Cost and prediction diagnostics

Core enumerated syntax budget sums to 612320; object candidates 118432 and actual training program/example checks 118486. Core memoizes prefixes; these counts are not comparable primitive-operation costs. Summed per-task core/object full-fit times are 1.77s/0.05s (overlap across workers; exclude validation/audit).

| Task | Fits grid/object | Joint predictions grid/object | Object program H | Answer H grid/object | Viable segmentations | Segmentation H | Validation p grid/object | Test correct grid/object | Oracle grid/object |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1190e5a7 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| c444b776 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 6d0aefbc | 36/0 | 1/0 | n/a | -0.000/n/a | 0 | n/a | 0.973/0.000 | true/false | true/false |
| 85c4e7cd | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 3de23699 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 1c786137 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 53b68214 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 444801d8 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 28bf18c6 | 2/0 | 1/0 | n/a | -0.000/n/a | 0 | n/a | 0.333/0.000 | true/false | true/false |
| d9f24cd1 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| a78176bb | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 32597951 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 952a094c | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| b0c4d837 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| a85d4709 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 6f8cd79b | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 6e19193c | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 67385a82 | 0/2 | 0/1 | 1.000 | n/a/-0.000 | 2 | 1.000 | 0.000/1.000 | false/false | false/false |
| b8825c91 | 4/0 | 2/0 | n/a | 1.000/n/a | 0 | n/a | 1.000/0.000 | false/false | true/false |
| 5bd6f4ac | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 2dd70a9a | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 0dfd9992 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 810b9b61 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 045e512c | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |

Entropies condition on full training fits; empty-pool entropy is n/a here. Segmentation entropy is over eight named interpretations, weighted by fitting AST counts, with initial entropy 3 bits. Interpretations can coincide on observed grids, so this is not semantic entropy. Joint prediction classes identify agreement only on observed test inputs, including undefined.

## Witnesses and selection failures

Shown whenever objects fit and core does not, or object/core test accuracy differs. Correct witnesses were found by post-hoc scoring; they did not guide inference.

| Task | Core selected | Object selected | Object correct witness |
|---|---|---|---|
| 67385a82 | `—` | `zero-4-mono / Widest / Recolour(8) / Original` | `—` |
