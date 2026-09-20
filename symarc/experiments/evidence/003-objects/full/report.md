# Object abstraction run report

Public training only; full; 400 tasks; 12 workers. Core depth 2 versus a finite object AST: segment → select → act → render. No symmetry, repair, or fitted-pool cap.

Primary search, reserved-example validation, and pool-writing wall time: 3.81s. Including depth-3 audit: 22.16s. Resource measurements are in stderr.txt.

## Predicting unseen test answers after fitting all training examples

| Arm | Training-fit tasks | Task exact match | Grid exact match | Prediction coverage (grids) | Accuracy given fit | Oracle tasks | Oracle-minus-selected |
|---|---:|---:|---:|---:|---:|---:|---:|
| grid | 55/400 (13.75%) | 53/400 (13.25%) | 57/416 (13.70%) | 59/416 (14.18%) | 53/55 (96.36%) | 54/400 (13.50%) | 1 |
| objects | 17/400 (4.25%) | 16/400 (4.00%) | 16/416 (3.85%) | 17/416 (4.09%) | 16/17 (94.12%) | 16/400 (4.00%) | 0 |
| fallback | 63/400 (15.75%) | 60/400 (15.00%) | 64/416 (15.38%) | 67/416 (16.11%) | 60/63 (95.24%) | 61/400 (15.25%) | 1 |
| validation gate | 63/400 (15.75%) | 60/400 (15.00%) | 64/416 (15.38%) | 67/416 (16.11%) | 60/63 (95.24%) | 61/400 (15.25%) | 1 |

Union oracle: 61/400 (15.25%). Objects add 8 training-fitting tasks and 7 oracle-correct tasks beyond core depth 2. Oracle means at least one SINGLE program gives every test output correctly; it is not a selection method. Grid selects first shortest; objects select first AST in the registered order. Fallback prefers any fitting core program; validation gate prefers objects only when reserved-training-answer probability is strictly higher (tolerance 1e-12), otherwise core. The oracle columns for fallback/gate concern the chosen family, not the union.

## Is an object interpretation predictively useful?

For each task with ≥2 training examples, fit all except the last; predict the last input. Its output is excluded from inference and palette construction. Test outputs are never used for this decision. Refit all training pairs for the test report above. These are within-task validation examples, not the public evaluation split.

| Family | Eligible tasks | Discovery fit | First-program validation accuracy | Correct-answer support | Mean correct-answer probability | Mean predictive entropy (bits) | Mean multiclass Brier loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| grid | 400 | 60/400 (15.00%) | 55/400 (13.75%) | 55/400 (13.75%) | 0.123 | 0.089 | 1.727 |
| objects | 400 | 19/400 (4.75%) | 16/400 (4.00%) | 17/400 (4.25%) | 0.038 | 0.021 | 1.916 |

Objects assign more probability to the reserved answer on 10 tasks; grid on 48; ties on 342. Changing ONLY the grid prior to normalized length mass q(l) ∝ 2^-l, distributed uniformly over K^l sequences at length l, changes the object-preference decision on 0 tasks.

Primary probabilities use a uniform prior over each family's syntax conditioned on discovery fits. Empty pools abstain (unit mass on undefined), giving zero correct-answer support, entropy zero, and Brier loss 2. Low entropy can therefore mean failure, and must be read alongside support and accuracy. Brier is the sum of squared probability errors (0 best, 2 worst). No calibrated probability that a puzzle is intrinsically about objects is claimed; these are comparisons under explicit languages and priors.

| Validation preference | Tasks | Core test correct | Object test correct | Object-only wins | Core-only wins |
|---|---:|---:|---:|---:|---:|
| objects higher | 10 | 2 | 9 | 7 | 0 |
| core higher or tie | 390 | 51 | 7 | 0 | 44 |

Both languages fit the discovery examples on 11 tasks. Within these, objects receive higher reserved-answer probability on 3, grid on 2, and 6 tie. Among the 3 shared-fit tasks preferring objects, object/core test selection is correct on 3/2. This separates preference between viable families from choosing the only family that fits.


## Depth-3 audit of newly fitting tasks

Selected by training fit and hash order, never test correctness; at most 16 tasks.

| Task | Core d3 fits | Core d3 oracle | Object oracle | Core d3 selected correct | Object selected correct | Audit task seconds |
|---|---:|---|---|---|---|---:|
| 67385a82 | 0 | false | false | false | false | 0.32 |
| 88a62173 | 0 | false | true | false | true | 0.92 |
| a87f7484 | 0 | false | true | false | true | 5.68 |
| 42a50994 | 0 | false | true | false | true | 5.25 |
| 25d8a9c8 | 0 | false | true | false | true | 3.77 |
| aedd82e4 | 0 | false | true | false | true | 0.26 |
| 9565186b | 0 | false | true | false | true | 1.19 |
| 23b5c85d | 12 | false | true | false | true | 18.35 |

This is a targeted audit, not full-corpus depth-3 accuracy. 7 audited tasks have an object oracle-correct solution absent from the enumerated core depth-3 pool.

## Cost and prediction diagnostics

Core enumerated syntax budget sums to 9401796; object candidates 1966064 and actual training program/example checks 1968676. Core memoizes prefixes; these counts are not comparable primitive-operation costs. Summed per-task core/object full-fit times are 24.02s/0.55s (overlap across workers; exclude validation/audit).

| Task | Fits grid/object | Joint predictions grid/object | Object program H | Answer H grid/object | Viable segmentations | Segmentation H | Validation p grid/object | Test correct grid/object | Oracle grid/object |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1190e5a7 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| c444b776 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 6d0aefbc | 36/0 | 1/0 | n/a | 0.000/n/a | 0 | n/a | 0.973/0.000 | true/false | true/false |
| 85c4e7cd | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 3de23699 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 1c786137 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 53b68214 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 444801d8 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 28bf18c6 | 2/0 | 1/0 | n/a | 0.000/n/a | 0 | n/a | 0.333/0.000 | true/false | true/false |
| d9f24cd1 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| a78176bb | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 32597951 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 952a094c | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| b0c4d837 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| a85d4709 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 6f8cd79b | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 6e19193c | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 67385a82 | 0/2 | 0/1 | 1.000 | n/a/0.000 | 2 | 1.000 | 0.000/1.000 | false/false | false/false |
| b8825c91 | 4/0 | 2/0 | n/a | 1.000/n/a | 0 | n/a | 1.000/0.000 | false/false | true/false |
| 5bd6f4ac | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 2dd70a9a | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 0dfd9992 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 810b9b61 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 045e512c | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 50846271 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| ff28f65a | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| b94a9452 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 0a938d79 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 5c0a986e | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 623ea044 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 27a28665 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| d9fac9be | 1/0 | 1/0 | n/a | 0.000/n/a | 0 | n/a | 1.000/0.000 | false/false | false/false |
| 3ac3eb23 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 7f4411dc | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 7468f01a | 6/28 | 1/1 | 4.807 | 0.000/0.000 | 4 | 2.000 | 1.000/1.000 | true/true | true/true |
| ce9e57f2 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 67e8384a | 70/0 | 2/0 | n/a | 0.108/n/a | 0 | n/a | 0.795/0.000 | true/false | true/false |
| 88a62173 | 0/20 | 0/1 | 4.322 | n/a/0.000 | 4 | 2.000 | 0.000/0.500 | false/true | false/true |
| 8efcae92 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| e3497940 | 1/0 | 1/0 | n/a | 0.000/n/a | 0 | n/a | 1.000/0.000 | true/false | true/false |
| 2bee17df | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 272f95fa | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| bbc9ae5d | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| b2862040 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 6e02f1e3 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 31aa019c | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| ea786f4a | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| d4a91cb9 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 82819916 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| dae9d2b5 | 35/0 | 1/0 | n/a | 0.000/n/a | 0 | n/a | 1.000/0.000 | true/false | true/false |
| 995c5fa3 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 3befdf3e | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 4be741c5 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 9172f3a0 | 45/0 | 13/0 | n/a | 3.148/n/a | 0 | n/a | 0.763/0.000 | true/false | true/false |
| a3325580 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 90f3ed37 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| a79310a0 | 4/0 | 1/0 | n/a | 0.000/n/a | 0 | n/a | 1.000/0.000 | true/false | true/false |
| 5c2c9af4 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| caa06a1f | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| ff805c23 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| a48eeaf7 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| c8cbb738 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 77fdfe62 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| d23f8c26 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 2281f1f4 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 662c240a | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 2204b7a8 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| a65b410d | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| e8593010 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 4c4377d9 | 1/0 | 1/0 | n/a | 0.000/n/a | 0 | n/a | 1.000/0.000 | true/false | true/false |
| d687bc17 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 95990924 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 7df24a62 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 3f7978a0 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 5ad4f10b | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| d364b489 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 855e0971 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 1f642eb9 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 150deff5 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| d43fd935 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 7fe24cdd | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 5daaa586 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 0e206a2e | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 39a8645d | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| d631b094 | 58/0 | 8/0 | n/a | 1.390/n/a | 0 | n/a | 0.829/0.000 | true/false | true/false |
| ce22a75a | 6/0 | 1/0 | n/a | 0.000/n/a | 0 | n/a | 1.000/0.000 | true/false | true/false |
| 1a07d186 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| f8ff0b80 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 3aa6fb7a | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 3c9b0459 | 117/0 | 19/0 | n/a | 2.336/n/a | 0 | n/a | 0.854/0.000 | true/false | true/false |
| 6cf79266 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| a416b8f3 | 72/0 | 8/0 | n/a | 2.213/n/a | 0 | n/a | 0.692/0.000 | true/false | true/false |
| a64e4611 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 94f9d214 | 30/0 | 1/0 | n/a | 0.000/n/a | 0 | n/a | 1.000/0.000 | true/false | true/false |
| f5b8619d | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 41e4d17e | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 496994bd | 32/0 | 5/0 | n/a | 1.497/n/a | 0 | n/a | 0.762/0.000 | true/false | true/false |
| 97a05b5b | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 928ad970 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 39e1d7f9 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| a68b268e | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 72ca375d | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 3af2c5a8 | 16/0 | 4/0 | n/a | 1.750/n/a | 0 | n/a | 0.889/0.000 | true/false | true/false |
| c0f76784 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 4347f46a | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 46442a0e | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 98cf29f8 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 36fdfd69 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 08ed6ac7 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 6d75e8bb | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 7b7f7511 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| e40b9e2f | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 8d5021e8 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| b190f7f5 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 60b61512 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 253bf280 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| b60334d2 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 44d8ac46 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 8e1813be | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| d6ad076f | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 54d82841 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 25ff71a9 | 6/0 | 1/0 | n/a | 0.000/n/a | 0 | n/a | 0.375/0.000 | true/false | true/false |
| e8dc4411 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| d5d6de2d | 3/0 | 1/0 | n/a | 0.000/n/a | 0 | n/a | 1.000/0.000 | true/false | true/false |
| 780d0b14 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 29623171 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| beb8660c | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| fcb5c309 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| eb281b96 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 6e82a1ae | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| a3df8b1e | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| cce03e0d | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 99b1bc43 | 50/0 | 5/0 | n/a | 0.562/n/a | 0 | n/a | 1.000/0.000 | true/false | true/false |
| a87f7484 | 0/10 | 0/1 | 3.322 | n/a/0.000 | 1 | 0.000 | 0.000/0.714 | false/true | false/true |
| 88a10436 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 57aa92db | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| ba26e723 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 941d9a10 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 4c5c2cf0 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| b230c067 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| a1570a43 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 1e32b0e9 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| cbded52d | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| e73095fd | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 42a50994 | 0/8 | 0/1 | 3.000 | n/a/0.000 | 4 | 2.000 | 0.000/1.000 | false/true | false/true |
| 2013d3e2 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 1cf80156 | 100/280 | 7/1 | 8.129 | 1.873/0.000 | 8 | 3.000 | 0.769/1.000 | true/true | true/true |
| 7447852a | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 6ecd11f4 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 2c608aff | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 91413438 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| aba27056 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 3428a4f5 | 35/0 | 2/0 | n/a | 0.094/n/a | 0 | n/a | 1.000/0.000 | true/false | true/false |
| 48d8fb45 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 4093f84a | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 0ca9ddb6 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| de1cd16c | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| d406998b | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| b7249182 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 22eb0ac0 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| d07ae81c | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| af902bf9 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| d90796e8 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 6cdd2623 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| f8b3ba0a | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 6430c8c4 | 50/0 | 1/0 | n/a | 0.000/n/a | 0 | n/a | 1.000/0.000 | true/false | true/false |
| f8a8fe49 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| dbc1a6ce | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 868de0fa | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 017c7c7b | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 1fad071e | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| ea32f347 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 6150a2bd | 62/0 | 17/0 | n/a | 3.273/n/a | 0 | n/a | 0.596/0.000 | true/false | true/false |
| b782dc8a | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| a8d7556c | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| b8cdaf2b | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| c9f8e694 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| b6afb2da | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 06df4c85 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 25d8a9c8 | 0/2 | 0/1 | 1.000 | n/a/0.000 | 2 | 1.000 | 0.000/1.000 | false/true | false/true |
| be94b721 | 121/40 | 11/1 | 5.322 | 1.145/0.000 | 8 | 3.000 | 0.992/1.000 | true/true | true/true |
| 8d510a79 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 44f52bb0 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| aedd82e4 | 0/2 | 0/1 | 1.000 | n/a/0.000 | 2 | 1.000 | 0.000/1.000 | false/true | false/true |
| fafffa47 | 34/0 | 1/0 | n/a | 0.000/n/a | 0 | n/a | 1.000/0.000 | true/false | true/false |
| bb43febb | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 10fcaaa3 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 91714a58 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 3345333e | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 7837ac64 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 2dc579da | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 5168d44c | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| ef135b50 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 1b60fb0c | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| b548a754 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| cdecee7f | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 178fcbfb | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| c1d99e64 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 9dfd6313 | 25/0 | 1/0 | n/a | 0.000/n/a | 0 | n/a | 0.342/0.000 | true/false | true/false |
| 2bcee788 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 63613498 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| b9b7f026 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| ac0a08a4 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 0520fde7 | 36/0 | 2/0 | n/a | 0.183/n/a | 0 | n/a | 0.837/0.000 | true/false | true/false |
| 72322fa7 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| bd4472b8 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 846bdb03 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 8f2ea7aa | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 228f6490 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 0d3d703e | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 794b24be | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| d2abd087 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 9565186b | 0/28 | 0/5 | 4.807 | n/a/2.020 | 4 | 1.592 | 0.000/0.700 | false/true | false/true |
| b27ca6d3 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| f8c80d96 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| f35d900a | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| f9012d9b | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 99fa7670 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 890034e9 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 9ecd008a | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| f25fbde4 | 2/0 | 1/0 | n/a | 0.000/n/a | 0 | n/a | 1.000/0.000 | true/false | true/false |
| 6fa7a44f | 66/0 | 1/0 | n/a | 0.000/n/a | 0 | n/a | 1.000/0.000 | true/false | true/false |
| b1948b0a | 44/2 | 1/1 | 1.000 | 0.000/0.000 | 2 | 1.000 | 1.000/1.000 | true/true | true/true |
| 8403a5d5 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 834ec97d | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 6855a6e4 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 469497ad | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| e98196ab | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 007bbfb7 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 6b9890af | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| a61f2674 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 22168020 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| ce602527 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 09629e4f | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| e50d258f | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 67a423a3 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| e76a88a6 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| ed36ccf7 | 25/0 | 6/0 | n/a | 2.080/n/a | 0 | n/a | 0.714/0.000 | true/false | true/false |
| 3bdb4ada | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| fcc82909 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| b91ae062 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 025d127b | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| d4469b4b | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 8e5a5113 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 4258a5f9 | 11/0 | 1/0 | n/a | 0.000/n/a | 0 | n/a | 0.917/0.000 | true/false | true/false |
| 1e0a9b12 | 45/0 | 10/0 | n/a | 2.379/n/a | 0 | n/a | 0.529/0.000 | true/false | true/false |
| 5582e5ca | 103/12 | 7/1 | 3.585 | 0.755/0.000 | 4 | 1.551 | 0.896/0.706 | true/true | true/true |
| eb5a1d5d | 2/0 | 1/0 | n/a | 0.000/n/a | 0 | n/a | 1.000/0.000 | true/false | true/false |
| 3631a71a | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| d037b0a7 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| dc433765 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| c9e6f938 | 12/0 | 1/0 | n/a | 0.000/n/a | 0 | n/a | 1.000/0.000 | true/false | true/false |
| c3e719e8 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| f15e1fac | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| ded97339 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 29ec7d0e | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| d10ecb37 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 56ff96f3 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 5521c0d9 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 1caeab9d | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| d13f3404 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 8eb1be9a | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 447fd412 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 3618c87e | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| d06dbe63 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 3906de3d | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 264363fd | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| f1cefba8 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| e9afcf9a | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| a5f85a15 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 9aec4887 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| ae3edfdc | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| c909285e | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 445eab21 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| f2829549 | 50/0 | 2/0 | n/a | 0.141/n/a | 0 | n/a | 1.000/0.000 | true/false | true/false |
| ecdecbb3 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 97999447 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 776ffc46 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 239be575 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 8be77c9e | 10/0 | 1/0 | n/a | 0.000/n/a | 0 | n/a | 0.909/0.000 | true/false | true/false |
| 4938f0c2 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 321b1fc6 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 0962bcdd | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| e5062a87 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 50cb2852 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 1f85a75f | 70/130 | 9/13 | 7.022 | 1.450/1.856 | 8 | 2.922 | 0.745/0.570 | true/true | true/true |
| 00d62c1b | 8/0 | 1/0 | n/a | 0.000/n/a | 0 | n/a | 1.000/0.000 | true/false | true/false |
| 137eaa0f | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 543a7ed5 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 5614dbcf | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 56dc2b01 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 3eda0437 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 49d1d64f | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| bda2d7a6 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| f76d97a5 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| a2fd1cf0 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| db93a21d | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| aabf363d | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 6aa20dc0 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| a61ba2ce | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| e179c5f4 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| a9f96cdd | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 4612dd53 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| d511f180 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 963e52fc | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 8a004b2b | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 90c28cc7 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 05f2a901 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 3bd67248 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| e21d9049 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 74dd1130 | 107/0 | 17/0 | n/a | 2.250/n/a | 0 | n/a | 1.000/0.000 | true/false | true/false |
| 75b8110e | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| bc1d5164 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| bdad9b1f | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 9d9215db | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| a740d043 | 2/0 | 1/0 | n/a | 0.000/n/a | 0 | n/a | 1.000/0.000 | true/false | true/false |
| ddf7fa4f | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| dc0a314f | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 760b3cac | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 23581191 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| d4f3cd78 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 7c008303 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 1b2d62fb | 34/0 | 1/0 | n/a | 0.000/n/a | 0 | n/a | 1.000/0.000 | true/false | true/false |
| 9af7a82c | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 8731374e | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| b527c5c6 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 6d0160f0 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 913fb3ed | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 40853293 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| e509e548 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| feca6190 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| d89b689b | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 54d9e175 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 6773b310 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 6c434453 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| ec883f72 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| a8c38be5 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| c3f564a4 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 1f0c79e5 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 6a1e5592 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 23b5c85d | 0/28 | 0/1 | 4.807 | n/a/0.000 | 4 | 2.000 | 0.000/1.000 | false/true | false/true |
| 73251a56 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 7e0986d6 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 5117e062 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 0b148d64 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 83302e8f | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| e6721834 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 681b3aeb | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 80af3007 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 4522001f | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 363442ee | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 746b3537 | 2/0 | 1/0 | n/a | 0.000/n/a | 0 | n/a | 1.000/0.000 | true/false | true/false |
| 29c11459 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 93b581b8 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 9f236235 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 11852cab | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 6455b5f5 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| c59eb873 | 71/0 | 24/0 | n/a | 4.112/n/a | 0 | n/a | 1.000/0.000 | true/false | true/false |
| 47c1f68c | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| f25ffba3 | 56/0 | 15/0 | n/a | 3.128/n/a | 0 | n/a | 0.778/0.000 | true/false | true/false |
| 9edfc990 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| db3e9e38 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| d8c310e9 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 234bbc79 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| d22278a0 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 673ef223 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| a699fb00 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| cf98881b | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 36d67576 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| e48d4e1a | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| ae4f1146 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 68b16354 | 60/63 | 1/6 | 5.977 | 0.000/1.453 | 3 | 1.585 | 1.000/1.000 | true/true | true/true |
| d0f5fe59 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 539a4f51 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| b775ac94 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 3e980e27 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 46f33fce | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 22233c11 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 28e73c20 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 2dee498d | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 4290ef0e | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 62c24649 | 19/0 | 1/0 | n/a | 0.000/n/a | 0 | n/a | 1.000/0.000 | true/false | true/false |
| 7b6016b9 | 1/0 | 1/0 | n/a | 0.000/n/a | 0 | n/a | 1.000/0.000 | true/false | true/false |
| 05269061 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| dc1df850 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 25d487eb | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 7ddcd7ec | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 1bfc4729 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| a5313dff | 8/0 | 1/0 | n/a | 0.000/n/a | 0 | n/a | 1.000/0.000 | true/false | true/false |
| 67a3c6ac | 48/42 | 1/1 | 5.392 | 0.000/0.000 | 2 | 1.000 | 1.000/1.000 | true/true | true/true |
| 1f876c06 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| c8f0f002 | 55/2 | 1/1 | 1.000 | 0.000/0.000 | 2 | 1.000 | 1.000/1.000 | true/true | true/true |
| ce4f8723 | 49/0 | 3/0 | n/a | 0.287/n/a | 0 | n/a | 0.980/0.000 | true/false | true/false |
| e26a3af2 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 508bd3b6 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 484b58aa | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 694f12f3 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| e9614598 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| 6d58a25d | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |
| ba97ae07 | 0/0 | 0/0 | n/a | n/a/n/a | 0 | n/a | 0.000/0.000 | false/false | false/false |

Entropies condition on full training fits; empty-pool entropy is n/a here. Segmentation entropy is over eight named interpretations, weighted by fitting AST counts, with initial entropy 3 bits. Interpretations can coincide on observed grids, so this is not semantic entropy. Joint prediction classes identify agreement only on observed test inputs, including undefined.

## Witnesses and selection failures

Shown whenever objects fit and core does not, or object/core test accuracy differs. Correct witnesses were found by post-hoc scoring; they did not guide inference.

| Task | Core selected | Object selected | Object correct witness |
|---|---|---|---|
| 67385a82 | `—` | `zero-4-mono / Widest / Recolour(8) / Original` | `—` |
| 88a62173 | `—` | `zero-8-mono / UniqueShape / Identity / Crop` | `zero-8-mono / UniqueShape / Identity / Crop` |
| a87f7484 | `—` | `zero-8-mono / Largest / Identity / Crop` | `zero-8-mono / Largest / Identity / Crop` |
| 42a50994 | `—` | `zero-8-mono / Smallest / Erase / Original` | `zero-8-mono / Smallest / Erase / Original` |
| 25d8a9c8 | `—` | `zero-4-mono / Widest / Recolour(5) / Selected` | `zero-4-mono / Widest / Recolour(5) / Selected` |
| aedd82e4 | `—` | `zero-4-mono / Smallest / Recolour(1) / Original` | `zero-4-mono / Smallest / Recolour(1) / Original` |
| 9565186b | `—` | `modal-4-mono / All / Recolour(5) / Original` | `modal-4-mono / All / Recolour(5) / Original` |
| 23b5c85d | `—` | `zero-4-mono / Smallest / Identity / Crop` | `zero-4-mono / Smallest / Identity / Crop` |
