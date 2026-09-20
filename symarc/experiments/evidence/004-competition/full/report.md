# Competing interpretations run report

Public training only; full; 12 workers; fixed grid DSL depth 2 plus eight frozen object interpretations. Shared first surviving program within each family. Wall time including both conditions and diagnostics: 2.46s.

Full uses all but the last training pair for discovery and the last as evidence. Two-pair uses only the first two pairs; extra training outputs are deliberately withheld. Program vocabularies are frozen before evidence, unlike full refitting in 003. These are controlled inference experiments, not complete production-solver scores.

## Was improvement possible?

| Condition | Tasks | Any surviving family | ≥2 surviving families | Competitive (disagree) | Fixed-order correct | Family-choice ceiling | Recoverable baseline errors | Correct baseline vulnerable to switching | Full-program-pool ceiling |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 400 | 63 | 16 | 2 | 60 | 60 | 0 | 2 | 61 |
| two-pair | 400 | 69 | 17 | 4 | 59 | 61 | 2 | 2 | 64 |

Competitive membership uses predictions only, never scoring labels. The family-choice ceiling chooses among the SAME representatives available to every policy. The full-program ceiling additionally changes programs and is not attainable merely by selecting a family. Recoverable errors are the actual improvement opportunity; fewer than five supports descriptive cases only.

## Accuracy and paired comparisons

| Condition | Cohort | Policy | Tasks | Task exact match | Test-grid exact match | Prediction coverage | Wins vs fixed | Losses vs fixed | Unused training grids correct |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| full | all | fixed order | 400 | 60/400 (15.00%) | 64/416 (15.38%) | 67/416 (16.11%) | 0 | 0 | n/a (0/0) |
| full | all | predictive evidence | 400 | 60/400 (15.00%) | 64/416 (15.38%) | 67/416 (16.11%) | 0 | 0 | n/a (0/0) |
| full | all | minimum answer entropy | 400 | 60/400 (15.00%) | 64/416 (15.38%) | 67/416 (16.11%) | 0 | 0 | n/a (0/0) |
| full | competitive | fixed order | 2 | 2/2 (100.00%) | 2/2 (100.00%) | 2/2 (100.00%) | 0 | 0 | n/a (0/0) |
| full | competitive | predictive evidence | 2 | 2/2 (100.00%) | 2/2 (100.00%) | 2/2 (100.00%) | 0 | 0 | n/a (0/0) |
| full | competitive | minimum answer entropy | 2 | 2/2 (100.00%) | 2/2 (100.00%) | 2/2 (100.00%) | 0 | 0 | n/a (0/0) |
| two-pair | all | fixed order | 400 | 59/400 (14.75%) | 63/416 (15.14%) | 73/416 (17.55%) | 0 | 0 | 87/502 (17.33%) |
| two-pair | all | predictive evidence | 400 | 60/400 (15.00%) | 64/416 (15.38%) | 72/416 (17.31%) | 1 | 0 | 88/502 (17.53%) |
| two-pair | all | minimum answer entropy | 400 | 59/400 (14.75%) | 63/416 (15.14%) | 72/416 (17.31%) | 0 | 0 | 87/502 (17.33%) |
| two-pair | competitive | fixed order | 4 | 2/4 (50.00%) | 2/4 (50.00%) | 4/4 (100.00%) | 0 | 0 | 3/5 (60.00%) |
| two-pair | competitive | predictive evidence | 4 | 3/4 (75.00%) | 3/4 (75.00%) | 3/4 (75.00%) | 1 | 0 | 4/5 (80.00%) |
| two-pair | competitive | minimum answer entropy | 4 | 2/4 (50.00%) | 2/4 (50.00%) | 3/4 (75.00%) | 0 | 0 | 3/5 (60.00%) |

Evidence selects the largest surviving/discovery program ratio. Minimum answer entropy selects the lowest mean per-test-input entropy under uniform surviving syntax. Fixed-order tie breaking (tolerance 1e-12) applies throughout. Neither sees scoring outputs. Undefined is a prediction category for entropy, but never correct; empty families are unselectable.

## Representation uncertainty

| Condition | Tasks retaining any family | Mean initial viable families | Mean final viable families | Mean initial family H | Mean posterior family H | Evidence-choice changes with grid length prior | Unused training prediction coverage (fixed/evidence/min-H) |
|---|---:|---:|---:|---:|---:|---:|---:|
| full | 63 | 1.968 | 1.889 | 0.525 | 0.502 | 0 | n/a (0/0) / n/a (0/0) / n/a (0/0) |
| two-pair | 69 | 2.435 | 1.942 | 0.730 | 0.502 | 0 | 102/502 (20.32%) / 101/502 (20.12%) / 101/502 (20.12%) |

Initial family prior is uniform over discovery-viable families, conditional on discovery; posterior mass is proportional to evidence probability. Entropies are bits over named families, not semantic equivalence classes. Some object labels describe identical partitions on these inputs. A single observed answer can increase or decrease entropy. Grid-prior sensitivity uses q(length) ∝ 2^-length / K^length per program; object priors remain uniform.

## Competing cases and missed within-family solutions

Includes every competitive task and every task with a pool-correct program but no correct family representative.

| Condition | Task | Surviving families | Fixed / evidence / min-H family | Correct fixed/evidence/min-H | Family / pool oracle | Initial → posterior family H |
|---|---|---:|---|---|---|---:|
| two-pair | 67385a82 | 2 | zero-4-mono / zero-4-mono / zero-4-mono | false/false/false | false/true | 1.000→1.000 |
| full | b8825c91 | 1 | grid / grid / grid | false/false/false | false/true | 0.000→0.000 |
| two-pair | b8825c91 | 1 | grid / grid / grid | false/false/false | false/true | 0.000→0.000 |
| two-pair | d9fac9be | 1 | grid / grid / grid | false/false/false | false/true | 0.000→0.000 |
| two-pair | 88a62173 | 9 | grid / zero-4-mono / zero-4-mono | false/false/false | true/true | 3.170→3.129 |
| full | 1f85a75f | 9 | grid / grid / zero-4-mono | true/true/true | true/true | 3.170→3.140 |
| two-pair | 1f85a75f | 9 | grid / grid / zero-4-mono | true/true/true | true/true | 3.170→3.140 |
| two-pair | 23b5c85d | 5 | grid / zero-4-mono / grid | false/true/false | true/true | 2.322→2.283 |
| full | 68b16354 | 4 | grid / grid / grid | true/true/true | true/true | 2.000→2.000 |
| two-pair | 68b16354 | 4 | grid / grid / grid | true/true/true | true/true | 2.322→1.997 |

## Recoverable errors and harmful switches

For each recoverable baseline error or actual harmful switch, list ALL surviving representatives. Correctness is post-hoc scoring, never a selection input.

| Condition | Task | Family | Discovery → surviving programs | Evidence probability | Answer entropy | Representative correct | Program |
|---|---|---|---:|---:|---:|---|---|
| two-pair | 88a62173 | grid | 125→39 | 0.312 | 1.417 | false | `cropLargest` |
| two-pair | 88a62173 | zero-4-mono | 6→5 | 0.833 | 0.000 | false | `zero-4-mono / UniqueShape / Identity / Crop` |
| two-pair | 88a62173 | zero-4-multi | 6→5 | 0.833 | 0.000 | false | `zero-4-multi / UniqueShape / Identity / Crop` |
| two-pair | 88a62173 | zero-8-mono | 6→5 | 0.833 | 0.000 | true | `zero-8-mono / UniqueShape / Identity / Crop` |
| two-pair | 88a62173 | zero-8-multi | 6→5 | 0.833 | 0.000 | true | `zero-8-multi / UniqueShape / Identity / Crop` |
| two-pair | 88a62173 | modal-4-mono | 6→5 | 0.833 | 0.000 | false | `modal-4-mono / UniqueShape / Identity / Crop` |
| two-pair | 88a62173 | modal-4-multi | 6→5 | 0.833 | 0.000 | false | `modal-4-multi / UniqueShape / Identity / Crop` |
| two-pair | 88a62173 | modal-8-mono | 6→5 | 0.833 | 0.000 | true | `modal-8-mono / UniqueShape / Identity / Crop` |
| two-pair | 88a62173 | modal-8-multi | 6→5 | 0.833 | 0.000 | true | `modal-8-multi / UniqueShape / Identity / Crop` |
| two-pair | 23b5c85d | grid | 9→2 | 0.222 | 0.000 | false | `recolour 2 0 ; cropSmallest` |
| two-pair | 23b5c85d | zero-4-mono | 16→7 | 0.438 | 0.000 | true | `zero-4-mono / Smallest / Identity / Crop` |
| two-pair | 23b5c85d | zero-8-mono | 16→7 | 0.438 | 0.000 | true | `zero-8-mono / Smallest / Identity / Crop` |
| two-pair | 23b5c85d | modal-4-mono | 16→7 | 0.438 | 0.000 | true | `modal-4-mono / Smallest / Identity / Crop` |
| two-pair | 23b5c85d | modal-8-mono | 16→7 | 0.438 | 0.000 | true | `modal-8-mono / Smallest / Identity / Crop` |

Compact family diagnostics and representative predictions for all tasks are saved in families.jsonl. No new DSL features or policies were added after observing results.
