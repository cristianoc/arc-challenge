# Competing interpretations run report

Public training only; pilot; 12 workers; fixed grid DSL depth 2 plus eight frozen object interpretations. Shared first surviving program within each family. Wall time including both conditions and diagnostics: 0.29s.

Full uses all but the last training pair for discovery and the last as evidence. Two-pair uses only the first two pairs; extra training outputs are deliberately withheld. Program vocabularies are frozen before evidence, unlike full refitting in 003. These are controlled inference experiments, not complete production-solver scores.

## Was improvement possible?

| Condition | Tasks | Any surviving family | ≥2 surviving families | Competitive (disagree) | Fixed-order correct | Family-choice ceiling | Recoverable baseline errors | Correct baseline vulnerable to switching | Full-program-pool ceiling |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 24 | 4 | 1 | 0 | 2 | 2 | 0 | 0 | 3 |
| two-pair | 24 | 4 | 1 | 0 | 2 | 2 | 0 | 0 | 4 |

Competitive membership uses predictions only, never scoring labels. The family-choice ceiling chooses among the SAME representatives available to every policy. The full-program ceiling additionally changes programs and is not attainable merely by selecting a family. Recoverable errors are the actual improvement opportunity; fewer than five supports descriptive cases only.

## Accuracy and paired comparisons

| Condition | Cohort | Policy | Tasks | Task exact match | Test-grid exact match | Prediction coverage | Wins vs fixed | Losses vs fixed | Unused training grids correct |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| full | all | fixed order | 24 | 2/24 (8.33%) | 2/25 (8.00%) | 4/25 (16.00%) | 0 | 0 | n/a (0/0) |
| full | all | predictive evidence | 24 | 2/24 (8.33%) | 2/25 (8.00%) | 4/25 (16.00%) | 0 | 0 | n/a (0/0) |
| full | all | minimum answer entropy | 24 | 2/24 (8.33%) | 2/25 (8.00%) | 4/25 (16.00%) | 0 | 0 | n/a (0/0) |
| full | competitive | fixed order | 0 | n/a (0/0) | n/a (0/0) | n/a (0/0) | 0 | 0 | n/a (0/0) |
| full | competitive | predictive evidence | 0 | n/a (0/0) | n/a (0/0) | n/a (0/0) | 0 | 0 | n/a (0/0) |
| full | competitive | minimum answer entropy | 0 | n/a (0/0) | n/a (0/0) | n/a (0/0) | 0 | 0 | n/a (0/0) |
| two-pair | all | fixed order | 24 | 2/24 (8.33%) | 2/25 (8.00%) | 4/25 (16.00%) | 0 | 0 | 5/32 (15.62%) |
| two-pair | all | predictive evidence | 24 | 2/24 (8.33%) | 2/25 (8.00%) | 4/25 (16.00%) | 0 | 0 | 5/32 (15.62%) |
| two-pair | all | minimum answer entropy | 24 | 2/24 (8.33%) | 2/25 (8.00%) | 4/25 (16.00%) | 0 | 0 | 5/32 (15.62%) |
| two-pair | competitive | fixed order | 0 | n/a (0/0) | n/a (0/0) | n/a (0/0) | 0 | 0 | n/a (0/0) |
| two-pair | competitive | predictive evidence | 0 | n/a (0/0) | n/a (0/0) | n/a (0/0) | 0 | 0 | n/a (0/0) |
| two-pair | competitive | minimum answer entropy | 0 | n/a (0/0) | n/a (0/0) | n/a (0/0) | 0 | 0 | n/a (0/0) |

Evidence selects the largest surviving/discovery program ratio. Minimum answer entropy selects the lowest mean per-test-input entropy under uniform surviving syntax. Fixed-order tie breaking (tolerance 1e-12) applies throughout. Neither sees scoring outputs. Undefined is a prediction category for entropy, but never correct; empty families are unselectable.

## Representation uncertainty

| Condition | Tasks retaining any family | Mean initial viable families | Mean final viable families | Mean initial family H | Mean posterior family H | Evidence-choice changes with grid length prior | Unused training prediction coverage (fixed/evidence/min-H) |
|---|---:|---:|---:|---:|---:|---:|---:|
| full | 4 | 1.250 | 1.250 | 0.250 | 0.250 | 0 | n/a (0/0) / n/a (0/0) / n/a (0/0) |
| two-pair | 4 | 1.250 | 1.250 | 0.250 | 0.250 | 0 | 7/32 (21.88%) / 7/32 (21.88%) / 7/32 (21.88%) |

Initial family prior is uniform over discovery-viable families, conditional on discovery; posterior mass is proportional to evidence probability. Entropies are bits over named families, not semantic equivalence classes. Some object labels describe identical partitions on these inputs. A single observed answer can increase or decrease entropy. Grid-prior sensitivity uses q(length) ∝ 2^-length / K^length per program; object priors remain uniform.

## Competing cases and missed within-family solutions

Includes every competitive task and every task with a pool-correct program but no correct family representative.

| Condition | Task | Surviving families | Fixed / evidence / min-H family | Correct fixed/evidence/min-H | Family / pool oracle | Initial → posterior family H |
|---|---|---:|---|---|---|---:|
| two-pair | 67385a82 | 2 | zero-4-mono / zero-4-mono / zero-4-mono | false/false/false | false/true | 1.000→1.000 |
| full | b8825c91 | 1 | grid / grid / grid | false/false/false | false/true | 0.000→0.000 |
| two-pair | b8825c91 | 1 | grid / grid / grid | false/false/false | false/true | 0.000→0.000 |

## Recoverable errors and harmful switches

For each recoverable baseline error or actual harmful switch, list ALL surviving representatives. Correctness is post-hoc scoring, never a selection input.

| Condition | Task | Family | Discovery → surviving programs | Evidence probability | Answer entropy | Representative correct | Program |
|---|---|---|---:|---:|---:|---|---|

Compact family diagnostics and representative predictions for all tasks are saved in families.jsonl. No new DSL features or policies were added after observing results.
