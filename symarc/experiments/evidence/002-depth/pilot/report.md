# Depth sensitivity run report

Public training only; pilot cohort; 12 workers; exhaustive depth 2 versus 3, uniform syntax prior. No symmetry or search fallback. Shortest fitting program predicts; original enumeration order breaks ties.

Full depth-2 census: 400 tasks, 55 fitting, 32 determined. Census wall time: 2.19s. Total computation and pool-writing wall time: 29.62s.

| Cohort | Depth | Tasks | Fitting | Task exact match | Grid exact match | Fully predicted tasks | Oracle tasks | Oracle headroom | Determined | Mean programs | Mean program H | Mean answer H |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ambiguous | 2 | 6 | 6 | 5 (83.33%) | 5/6 | 6 | 6 | 1 | 0 | 61.000 | 5.420 | 1.699 |
| ambiguous | 3 | 6 | 6 | 5 (83.33%) | 5/6 | 6 | 6 | 1 | 0 | 3478.000 | 11.168 | 2.575 |
| unfitted | 2 | 6 | 0 | 0 (0.00%) | 0/7 | 0 | 0 | 0 | 0 | n/a | n/a | n/a |
| unfitted | 3 | 6 | 0 | 0 (0.00%) | 0/7 | 0 | 0 | 0 | 0 | n/a | n/a | n/a |
| all selected | 2 | 12 | 6 | 5 (41.67%) | 5/13 | 6 | 6 | 1 | 0 | 61.000 | 5.420 | 1.699 |
| all selected | 3 | 12 | 6 | 5 (41.67%) | 5/13 | 6 | 6 | 1 | 0 | 3478.000 | 11.168 | 2.575 |

Means condition on fitting tasks; depths may have different denominators. Entropies are bits. Oracle is any single fitting program correct on all test outputs; it is a post-hoc ceiling, not a selection rule. Empty pools count as incorrect/unpredicted. Grid scores use all test grids. This deliberately selected cohort is not a corpus accuracy estimate.

| Task | Cohort | Primitives | Programs d2→d3 | Joint predictions d2→d3 | New joint predictions | Answer H d2→d3 | Selected correct d2→d3 | Oracle d2→d3 | Undefined programs d3 | Depth-3 task seconds |
|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|
| b8825c91 | ambiguous | 247 | 4→444 | 2→15 | 13 | 1.000→2.168 | false→false | true→true | 0 | 27.43 |
| 67e8384a | ambiguous | 216 | 70→3781 | 2→5 | 3 | 0.108→0.213 | true→true | true→true | 2 | 3.11 |
| 9172f3a0 | ambiguous | 160 | 45→1594 | 13→50 | 37 | 3.148→4.500 | true→true | true→true | 0 | 1.04 |
| d631b094 | ambiguous | 135 | 58→2620 | 8→15 | 7 | 1.390→2.068 | true→true | true→true | 162 | 1.30 |
| 3c9b0459 | ambiguous | 247 | 117→10389 | 19→109 | 90 | 2.336→3.729 | true→true | true→true | 0 | 4.26 |
| a416b8f3 | ambiguous | 187 | 72→2040 | 8→8 | 0 | 2.213→2.771 | true→true | true→true | 0 | 2.44 |
| 1190e5a7 | unfitted | 135 | 0→0 | 0→0 | 0 | n/a→n/a | false→false | false→false | 0 | 5.32 |
| c444b776 | unfitted | 216 | 0→0 | 0→0 | 0 | n/a→n/a | false→false | false→false | 0 | 17.30 |
| 85c4e7cd | unfitted | 216 | 0→0 | 0→0 | 0 | n/a→n/a | false→false | false→false | 0 | 9.90 |
| 3de23699 | unfitted | 135 | 0→0 | 0→0 | 0 | n/a→n/a | false→false | false→false | 0 | 5.18 |
| 1c786137 | unfitted | 216 | 0→0 | 0→0 | 0 | n/a→n/a | false→false | false→false | 0 | 21.78 |
| 53b68214 | unfitted | 135 | 0→0 | 0→0 | 0 | n/a→n/a | false→false | false→false | 0 | 3.56 |

Joint prediction counts include undefined and identify agreement only on the observed test inputs. Saved pools contain every fitting program at length ≤3 and its predictions; length filters recover depth 2 without rerunning. Per-task seconds overlap across workers and include scoring/writing; process peak RSS is recorded in stderr.txt by the runner.
