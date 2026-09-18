# Depth sensitivity run report

Public training only; selected cohort; 12 workers; exhaustive depth 2 versus 3, uniform syntax prior. No symmetry or search fallback. Shortest fitting program predicts; original enumeration order breaks ties.

Full depth-2 census: 400 tasks, 55 fitting, 32 determined. Census wall time: 2.09s. Total computation and pool-writing wall time: 51.77s.

| Cohort | Depth | Tasks | Fitting | Task exact match | Grid exact match | Fully predicted tasks | Oracle tasks | Oracle headroom | Determined | Mean programs | Mean program H | Mean answer H |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ambiguous | 2 | 23 | 23 | 22 (95.65%) | 23/24 | 23 | 23 | 1 | 0 | 60.609 | 5.647 | 1.615 |
| ambiguous | 3 | 23 | 23 | 22 (95.65%) | 23/24 | 23 | 23 | 1 | 0 | 3294.348 | 11.087 | 2.366 |
| unfitted | 2 | 24 | 0 | 0 (0.00%) | 0/27 | 0 | 0 | 0 | 0 | n/a | n/a | n/a |
| unfitted | 3 | 24 | 0 | 0 (0.00%) | 0/27 | 0 | 0 | 0 | 0 | n/a | n/a | n/a |
| all selected | 2 | 47 | 23 | 22 (46.81%) | 23/51 | 23 | 23 | 1 | 0 | 60.609 | 5.647 | 1.615 |
| all selected | 3 | 47 | 23 | 22 (46.81%) | 23/51 | 23 | 23 | 1 | 0 | 3294.348 | 11.087 | 2.366 |

Means condition on fitting tasks; depths may have different denominators. Entropies are bits. Oracle is any single fitting program correct on all test outputs; it is a post-hoc ceiling, not a selection rule. Empty pools count as incorrect/unpredicted. Grid scores use all test grids. This deliberately selected cohort is not a corpus accuracy estimate.

| Task | Cohort | Primitives | Programs d2→d3 | Joint predictions d2→d3 | New joint predictions | Answer H d2→d3 | Selected correct d2→d3 | Oracle d2→d3 | Undefined programs d3 | Depth-3 task seconds |
|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|
| b8825c91 | ambiguous | 247 | 4→444 | 2→15 | 13 | 1.000→2.168 | false→false | true→true | 0 | 31.67 |
| 67e8384a | ambiguous | 216 | 70→3781 | 2→5 | 3 | 0.108→0.213 | true→true | true→true | 2 | 3.35 |
| 9172f3a0 | ambiguous | 160 | 45→1594 | 13→50 | 37 | 3.148→4.500 | true→true | true→true | 0 | 1.03 |
| d631b094 | ambiguous | 135 | 58→2620 | 8→15 | 7 | 1.390→2.068 | true→true | true→true | 162 | 1.21 |
| 3c9b0459 | ambiguous | 247 | 117→10389 | 19→109 | 90 | 2.336→3.729 | true→true | true→true | 0 | 4.98 |
| a416b8f3 | ambiguous | 187 | 72→2040 | 8→8 | 0 | 2.213→2.771 | true→true | true→true | 0 | 2.47 |
| 496994bd | ambiguous | 112 | 32→846 | 5→5 | 0 | 1.497→1.949 | true→true | true→true | 0 | 1.06 |
| 3af2c5a8 | ambiguous | 91 | 16→257 | 4→4 | 0 | 1.750→1.876 | true→true | true→true | 0 | 0.39 |
| 99b1bc43 | ambiguous | 112 | 50→2048 | 5→5 | 0 | 0.562→0.853 | true→true | true→true | 0 | 1.72 |
| 1cf80156 | ambiguous | 112 | 100→2648 | 7→16 | 9 | 1.873→2.627 | true→true | true→true | 310 | 3.30 |
| 3428a4f5 | ambiguous | 91 | 35→1059 | 2→4 | 2 | 0.094→0.155 | true→true | true→true | 0 | 1.30 |
| 6150a2bd | ambiguous | 216 | 62→2968 | 17→81 | 64 | 3.273→4.792 | true→true | true→true | 0 | 2.48 |
| be94b721 | ambiguous | 216 | 121→11042 | 11→30 | 19 | 1.145→1.881 | true→true | true→true | 14 | 15.60 |
| 0520fde7 | ambiguous | 91 | 36→1124 | 2→2 | 0 | 0.183→0.278 | true→true | true→true | 0 | 0.59 |
| ed36ccf7 | ambiguous | 112 | 25→527 | 6→11 | 5 | 2.080→2.695 | true→true | true→true | 6 | 0.70 |
| 1e0a9b12 | ambiguous | 247 | 45→1576 | 10→11 | 1 | 2.379→3.037 | true→true | true→true | 0 | 9.95 |
| 5582e5ca | ambiguous | 160 | 103→7970 | 7→8 | 1 | 0.755→1.238 | true→true | true→true | 8 | 1.33 |
| f2829549 | ambiguous | 112 | 50→2096 | 2→3 | 1 | 0.141→0.224 | true→true | true→true | 0 | 1.87 |
| 1f85a75f | ambiguous | 160 | 70→3752 | 9→34 | 25 | 1.450→2.404 | true→true | true→true | 6 | 26.02 |
| 74dd1130 | ambiguous | 216 | 107→8694 | 17→91 | 74 | 2.250→3.596 | true→true | true→true | 0 | 3.35 |
| c59eb873 | ambiguous | 187 | 71→3867 | 24→199 | 175 | 4.112→6.337 | true→true | true→true | 0 | 2.18 |
| f25ffba3 | ambiguous | 187 | 56→2441 | 15→66 | 51 | 3.128→4.581 | true→true | true→true | 0 | 4.78 |
| ce4f8723 | ambiguous | 112 | 49→1987 | 3→3 | 0 | 0.287→0.436 | true→true | true→true | 0 | 1.64 |
| 1190e5a7 | unfitted | 135 | 0→0 | 0→0 | 0 | n/a→n/a | false→false | false→false | 0 | 6.14 |
| c444b776 | unfitted | 216 | 0→0 | 0→0 | 0 | n/a→n/a | false→false | false→false | 0 | 21.03 |
| 85c4e7cd | unfitted | 216 | 0→0 | 0→0 | 0 | n/a→n/a | false→false | false→false | 0 | 12.34 |
| 3de23699 | unfitted | 135 | 0→0 | 0→0 | 0 | n/a→n/a | false→false | false→false | 0 | 6.05 |
| 1c786137 | unfitted | 216 | 0→0 | 0→0 | 0 | n/a→n/a | false→false | false→false | 0 | 25.14 |
| 53b68214 | unfitted | 135 | 0→0 | 0→0 | 0 | n/a→n/a | false→false | false→false | 0 | 3.74 |
| 444801d8 | unfitted | 187 | 0→0 | 0→0 | 0 | n/a→n/a | false→false | false→false | 0 | 13.35 |
| d9f24cd1 | unfitted | 72 | 0→0 | 0→0 | 0 | n/a→n/a | false→false | false→false | 0 | 0.74 |
| a78176bb | unfitted | 135 | 0→0 | 0→0 | 0 | n/a→n/a | false→false | false→false | 0 | 5.82 |
| 32597951 | unfitted | 91 | 0→0 | 0→0 | 0 | n/a→n/a | false→false | false→false | 0 | 1.95 |
| 952a094c | unfitted | 247 | 0→0 | 0→0 | 0 | n/a→n/a | false→false | false→false | 0 | 24.92 |
| b0c4d837 | unfitted | 72 | 0→0 | 0→0 | 0 | n/a→n/a | false→false | false→false | 0 | 1.02 |
| a85d4709 | unfitted | 112 | 0→0 | 0→0 | 0 | n/a→n/a | false→false | false→false | 0 | 0.71 |
| 6f8cd79b | unfitted | 55 | 0→0 | 0→0 | 0 | n/a→n/a | false→false | false→false | 0 | 0.10 |
| 6e19193c | unfitted | 91 | 0→0 | 0→0 | 0 | n/a→n/a | false→false | false→false | 0 | 1.30 |
| 67385a82 | unfitted | 72 | 0→0 | 0→0 | 0 | n/a→n/a | false→false | false→false | 0 | 0.39 |
| 5bd6f4ac | unfitted | 247 | 0→0 | 0→0 | 0 | n/a→n/a | false→false | false→false | 0 | 24.99 |
| 2dd70a9a | unfitted | 91 | 0→0 | 0→0 | 0 | n/a→n/a | false→false | false→false | 0 | 1.83 |
| 0dfd9992 | unfitted | 247 | 0→0 | 0→0 | 0 | n/a→n/a | false→false | false→false | 0 | 37.12 |
| 810b9b61 | unfitted | 72 | 0→0 | 0→0 | 0 | n/a→n/a | false→false | false→false | 0 | 1.19 |
| 045e512c | unfitted | 187 | 0→0 | 0→0 | 0 | n/a→n/a | false→false | false→false | 0 | 27.37 |
| 50846271 | unfitted | 91 | 0→0 | 0→0 | 0 | n/a→n/a | false→false | false→false | 0 | 2.67 |
| ff28f65a | unfitted | 72 | 0→0 | 0→0 | 0 | n/a→n/a | false→false | false→false | 0 | 1.13 |
| b94a9452 | unfitted | 160 | 0→0 | 0→0 | 0 | n/a→n/a | false→false | false→false | 0 | 8.80 |

Joint prediction counts include undefined and identify agreement only on the observed test inputs. Saved pools contain every fitting program at length ≤3 and its predictions; length filters recover depth 2 without rerunning. Per-task seconds overlap across workers and include scoring/writing; process peak RSS is recorded in stderr.txt by the runner.
