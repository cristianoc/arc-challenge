# 008: can explicit symmetry contracts distinguish the repair pairs?

Registered before execution. Baseline `fd1d166`; stable core unchanged.
This is a diagnostic of ten retrospective original/repaired program pairs from
006, not blind synthesis or an independent evaluation. Candidates and symmetry
contracts were informed by inspected answers in 006. No new repair is proposed.

Question: although each pair agrees on all supplied training examples, does
requiring the inherited task-specific equivariance contracts distinguish them?
For each labelled training pair (x,y), require P(T(x)) = T(y). This is an added
semantic assumption, not information implied by the finite training pairs.

Frozen probes: every single transposition of two colours not designated fixed;
fixed {0}, except {0,5} for e3721c99 and {0,3,4} for 221dfab4. Include horizontal
reflection for 1ae2feb7 and transpose for 135a2760 and 221dfab4. Same input/output
transformation. Do not assume other geometric symmetries. Identity training
checks first. Count identity-action probes and distinct transformed labelled
pairs separately: inactive colour swaps are not independent evidence. These
are exhaustive single swaps, not enumeration of the full permutation group.

Use the existing 006 implementations without editing them. Candidate choice:
retain exactly those programs passing every original and transformed training
pair. Report repair-only, original-only, both, neither; never break ties using
the official test answer. Selection is permitted to abstain. Report per-contract
failures, exceptions/timeouts separately and save the first mismatch per program
per contract. Test data is removed before diagnostic execution; known test
results are cited only from retained 006 evidence. No evaluation-score claim.

All ten pairs, 25 training examples; 12 processes, serial run, each candidate
call limited to 5 seconds and whole run to 600 seconds. No tuning after results.
Verify pinned source and data hashes from 006 before discarding test pairs.
Two controls: identity and correct colour renaming on synthetic grids; a
colour-equivariant histogram/path pair should remain indistinguishable until a
labelled path with noncontiguous repeated colours is supplied (007 already
checks the latter). No benchmark speed comparison or stable-core changes.

Close after recording outcomes and limitations. Preserve protocol, compact
per-pair results, concrete counterexamples and a reproducible source commit;
remove completed diagnostic code. No integration criterion: this study asks
which constraints discriminate known candidates, not whether a new solver wins.
