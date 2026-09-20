# Retrospective repair audit evidence

Ten repairs were hand-written after inspecting official test answers. They
preserve 25/25 training pairs and pass 14/14 known test pairs. The runner checks
these supplied repairs; it does not discover them. The colour and geometry
checks also use manually chosen task-specific assumptions.

- `baseline.json`: per-example outcomes for all 120 external original solvers.
- `repair_results.json`: ten repaired witnesses, exact matches and metamorphic checks.
- `provenance.json`: original source/data revisions and solver blob hashes.
- `reproduction.json`: environment and hashes from the repeat check during import.

The [protocol](../../006-repair-audit/README.md) and
[case catalogue](../../006-repair-audit/repair_patterns.md) explain scope and limits.
The initial audit was performed before registration and inspected test answers.
This is not a SymArc baseline or a blind benchmark. Historical timings were not
retained; reproduction duration is not a performance comparison.

External inputs are fetched at pinned revisions and verified using
`../../006-repair-audit/inputs.json`; regenerable caches are not committed.
