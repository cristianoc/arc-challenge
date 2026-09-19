# Retrospective repair audit evidence

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
