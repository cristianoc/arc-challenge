# Experiment record

Compact findings and decisions, retained even when code and raw outputs are
deleted. This is an experiment ledger, not a code changelog or a theory history.
Status meanings and code-retention rules are in [README.md](README.md).

### 001-pruning — Does aggressive hypothesis pruning improve predictions?
Status: closed — no integration.
Baseline: `be25423`; core hash `0633d22fb7d1c9d4699fecf4ff851b59cb97c638bcbd6be17db0176865331146`; experiment implementation `f7f4b9b`; 2026-09-18.
Comparison: 400 public training tasks only; 55 have a depth-2 fitting program, 32/55 already have unanimous defined test answers. Frozen uniform syntax class, no repair/fallback; no symmetry vs fixed/random/marginal-pruning policies; seeds 0–4; 12 workers; 500,000 charged checks per task/policy; closure caps 300 and 1000.
Result: at **both caps**, no symmetry and fixed order solve 53/400; marginal pruning 51/400 in every seed; random order mean 51/400 (range 50–52). Fixed/pruning reduce mean program entropy by 0.41/0.57 bits and mean answer entropy from 0.68 to ~0.02 bits, but leave 2/4 unanimous wrong tasks. Pruning loses all initially correct programs on 3 tasks versus fixed's 1. No budget exhaustion. At cap 1000 every pruning trial closure completes, so its losses persist without closure sampling. Full five-seed rounds take 10.71s / 13.86s including shared preparation.
Decision: reject maximum pruning as a standalone selection objective in this setup. It increases certainty while discarding correct explanations. This does not reject entropy as a diagnostic, other priors, or other selection objectives. No evaluation-set run or stable-core change. This compares selection/stopping policies within a common functionality prefilter, not the complete stable solver; it is exploratory evidence on the same 400 tasks across seeds.
Counterexamples: `1cf80156`: fixed 100→58 programs, correct `cropBBox`; pruning 100→6, unanimously wrong `cropBBox ; dedupCols`. `1f85a75f`: fixed 70→25, correct `cropLargest`; pruning 70→2, unanimously wrong `cropLargest ; recolour 2 0`.
Code/evidence: completed code deleted; runnable implementation is preserved at `f7f4b9b`. Retained [protocol](evidence/001-pruning/protocol.md), direct [cap-300 report](evidence/001-pruning/cap-300/report.md) and [cap-1000 report](evidence/001-pruning/cap-1000/report.md), with original manifests and logs. Depth sensitivity is examined in 002-depth.

### 002-depth — Does depth 3 create useful headroom for program entropy?
Status: active.
Baseline: `3b8cf66`; baseline core hash `0633d22fb7d1c9d4699fecf4ff851b59cb97c638bcbd6be17db0176865331146`; 2026-09-18. Minimal behavior-preserving core change omits unused final enumeration frontier.
Comparison: exhaustive depth 2 vs 3; all 23 depth-2 ambiguous training tasks plus 24 deterministic hash-selected unfitted tasks; 12-task pilot first; 12 workers, 600s run timeout. Same primitives and shortest-program selection; no symmetry/fallback. Primary metrics oracle availability and oracle-minus-selected headroom; distinct predictions distinguish extra syntax from answer diversity.
Result: not run yet.
Decision: establish depth sensitivity before another entropy-policy comparison; no policy integration in this experiment.
Code/evidence: [protocol and runner](002-depth/README.md).

## Entry format

Use one short entry per experiment; update it in place. Keep decisive facts
here so the entry remains useful after its directory is deleted.

```text
### NNN-short-name — short question
Status: active | candidate | integrated | closed
Baseline: revision + core hash; date
Comparison: control vs treatment; tasks, depth/caps, seeds, compute budget
Result: decisive numbers, runtime, uncertainty/limitations; or "not run yet"
Decision: conclusion and integration criterion met/missed; next step if retained
Code/evidence: path while retained, or "deleted"; integration revision if applicable
```
