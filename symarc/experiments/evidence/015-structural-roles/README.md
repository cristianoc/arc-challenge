# 015 — Share an operation, not necessarily an output

The known rectangle fixture now works as an executed selection example. Within
our supplied family, withheld-demonstration prediction selects the rectangle
rule over local lookup. But only four other public training tasks and no
evaluation tasks fit the structural family. The prespecified five-task minimum
was not met, so the broad query-scoring comparison was not run.

A supplementary training-only audit found that 3,920 of 3,930 incompatible
structural candidates already contradict a single demonstration. The immediate
bottleneck is this representation/rule family's expressivity, not evidence
available only from additional demonstrations.

## Protocol and scope

Baseline `f285542`; [protocol](../../015-structural-roles/README.md) and source
hashes published at `ab39914` before corpus execution. The active entry was
registered in the local working ledger before that run; the final branch ledger
records the result. The supplementary [audit protocol](../../015-structural-roles/TRAINING_AUDIT.md)
was registered after the coverage result, at `8374868`, before its own run.

Data: all 1,000 training and 120 evaluation tasks at ARC-AGI-2 `f3283f7`.
Evaluation-task demonstrations were used for coverage, not their query answers.
These public data were used previously in this project; no untouched-holdout
claim. Known `e88171ec` is excluded from primary counts. Twelve workers and no
random seed. Stable solver and accepted mathematics are unchanged.

There are 38 supplied representations: raw and colour-canonical patches at
radii 0/1/2/3; and three masks for each literal colour 0..9 (all monochromatic
4-connected components, unique largest component, unique largest rectangle).
Each structural key retains outside/boundary/interior role. Absent colours and
tied maxima make its domain undefined; they are not arbitrarily tie-broken.

Each representation has both output languages: literal colours, or those same
constants plus CopyCentre. All consistent actions are retained. Unknown keys,
conflicting evaluated actions and undefined domains abstain. No default copy.
First-order, validation and conservative-consensus selectors were frozen, but
only coverage and the isolated development fixture reached execution.

## A correction to the quotient formulation

Different-coloured cells can share the operation “copy this cell” without
sharing their final output. The actual model is

$$
p(u)=E(h(a(u)),u),
$$

where u is a pointed grid, a chooses a context, h selects an action, and E
decodes it. The decoder's access to the input is explicit: CopyCentre reads the
centre colour. This is a quotient for rule selection, NOT a claim that the
complete predictor factors through the role key alone.

For finite action language Theta, retain

$$
V_z=\bigcap_{(u,y):a(u)=z}\{\theta\in\Theta:E(\theta,u)=y\}.
$$

A query is determined only when all surviving actions evaluate to the same
colour. With literals this reduces to equal output labels on fibres; with copy
it can share an operation across different output labels. This is an elementary
finite consistency construction, not an entropy objective or a novelty claim.
The representations and action language are explicit priors.

## Primary result: coverage criterion failed

| Original corpus split | Eligible, excluding development | Structural/literal fits | Structural/copy-or-literal fits |
|---|---:|---:|---:|
| Training (999 remaining tasks) | 679 | 3 | 4 |
| Evaluation (120 tasks) | 81 | 0 | 0 |

The four nondevelopment tasks are `6f8cd79b`, `b1948b0a`, `bb43febb`, `c8f0f002`.
All four have a structural candidate predicting every withheld demonstration.
The family is nevertheless below the frozen minimum; no broad query predictions
or query-accuracy table were produced, and the threshold was not relaxed.

This rejects the narrow family as a basis for the intended selector comparison,
not structural reasoning generally. One action per outside/boundary/interior
role is not a generic language for all object or rectangle programs.

## Exact rejection audit

For a role's labelled occurrences, a copy/literal action exists iff either all
target colours agree OR every occurrence is unchanged. Every rejection thus has
a two-cell certificate: one changed cell, and another requiring a different
output colour. Copy fails the changed cell; no constant fits both. Conversely,
if no such pair exists, either everything is unchanged or all targets equal the
changed cell's target. The audit independently enumerates actions, rather than
reusing the learner's bitset intersections.

| Training-only audit, excluding e88171ec | Training split | Evaluation split |
|---|---:|---:|
| Tasks with some parser defined on every demonstration | 665 | 75 |
| Defined structural candidates | 3,429 | 508 |
| Fitting candidates | 7 | 0 |
| Rejected candidates | 3,422 | 508 |
| Rejected already within a demonstration | 3,412 | 508 |
| Rejected only across demonstrations | 10 | 0 |

Candidate counts are not independent tasks. Almost all contradictions appear
before the extrapolation question. A supplementary expressivity diagnostic
refined role keys with the input-cell colour: training-task fits rose from four
to seven; evaluation remained zero. This diagnostic was NOT test-scored or
substituted into the frozen policy.

### Concrete training contrast: 00d62c1b

In demonstration 0, cells `(2,3)` and `(0,0)` are both zero cells on the boundary
of the union of zero components. The first must become 4; the second remain 0.
A shared constant/copy rule cannot serve that role, and input colour does not
separate the points either.

Inspection of these training-only points exposes a candidate distinction: the
first lies in a four-cell zero component that does not touch the canvas border;
the second in a 48-cell component that does. Boundary reachability is therefore
a concrete next feature proposal motivated by labelled training evidence. This
observation is not a checked repair of the whole task or a new test result.
All coordinates are zero-based; feature values are retained in results.json.

## Positive development mechanism: e88171ec

Eleven candidates fit the demonstrations: five local representations under each
of the two action languages, plus the largest-zero-rectangle/copy-language
candidate. Every fitting local candidate predicts 0/3 whole withheld
demonstrations; the rectangle candidate predicts 3/3. The fixed validation
selector and its tied-best consensus select the rectangle candidate; fixed
order selects a local candidate. Only the rectangle prediction matches the
previously inspected query answer exactly.

The inferred action sets are outside={CopyCentre}, boundary={constant 0,
CopyCentre}, interior={constant 8}. Boundary alternatives agree on this domain;
we do not claim a unique program syntax was identified. The rectangle hypothesis
was supplied after earlier test inspection in 014. This demonstrates executable
selection on a development fixture, not blind abstraction discovery. Its
predictions are persisted before rereading that known answer.

## Decision and next target

No solver integration and no broad selector-performance claim. Retain the exact
learner, tests and rejection audit as reusable controls and certificates.

The next productive direction is to construct candidate distinctions from
labelled training contradictions, rather than repeatedly invent whole-grid
languages and hope they cover the corpus. Require each added feature to expose
the contradiction it removes; relearn the construction inside held-out
demonstrations. Raw coordinates and example identifiers should remain explicit
memorisation controls, not count as structural discovery. This still requires
a declared primitive language; useful feature construction and scope selection
are the missing mechanism, not another scalar score on the current candidates.

## Checks and reproduction

Fourteen synthetic tests pass, including exhaustive 2x3 binary-grid rectangle
checks. Eligibility and first raw-radius selection agree with 014 on all 1,120
tasks. A second 12-worker coverage run was byte-identical. The independent action
audit covers all 3,937 nondevelopment defined structural candidates and checks
3,930 two-cell rejection certificates. The isolated development script
reproduces its predictions and results. No Actions run was used for 015.

Coverage SHA-256: `2a16fa403ebfeabfee3278a1db3102c8f263b1df5894cb98869e47b8fec67e11`.
[Compact results](results.json) retain decisive counts and provenance; full
coverage/fold records and rejection certificates are regenerable and included
in the accompanying investigation archive.

From repository root, prepare projected problems with 014's run.py, then:

```sh
E=symarc/experiments/015-structural-roles
python3 "$E/test_run.py"
python3 "$E/run.py" coverage --problems /tmp/arc015/input/problems.json --out /tmp/arc015/run --workers 12
python3 "$E/audit.py" --problems /tmp/arc015/input/problems.json --out /tmp/arc015/run
python3 "$E/development.py" --problems /tmp/arc015/input/problems.json --task /path/to/ARC-AGI-2/data/training/e88171ec.json --out /tmp/arc015/run
```

The protocol's broad predict/score commands describe the frozen, unexecuted
stage. They were not run after the coverage stop. Coverage took 4.650s locally;
this is provenance, not a speed comparison with earlier experiments.
