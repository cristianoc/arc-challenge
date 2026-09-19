# 016 — Construct distinctions from labelled conflicts

## Question and scope

Does selecting conjunctions of relational features from exact training conflicts
improve prediction over selecting one feature? Does same-colour reachability add
useful distinctions beyond local and colour-global aggregation?

Baseline: `d9e4ff652bcc07e3d0066175b99c67a0ae088e83` (015). This is a finite
feature-construction/selection experiment, not primitive-free discovery.
The motivating `00d62c1b` and previously inspected `e88171ec` are development
fixtures, excluded from primary counts. All public ARC-AGI-2 tasks at
`f3283f727488ad98fe575ea6a5ac981e4a188e49` are processed. These datasets have
been used in this project before; no untouched-holdout claim is made.

The crucial change from 015 is that every retained feature has a labelled
necessity certificate: removing it merges two cells for which no shared output
operation exists. This makes a proposed distinction accountable, not true on
future inputs. The construction and final choice are both relearned inside an
outer withheld-demonstration evaluation.

## Supplied grammar

Inputs are pointed grids. Primitive fields are raw colour, canvas-border
membership, and the number of four-neighbours with the same colour. The supplied
scopes are: the point plus its same-colour immediate neighbours (`near`); all
points reachable by same-colour four-neighbour edges (`reach`); and all points
of that colour anywhere in the grid (`colour_peers`). Reach is reflexive and
transitive. Connectivity/closure is a supplied constructor, NOT discovered
recursion or a newly invented concept.

Generate each scope with each of four aggregators: count, existence of a
canvas-border point, minimum same-colour degree, maximum same-colour degree.
Together with the three primitive fields, this gives 15 terms, identically
supplied to every task. No rectangles, learned numerical thresholds, raw
coordinates, example IDs, task-specific colours, or exceptions are added to
this structural family. Counts are exact categorical values.

For example, `reach.any_border(u)` is the composed predicate

$$
\exists v.\;\operatorname{SameColourStep}^{*}(u,v)\land\operatorname{Border}(v).
$$

Terms are generated before seeing labels; labels guide which combinations are
needed. The experiment does not claim to invent operators outside this grammar.

## Rule semantics and conflict constraints

Reuse 015's output language: the ten constant colours and CopyCentre. For each
labelled pointed input `(u,y)`, let

$$
A(u,y)=\{\theta:E(\theta,u)=y\}.
$$

An abstraction key is the tuple of selected feature values. A key is consistent
when the intersection of its allowed-action sets is nonempty. Unknown keys and
multiple actions that evaluate differently abstain; there is no default copy.

For this particular action language, any empty intersection has a two-cell
witness. There must be a changed cell and another cell with a different output:
copy fails the changed cell, and no constant fits both. Conversely, absent
such a pair, either every occurrence is unchanged or all outputs agree.
This small-witness property need not hold for arbitrary action languages.

For an incompatible pair `(u,y),(v,z)`, define its separator set

$$
S_{u,v}=\{f:f(u)\ne f(v)\}.
$$

A feature set F is sufficient on the labelled training data exactly when it
meets every such separator set. An empty separator certifies that this entire
feature vocabulary cannot express the labels with the supplied actions.

The search starts with no features, obtains a conflict, and branches on every
feature separating it. It enumerates all inclusion-minimal sufficient subsets
of size at most three, with no node timeout or heuristic beam. Single-feature
and no-reach controls use the same procedure. Every chosen feature f receives
a pair that is indistinguishable under `F - {f}` and incompatible in action,
but distinguished by f. These certificates concern training sufficiency, not
causal relevance or test correctness.

This is a finite conflict-clause/hitting-set construction and uses established
ideas of relational feature learning and abstraction refinement; there is no
novelty claim for the construction itself. Relevant precedent for relational
feature generation is Dutta and Srinivasan, *Consensus-Based Modelling using
Distributed Feature Construction* (2014), arXiv:1409.3446. The question here is
what this explicit, auditable procedure achieves on the declared ARC setting.

## Frozen policies

- `atoms_cv`: at most one term, choose using internal demonstration validation.
- `refined_cv`: at most three terms, same selection procedure.
- `refined_cost`: choose fewest terms, then lowest supplied constructor cost,
  then stable syntax order, without using validation to select.
- `refined_consensus`: retain all best internal-validation candidates and emit
  a cell only when all their action interpretations agree.
- `no_reach_cv`: at most three terms, but remove the four reach-scope terms.
  Local scopes and colour-global scopes remain. This isolates the reach
  constructor relative to this menu, not all forms of structural reasoning.
- `coordinates`: one key `(height,width,row,column)`, same action table.
  It is an explicit positional fitting control, not structural discovery.
- `identity`: unchanged input, no learned fallback.

Constructor costs: colour/border cost 1, degree costs 2; scope costs are 2 for
near/colour_peers and 3 for reach; aggregators cost 1 for count, 2 for any_border,
3 for min/max_degree. These are declared tie-break priors, not measured MDL.

Internal validation scores each fitting feature tuple by fitting its action
table without each distinct demonstration input, then predicting that grid.
Rank by exact whole-grid predictions, then the sum of per-grid correct-known-
cell fractions; break ties by feature count, constructor cost and syntax order.
Candidate features and selection are learned using this training set. These
internal scores are selection evidence, NOT a clean held-out evaluation of the
selected model.

For the outer evaluation, withhold each distinct input and all its duplicates,
then rerun feature search AND internal model selection on the remainder. Only
then predict the outer held-out labels. Thus the complete algorithm, not merely
the table after representation selection, is evaluated. Reusing input-derived
feature values is allowed; no outer output contributes to inner construction.
Record outer exactness and, secondarily, a strict gate that keeps a final query
answer only if every outer grid was exact. This gate can only abstain.

## Execution and metrics

Reuse 014's prepare command to project demonstrations/query inputs into a
problem file separate from query labels. The prediction process reads only
projected problems. Hash predictions before running the separate score command.
No corpus pilot or score is used to alter the vocabulary, bounds or policies.
Twelve workers, deterministic ordering, no random seed for the corpus run.
Wall time is provenance, not a comparison with earlier Rust implementations.

Training-only eligibility: at least two distinct demonstration inputs and equal
input/output dimensions for every demonstration. Query shape changes count as
failures, not exclusions. Exclude the two development fixtures from primary
split-level counts, but retain their predictions as development checks.

Primary comparison: refined_cv versus atoms_cv for correct complete tasks,
complete wrong tasks and abstentions. Also report train fits, attainable
candidate oracle coverage, no_reach comparison, cost versus validation,
consensus, outer-validation passes and the strict gate's error/coverage tradeoff.
Report wrong defined cells and changed-cell results only as descriptive
supplements; cells are not independent observations.

Separate failure modes: full-vocabulary conflict, no sufficient subset within
the three-feature bound, unknown query keys, action ambiguity, wrong defined
prediction, and selector loss relative to the finite candidate oracle.
Do not interpret zero coverage as an unsupported universal impossibility claim.

Thirteen synthetic controls passed before the corpus run, including exhaustive
small action-intersection checks and comparison of search with exhaustive
subset enumeration on 120 deterministic synthetic tables. An independent
post-run audit will recompute selected predictions, training-necessity
certificates, scoring and duplicate isolation. No stable-core integration is
part of this experiment. Retain a candidate only if useful predictive coverage
and transparent limitations justify it; otherwise close with the certificates.

## Source freeze and reproduction

Pre-corpus SHA-256:
- `run.py`: `4caedc1938c320527df0e63a31607db7fbcb17ff268f22923b2f6798a86da387`
- `test_run.py`: `6d2940eaf03a57955b81a7bca7519ea43010c8c9105b6065d8cf1e7d7fce6c2b`

```sh
E=symarc/experiments/016-conflict-refinement
python3 "$E/test_run.py"
python3 symarc/experiments/014-cross-demonstration-transport/run.py prepare --data /path/to/ARC-AGI-2/data --out /tmp/arc016/input
python3 "$E/run.py" predict --problems /tmp/arc016/input/problems.json --out /tmp/arc016/run --workers 12
python3 "$E/run.py" score --predictions /tmp/arc016/run/predictions.json --answers /tmp/arc016/input/answers.json --out /tmp/arc016/run
```

The protocol is committed before corpus execution. The active entry is also
registered in the local working ledger before the run and consolidated in the
branch ledger with the findings. Stable code and accepted mathematics remain
unchanged.
