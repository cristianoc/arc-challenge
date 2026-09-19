# Supplementary training audit (registered after coverage, before this audit)

The frozen 015 coverage run found only four nondevelopment structurally fitting
training tasks and zero evaluation tasks. Its five-task threshold is unmet.
Do not execute the broad query-scoring comparison or retune it to pass that gate.
The pre-run source is unchanged. Run a supplementary diagnostic instead.

For each structurally defined representation, independently enumerate its 11
possible output actions on each role's training occurrences. The role admits an
action exactly when either all output colours are equal, or every cell is
unchanged. Consequently every rejection has a certificate of at most two
occurrences: choose one changed cell and another cell with a different desired
output. No literal constant fits both, and CopyCentre fails the changed one.

Record whether contradictions already occur inside a single demonstration or
only after combining demonstrations. This distinguishes a representation/rule
expressivity problem from a shortage of cross-demonstration validation evidence.

As a training-only expressivity diagnostic, refine each role key with the
original input-cell colour. This is not a new test-scored solver. It measures
whether the rejected interpretation is rescued by allowing colour-conditioned
rules rather than assigning one output action to the entire role. Keep the
same masks, domains, output actions and data. Report task-level counts separately
from candidate counts; exclude e88171ec from primary counts as before.

The fixed known e88171ec fixture may be inferred and checked separately. Its
rectangle hypothesis was supplied after earlier test inspection; it is not a
blind result even when this new learner selects it without reading its test
answer. No other query output will be read by this investigation.

Audit source SHA-256 before its execution:
`a439d88444aa9f7f7a4e1904218ed3a91643580d154b104d3e8876714b263a9c`.

Run `python3 audit.py --problems /tmp/arc015/input/problems.json --out /tmp/arc015/run`.
All workers receive only demonstration inputs and outputs. Structural-rule and
role-plus-colour counts are expressivity measurements, not generalisation
scores, solver gains or evidence of automatically discovered primitives.
