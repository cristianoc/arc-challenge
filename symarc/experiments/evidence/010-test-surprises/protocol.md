# 010: training-supported symmetry patterns broken by intended test outputs

Registered before corpus execution. All 1,120 public ARC-AGI-2 tasks: 1,000
training and 120 evaluation, repository revision
`f3283f727488ad98fe575ea6a5ac981e4a188e49`. Report splits separately; do not confuse
task-level train/test pairs with corpus training/evaluation splits. No private
tasks are accessible. Previous 009 covered the 120 public evaluation tasks.

Same candidate patterns on every task, no hand-picked exceptions:

1. Output symmetry: each of seven nonidentity D4 transformations fixes every
   training output, but not a test output. Require at least two training examples
   and nonidentity coordinate action on every training output and flagged test
   output. Report whether all training outputs are monochrome; these can produce
   incidental symmetry evidence.
2. Conditional symmetry preservation: whenever a training input is fixed by a
   spatial action, its output is too; at least two training inputs provide this
   evidence and the action moves coordinates on them. Flag a test input fixed by
   that action whose required output is not. Highlight the subset where EVERY
   training pair supplies this evidence. These are the strongest inspection leads,
   not automatic proofs of a flawed task.
3. Uniform equivariance compatibility: individually test cyclic groups generated
   by seven D4 actions or 45 colour swaps, same action on input/output. A group
   compatible with training becomes incompatible after test pairs are added.
   Record training self-transports and cross-example transports, including whether
   inputs actually change. Compatibility without informative observed transports
   is not described as a training-supported law. Test all powers, not just the
   generator. These individual laws can expose breaks masked by 009's aggregate
   group-level results.

All patterns and support counts are selected from training pairs before reading
test outputs. Then score intended outputs; no solver or repair is used. Known
public data, descriptive task audit, not a blind benchmark. No probability of
surprise or multiple-testing significance claim. Distinguish a symmetry broken
by a newly asymmetric INPUT from one broken despite a symmetric input.

12 workers; 600s cap. Validate controls for genuine preservation break, output-only
break explained by asymmetric query input, degenerate one-row symmetry, and a
colour-transport contradiction. Save all flags and support counts, per-task input
hashes, compact summary, and concrete grids for the top six preservation leads
ranked by all-training support, then support count descending, then task ID and
transform index. Inspect those cases; no post-result threshold tuning. If there
are fewer than six, inspect all. A task-quality concern remains conditional on
an interpretation and needs an explanation of the intended rule, not just a flag.

Close with evidence and runnable source history; no stable-core or accepted-math
change. This study detects counterexamples to observed patterns; it does not
infer the task author's full intended rule or certify dataset defects.
