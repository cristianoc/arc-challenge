# 009: what can the examples themselves say about a symmetry law?

Registered before data runs. Baseline `e407f53`. Apply the same three candidate
laws to all 120 tasks pinned by 006: arbitrary permutation of all ten colours
(including 0), the eight square symmetries D4 acting on rectangular grids, and
their product. The same action is applied to inputs and outputs. No task-specific
fixed colours, selected orientations, original programs or repaired programs.
The three law families themselves are supplied, not discovered.

Question: does there exist a deterministic equivariant grid-to-grid function
fitting the given examples? Check every transport between supplied inputs,
including each input to itself. A transformation that preserves an input but
changes its required output is a contradiction, even without two distinct
examples. Arbitrary colour permutations are checked symbolically via the partial
bijection forced by input cells; account for every completion on unused colours.
Do not treat absence of contradiction as evidence that the intended rule obeys
the law. No ranking or law selection is implemented.

First compute compatibility from training pairs only. Then separately check
training plus known test pairs to count additional contradictions. No decision
uses test labels. All tasks are public development data and their answers were
previously available during 006; this is not a blind validation claim. This
checks whether a law can fit the examples, not whether an existing solver is
itself equivariant. No synthesized outputs are treated as ground truth.

Record per-law training contradictions, compatible cases, cross-example input
transports, and contradictions first exposed by adding test pairs. Retain a
concrete transformation witness for each contradiction. Distinguish no observed
cross-example transport from positive evidence for the law. Whole corpus: 120
tasks, 12 workers, 600-second limit; no speed comparison. Verify input hashes.
Validate the symbolic checker against brute-force permutations on a three-colour
synthetic domain, plus identity, asymmetric-output/stabilizer, unused-output-colour
and cross-example contradiction controls before corpus execution.

Close as a diagnostic, retain results and runnable source history. No production
integration or accepted-math change. Explain the compatibility criterion and its
limits locally. No post-result changes to law families or task selection.
