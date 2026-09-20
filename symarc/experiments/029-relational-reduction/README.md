# 029 — Does reduction reveal relations before individual answers?

Baseline: `2f534a92f728bde2446b368cc03f68a8ad14cd34`. Follow 028 without adding a geometric feature, operation, candidate classifier or output label. Status is recorded in the experiment ledger. This is a finite semantic/label-acquisition investigation, not an ARC benchmark or a runtime comparison.

## Question

028 retains each input's possible labels R_V(x). This is a nonrelational projection of a family of complete classifiers. The exact joint projection is R_V(x,x')={(h(x),h(x')):h in V}; it may be strictly smaller than R_V(x) times R_V(x'). An observation can eliminate a combination without determining either remaining individual answer.

For the fixed semantic domain X, compare W1(V)=sum_x(|R_V(x)|-1), which is exactly 028's binary disputed-region size, with W2(V)=sum_{x<x'}(|R_V(x,x')|-1). Each distinct input class or unordered pair receives unit weight. This is a declared weighting, not a canonical distribution or a scale-independent confidence measure. Empty V is an inconsistency, never certainty.

Record also G2(V)=sum_{x<x'}(|R_V(x)|*|R_V(x')|-|R_V(x,x')|), the spurious combinations introduced by a Cartesian projection. G2 need not be monotone because the marginal product also shrinks. For every observation separately count new excluded pairs whose two coordinates remain individually ambiguous afterwards; these are genuinely relational discoveries, not just restatements of a newly fixed label.

An induced observational equivalence is x~_V x' iff all h in V give them equal outputs. Shrinking nonempty V can only add these equalities. Each surviving h factors through the resulting input quotient. This constructs a finite consequence abstraction relative to H, not a cheap geometric implementation or a target-independent feature. The complementary relation h(x)=1-h(x') is tracked too. Do not call conditional entailment target truth unless realizability is explicitly assumed.

## Frozen experiment

Reuse 027/028's exact 401 input classes and 260 extensionally distinct classifiers, with unchanged costs and source catalogue of all nonempty 3x3 sets. The two initial inputs are masks 255 and 495. Labels come from a noiseless, supplied teacher oracle. Reuse is via imports, not copying the synthesis engine.

Primary comparison: EVERY one of the 260 candidate classifier functions serves in turn as target, not just the four hand-designed rules. This exhaustive closed-world evaluation is not 260 independent natural tasks. Retain separately the eight original geometric teachers for exact baseline comparison with 028.

Three acquisition policies, all using the same inputs, labels and maximum eight extra requests:
- halving: minimize the worst-case remaining semantic hypothesis count;
- marginal: minimize the worst-case W1;
- pair: minimize the worst-case W2.
Only informative queries are considered; ties use increasing original source mask. Stop at a singleton or when no informative source query exists. No prospective oracle label, teacher ID, cheapest-program accuracy or query-bank answer enters selection. Policy trees may share memoized states because choices depend only on V. Score identification and remaining W1/W2, not only the cheapest selected classifier. Report every target and budget, mean/maximum requests censored at eight, individual wins/losses, and any observationally indistinguishable targets. Do not tune the metric after outcomes.

The primary raw source masks are held fixed. The inherited 401-class domain is exact for the declared language at arbitrary sizes; the experiment does not rerun huge geometry banks as if they were independent trials. Costs and alias spellings are irrelevant to the policies after semantic deduplication.

## Mathematical and implementation controls

Synthetic controls before the run: relational shrinkage with unchanged marginals (four binary pairs reduced to equality pairs), arbitrary finite output alphabets, empty family, repeated hypotheses, label complementation, and a three-bit parity family with identical single/pair projections but different full joint relation. Pair consequences do not capture arbitrary higher-order dependencies.

For the geometric runs, retain deterministic relational exclusion witnesses, newly forced equal/complement input pairs, exact before/after version spaces, and the queried label causing the change. Recompute each witness from actual classifier evaluations. Report if no such witness occurs; do not replace it with a hand-selected claim.

Independent auditor: reuse the already-audited 028 semantic family, but recompute pair scores through intersections of hypothesis-column sets, not the scientific pair-bitset code; independently select every reachable query, replay every target, verify monotonicity and all identification scores. Compare all original halving/marginal paths with retained 028 traces. Audit the inherited semantic-pool identity and record that primitive grammar construction is reused. Repeat all scientific outputs byte-for-byte. Record a deterministic order of extracted witnesses rather than choosing a target after scoring.

Prediction traces and policy decisions are persisted and hashed before the separate summary/report stage. Source and test hashes are frozen before scientific execution. The protocol is committed before construction or measurement; a temporary workflow may register its ledger entry without changing existing results. No stable solver policy or accepted Lean result is changed in this experiment.

## Interpretation

A more precise consequence representation need not produce a better greedy acquisition policy. W2 reweights distinctions as well as representing dependencies; any empirical improvement must be distinguished from the exact information it retains. At full arity the joint consequence set is the extensional version space itself, so this is a hierarchy of abstractions, not a new universal measure of intelligence. The practical target is an executable, inspectable answer to 'what relations did this observation establish?', including when no new individual output is known.

Prior context: candidate elimination and generalized binary search are established. Relational versus Cartesian abstraction is established in abstract interpretation. No novelty is claimed for these frameworks. This study measures their precise interaction in the retained construction fixture.
