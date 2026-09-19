# 009: examples can refute a symmetry law without establishing another

**Implemented:** an exact Python checker for whether any deterministic
colour/geometry-equivariant function could fit a task's supplied examples.
It applies the **same three laws to every task**, with no task-specific choices:
all permutations of ten colours (including 0), all eight rotations/reflections,
and their combination. Inputs and outputs undergo the same action. The checker
uses the grids alone, not original solvers or hand-written repairs.

**Result:** the training pairs contradict full colour equivariance on 15/120
tasks and full geometric equivariance on 1/120. Their combination is contradicted
on 16/120. These are actual contradictions in supplied examples, not failures
against artificially labelled examples. The other tasks are compatible with the
laws, which means an equivariant explanation exists—not that the intended
explanation is equivariant.

| Supplied law | Contradicted by training | Compatible with training | New contradictions after adding known tests |
|---|---:|---:|---:|
| All colour permutations | 15 | 105 | 0 |
| All rotations/reflections | 1 | 119 | 0 |
| Both together | 16 | 104 | 0 |

Crucially, **no two distinct supplied inputs in any task are related by even the
combined transformations**, in training alone or training plus tests. The
compatibility checks therefore constrain how each individual example respects
its own symmetries, but do not link different examples. The zero additional
contradictions from test pairs does not mean these laws predicted any test answer.

[Direct report](report.md) · [per-task results and contradiction witnesses](results.json) ·
[protocol](protocol.md) · [run manifest](run.json).

## A concrete contradiction requiring no knowledge of the test answer

Training example 0 of `221dfab4` has input colours {1,4,8}. Its output contains
{1,3,4,8}. Let the colour permutation exchange 0 and 3 and leave all others fixed.
The input is unchanged; the required output changes. Any deterministic program
fitting this example therefore violates that proposed symmetry law:

```math
T(x)=x,\qquad T(y)\ne y,\qquad
P(T(x))=P(x)=y\ne T(y).
```

This explains why fixing designated colours matters. Experiment 008 manually
exempted colours 3 and 4 on this task. Here the data itself proves that this
particular swap must be excluded from any equivariance group of a fitting rule.
It does not uniquely determine the right smaller group, or establish that all
remaining swaps are valid. In general, colours absent from an input but present
in its output can expose contradictions of this kind.

The geometric contradiction is training example 1 of `da515329`: reflection
across the main diagonal preserves the input but changes its required output.
The stored witness has geometry index 7 (horizontal reflection followed by three
clockwise quarter-turns). All 16 reported training contradictions are witnessed
by a transformation preserving a single input, without needing a second example.

## Exactly what is decided

The three transformation groups are manually supplied mathematical candidates;
the checker does not discover transformations, select a law, or synthesize a
solver. Unlike 008, applicability and fixed-colour exemptions are not chosen
separately for each task. Full colour permutation is deliberately a strong law;
its rejection does not reject more restricted colour symmetries.

Let $D$ contain labelled pairs $(x_i,y_i)$ and let $G$ act on both input and output
grids. A deterministic equivariant function fitting $D$ exists precisely when

```math
\forall i,j\;\forall g\in G,\qquad
 g\cdot x_i=x_j\ \Longrightarrow\ g\cdot y_i=y_j.
```

The case $i=j$ checks transformations that fix an input (its stabilizer). The
case $i\ne j$ checks agreement between different examples related by a
transformation. Both are necessary; merely searching for related distinct
examples would miss every contradiction found here.

The criterion is sufficient as well as necessary for the unrestricted class of
grid-to-grid functions used here. On each orbit containing an example, define
$P(g\cdot x_i)=g\cdot y_i$. The criterion makes this definition independent of
the chosen representative and transformation. On all remaining orbits choose
$P(x)=x$. This constructs an equivariant extension. It can simply memorise a
separate answer for every observed orbit; it need not discover a shared task
rule. This is an elementary existence argument, not a claim about short programs,
SymArc's language, MDL, or a new theorem.

Because none of the supplied test inputs lies in a training input's orbit here,
these laws cannot obtain its output by transforming a training answer. Self
symmetries may still constrain possible outputs. Further relations or a prior
would be needed to choose among compatible extensions. Compatibility alone does
not justify the symmetry assumption or establish its intended generalisation.

## Implementation and checks

For each pair of inputs and each spatial action, the checker finds the partial
colour bijection forced by corresponding cells. If no bijection exists, there
is no transport. Otherwise it checks the corresponding outputs for every
completion of that bijection. This is symbolic: if an output uses an unconstrained
colour and at least two target colours remain free, swapping their assignments
provides a concrete violating completion. Otherwise one completion suffices.
Thus the colour/product check covers the full permutation group, not just
single swaps. Every rejection records an actual permutation and spatial action.

Four targeted controls pass, including an asymmetric output for a symmetric
input and an output colour absent from the input. On a three-colour synthetic
domain, 300 comparisons against explicit enumeration of all permutations and
spatial actions agree. Each recorded witness can be directly checked by applying
its permutation and geometry: the transformed input matches a supplied
input, but the transformed output disagrees. All 64 stored witnesses across
laws and splits were independently rechecked. A separate canonical-form
calculation also verified the absence of cross-example orbit matches in all
120 tasks.

Protocol and code were committed before corpus execution at
`1e69af401e6174fc0fde87588b404553475b705e`. The input hashes match 006's 120 pinned
tasks. All training results are computed before adding test pairs for the
separate diagnostic. These tasks and their test answers were already available
in earlier research: there is no untouched-holdout or blind-discovery claim.
Twelve workers completed this functional run in 0.264s; no performance comparison
is claimed. Stable core, accepted mathematics and previous results are unchanged.

## Disposition and next question

Closed diagnostic. Runnable code remains at `1e69af4`; the completed experiment
directory is removed. To reproduce on that revision, from repo root:

```sh
python3 symarc/experiments/009-symmetry-compatibility/run.py --test
timeout 600 python3 symarc/experiments/009-symmetry-compatibility/run.py \
  /absolute/path/to/006-corpus.json symarc/out/experiments/009-symmetry-compatibility
```

Use 006's pinned fetcher to obtain the corpus. The retained results have only
whitespace compacted; [serialization hashes](evidence.json) identify the original
and retained bytes. The original manifest describes the unmodified runner output.

The next substantive question is what evidence connects different input orbits.
Adding a preferred symmetry merely because it is compatible would reintroduce
an unsupported assumption. A useful further study must state what cross-example
relation or prior it supplies and test what that extra premise explains.
