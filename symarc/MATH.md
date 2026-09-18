# Functional closure under symmetry

This is the current mathematical account for the experiments. The checked
model is [SymArc/Theory.lean](SymArc/Theory.lean); the executable model is
Rust in [src/](src/). Lean checks the structural theorems below, not the
Rust implementation, numerical entropy calculations, or sampling procedure.

## Examples and partial actions

Let $A$ be inputs, $B$ outputs, and $D \subseteq A\times B$ a finite,
nonempty training specification. For ARC, both carriers are rectangular grids.
A program is a partial function $p:A\rightharpoonup B$.

A generator name $g\in\Gamma$ has partial actions on $A$ and $B$. It acts on
an example by

$$g\cdot(x,y)=(gx,gy),$$

when both component actions are defined. The same name can act on input and
output grids of different shapes. No group laws are required for the general
construction. [`PAction`, `pairAct`, `pairAct_some`]

For $S\subseteq\Gamma$, define $C_S(D)$ as the least set containing $D$
and closed under every generator in $S$. Equivalently, it contains every
example reachable by a finite word of applicable generators. [`Reach`]

Closure is monotone in both $D$ and $S$, and closing twice adds nothing.
If each generator has an inverse in $S$, singleton reachability is symmetric;
together with reflexivity and transitivity, this gives orbit equivalence.
The checked symmetry theorem assumes a single inverse generator; inverse
words require an additional argument. [`Reach.mono_data`, `Reach.mono_gens`,
`Reach.trans`, `Reach.symm`]

## Functionality

A relation $R\subseteq A\times B$ is functional when

$$(x,y),(x,y')\in R\implies y=y'.$$

Call $S$ admissible for $D$ when $C_S(D)$ is functional. For a finite relation
with a distribution giving positive probability to every pair, functionality
is equivalent to $H(Y\mid X)=0$. [`Functional`]

Admissibility is downward closed in $S$. Thus, with exact closure checks and
functional starting data, greedy addition preserves functionality. Trying
every candidate once produces an inclusion-maximal admissible set, not
necessarily one of maximum size or maximum entropy gain. Different orders
can produce different sets. [`functional_antitone`]

For a group acting on a single example, functionality has the characterization
$\operatorname{Stab}(x)\subseteq\operatorname{Stab}(y)$. With several examples,
there can also be conflicts between their orbits. This characterization is
mathematical context, not a separately checked theorem in the Lean file.

## Entropy and coverage gain

For a finite nonempty set $R$, the entropy of the uniform distribution on its
elements is $\mathcal H(R)=\log_2|R|$. Define

$$G(S,D)=\log_2|C_S(D)|-\log_2|D|.$$

For functional closures, distinct pairs correspond to distinct inputs. Gain
then measures expansion of the input coverage of the assumed specification.
It is monotone in $S$. It does **not** measure the probability that the
assumption is correct, information supplied by independent observations, or
uncertainty reduction over programs.

Uniformity on pairs does not generally imply uniformity on distinct outputs.
In particular, $\log_2|\pi_B R|$ is output support size in bits; it is not
in general the marginal Shannon entropy $H(Y)$ induced by uniform pairs.
The solver reports `outs` as a count, not an entropy.

The denominator uses distinct training pairs mathematically. The current
solver uses the training-list length; these agree for distinct examples.
Duplicate examples should be removed before interpreting gain as this formula.

A capped search reports discovered coverage, hence a lower bound on gain when
the relation really is functional. A cap flag means closure completion was
not established. Equal capped gains need not mean equal true gains. When a
conflict exists outside the explored part, the functionality premise itself
has not been established.

### What the entropy objective leaves open

The intended objective is to choose symmetry assumptions with large gain,
subject to functionality and realizability by a program. The generator
vocabulary and program class supply essential priors: allowing arbitrary
bijections of grids admits enormous but uninformative orbit constructions.
Even within a fixed vocabulary, equal closures on $D$ can imply different
behavior outside that closure.

Three quantities should remain distinct in experiments:

* **Coverage gain:** the expansion $G(S,D)$ above.
* **Program uncertainty:** uncertainty over programs consistent with the
  specification, requiring an explicit prior (uniform over syntax is one
  possible choice and counts equivalent programs separately).
* **Prediction uncertainty:** uncertainty over outputs at a particular input,
  induced by that program distribution, with undefinedness handled explicitly.

Only coverage gain and counts/agreement are implemented now. The solver does
not rank candidate sets by gain; it accepts admissible candidates in a fixed
order. Whether gain improves assumption selection is an experimental question.

## Fitting and equivariance

A program fits $R$ when $p(x)=y$ for every $(x,y)\in R$. [`Fits`]

Equivariance for $g$ at $x$ means

$$gx=x',\quad p(x)=y,\quad gy=y'\quad\implies\quad p(x')=y'.$$

This implication is vacuous when its premises are undefined. Definedness is
therefore a separate condition. [`EquivariantAt`]

A program fits $C_S(D)$ **if and only if** it fits $D$ and is equivariant for
every $g\in S$ at every input in the closure. This is equivariance along the
closure, not everywhere. [`fits_closure_of_equivariant`,
`equivariant_of_fits_closure`]

Realizability means some program in the chosen class fits the closure. A
sample-based executable check is evidence for this property, not a proof.

## Determination at a test input

For a program class $P$ and specification $R$, an input $x^*$ is determined
when all programs in $P$ that fit $R$ and are defined at $x^*$ agree there.
This does not assert that any such program exists or is defined. [`Determined`]

If $(x^*,y)\in R$, determination follows for every program class: every fitting
program must return $y$. [`determined_of_reach`]

Outside the closure, agreement depends on the chosen program class. It is
neither guaranteed by coverage gain nor a correctness guarantee. Different
shapes or object counts may require generalization carried by the program
prior rather than by the symmetry action.

## Concrete model and approximation boundaries

The candidate families are flips/transpose, nonzero colour swaps, cyclic row
and column shifts, and adjacent row and column swaps. Bounds and palettes
use training grids and test inputs; test outputs are used only for scoring.
A fresh nonzero colour, when available among 1–9, represents an unseen colour.

The DSL composes bounded grid transformations: geometry, tiling, cropping,
colour changes, painting, boolean combinations of halves, and counting.
Intermediate grids must have sides between 1 and 30. The exact pool and its
order live in `src/dsl.rs`.

The executable procedure is:

1. Greedily select generators passing a capped functionality check.
2. Enumerate fitting programs up to the requested depth. If none are found,
   use hill climbing with random edits, restarts, and fitness-preserving
   shortening. Improving edits may lengthen; equal-fitness edits may not.
3. For each functional generator, retain programs fitting an enlarged closure
   sample and commuting with that generator at test inputs. If none remain,
   try a hill-climbing repair from the first surviving program. A successful
   repair replaces the survivor set. If no program fits the original data,
   report the functionality-only generator set.
4. Select the shortest survivor, and report coverage, fitness, test-input
   equivariance, survivor agreement, and scored predictions.

Samples include the data and defined one-step images, then random walks for
capped closures or random pairs for complete closures. Small complete closures
are used in full. These samples are not generally uniform draws from the true
closure. Later filtering or repair can invalidate earlier sampled constraints;
`fitC` and final equivariance checks are diagnostics, not universal guarantees.

The closure routine stores one output per input and flags conflicts. Caps are
checked between BFS expansions, so a final expansion can exceed the nominal
cap. Changing traversal order can change the observed conflicts under a cap.
The symmetry-only lookup is used as a fallback when the selected program is
undefined. Runtime determination is reported only when at least two surviving
programs are defined at the input; otherwise it is `-`.

The Lean file additionally proves flip, transpose, rotation-inverse, and
colour-swap identities on a functional grid model (`FGrid`). No theorem
connects that model to the Rust byte arrays. Entropy identities and the
statistical validity of any experimental estimator are not machine-checked.
