# What the three surviving wrong programs miss

We implemented an audit of three existing original/repaired program pairs,
measured relationships in their training examples, and constructed one
unlabelled disagreement witness per case. **We did not discover the repairs.**
They were written in 006 after inspecting the intended test outputs.

Both programs fit all **8 training pairs**; the original programs solve **0/3
test pairs**, the repairs **3/3**. This reproduces a known result. The new result
is an explicit account of what training does and does not distinguish, with
executable witnesses to the mechanisms.

| Task | Relationship in training | Where agreement ends | Additional premise for the repair |
|---|---|---|---|
| [7b5033c1](https://arcprize.org/tasks/7b5033c1) | In both examples each colour occupies one path run; path run order equals first appearance in row scanning. Both predict the entire output. | Test path revisits colours: runs `1,3,8,4,1,3`; histogram groups repeated occurrences. | The output serialises the connected path, including revisits, rather than grouping colours. Training does not distinguish these hypotheses. |
| [8f215267](https://arcprize.org/tasks/8f215267) | Across 3 grids, all 9 frames receive exactly as many marks as there are separate objects of that frame's colour to the right. Counts range 0–4. Local lookup agrees on all 9. | Test local counts are `2,1,2,2`; global counts and intended marks are `2,4,1,1`. | Frame colour identifies which objects to count anywhere in the instruction region; the same counting rule persists on new arrangements. |
| [97d7923e](https://arcprize.org/tasks/97d7923e) | Across 3 grids, all 4 colour groups select the bar whose descending length rank equals the top marker's length. Observed marker lengths are `2,1,1,2`. Positional guards make the same choices. | Test marker lengths are `2,1,3,1` by ascending colour ID. The original selects ranks `3,none,1,none`, instead of `2,1,3,1`. | Marker length denotes an ordinal, positions are incidental, and the rule extends to previously unseen rank 3. |

The 9 frames and 4 groups are within-grid observations, **not independent
training tasks or statistical confidence estimates**. Segmentation, candidate
relationships, and hypotheses were supplied by the researcher. No search over
alternative relationships or multiple-comparison correction was performed.

Witness boards (predictions are explicitly labelled): [path](figures/7b5033c1.svg), [counts](figures/8f215267.svg), [rank](figures/97d7923e.svg).

## Agreement conditions and disagreement witnesses

**Path.** On simple paths with the same chosen background, histogram and path
programs agree exactly when each colour has one contiguous run and the run
order equals the histogram's first-seen colour order. Counts then determine
each run length. Repeated noncontiguous colours necessarily break agreement.
The witness uses straight sequences `1,1,2` and `1,2,1`, surrounded by background.
The histogram and colour order are identical, but the repaired outputs differ.
Neither synthetic output is declared correct. This is genuine information loss
at the histogram/order representation, not a bug in its downstream renderer.
The path repair still assumes the lexicographically first endpoint is the start;
this audit does not independently justify that orientation choice.

**Counting.** For the observed frame layouts, the two programs share a renderer.
They agree when the per-frame local-lookup and global-component counts agree
after clipping to available slots. All 9 training counts agree even before
clipping. The witness adds one isolated yellow cell at zero-based `(0,12)` in
training input 0, outside the yellow frame's row interval. Its local patch stays
identical and the *entire original output* stays identical. The repair adds one
mark inside that frame. This proves that a decision using only that patch
cannot implement the global-count hypothesis on both inputs. Whether a new
one-cell component should count as an object is an additional premise; the
witness does not establish it as task ground truth.

**Rank.** With distinct bar lengths and valid markers, agreement requires that
the original guards select exactly the marker-indicated ranked bar in each
colour group. All 4 training groups satisfy this condition. Swapping columns 2
and 13 of training input 0 preserves all column contents and marker/rank data.
The repaired output follows the swap exactly; the original gains an incorrect
fill (two cells relative to the repaired prediction). This is a *decision-rule
failure*, not loss of raw input information: the original parser retains bar
lengths and the marker. We therefore do not force it into the same
representation-loss explanation as the first two cases. The new column-swap
law is hand-supplied and task-specific; this is not another learned invariant.
Ties and invalid/out-of-range markers remain unspecified by these examples.

## What is principled here?

An elementary factorisation obstruction captures the first two mechanisms.
If a program decides using only a statistic `s(x)`, it has the form `h(s(x))`.
If two inputs share `s(x)` but the proposed repair requires different decisions,
no choice of `h` can implement that repair on both. The representation or its
scope must change. For counting, the decision is for one fixed frame, not an
assertion that the entire original solver reads only that frame's patch.
This is a conditional mathematical fact about the *proposed repair*, not a
proof that it is the intended task function. The rank case instead changes how
already-available features are used.

These repairs do not follow merely by replacing constants with variables.
First-order syntactic anti-unification produces a common term pattern under
substitution; it does not itself choose path traversal, global scope, or rank
as the input-dependent computation for a hole. More expressive abstraction
systems could use such operators, but none is implemented here. See
[Cerna and Kutsia's survey](https://arxiv.org/abs/2302.00277).

The global-count and rank rules offer compact explanations for several training
relationships. That makes them candidates for an MDL comparison, **not measured
MDL winners**. A comparison would need a declared coding language and the cost
of segmentation, counting/ranking operators, constants, and any exceptions.
No code lengths or Kolmogorov complexities were computed. See
[Grünwald's MDL tutorial](https://arxiv.org/abs/math/0406077).

## What we can conclude

The path case is observationally confounded in its training examples. The other
two repairs describe repeated, measurable training relationships that the wrong
programs also happen to fit. Choosing those explanations still requires a
preference for relational rules over local lookup/positional guards. Neither
training fit nor these witnesses uniquely establishes that preference.

The next discriminating experiment would freeze a small feature/rule language,
its costs, and a selection procedure **before examining new tasks**. Its target
would be whether training alone selects useful relational repairs, not whether
we can explain known answers after seeing them. This study supplies three
mechanism examples for designing that experiment; they cannot be its holdout.

## Reproduction and checks

Protocol and original runner were committed before the run at `ca83781`.
[Protocol](../../012-information-loss/README.md),
[runner](../../012-information-loss/run.py), and
[independent witness checks](../../012-information-loss/verify.py) are retained
as a small reusable catalogue for future representation/rule-selection work.
The stable solver and accepted mathematics were not changed.

From the repository root, supply the revision-pinned 006 corpus:

```sh
python3 symarc/experiments/012-information-loss/run.py --corpus /path/to/corpus.json --out /tmp/arc012
python3 symarc/experiments/012-information-loss/verify.py
```

The runner checks all three source and labelled-data hashes against 006's
manifest. It reuses the existing repairs rather than copying them. Twelve-worker
pool, three case jobs, deterministic searches, no random seed, elapsed 0.124s;
this is not a benchmark. [Results](results.json) retain every relationship,
witness input, and both predictions. [Manifest](run.json) records hashes and
runtime. Independent checks verify histogram equality, unchanged local patch
and isolated added component, and the column-swap relation. All three pass.
