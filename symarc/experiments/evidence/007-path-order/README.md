# 007: path order is expressible, but training fit does not select it

**Implemented experiment:** I manually designed a small language for traversing
coloured paths, based on a repair whose test answer had already been inspected.
A Rust enumerator searched that supplied language using training pairs, alongside
a histogram language and existing grid/object solvers. The search selected
programs automatically; the path concept, grammar and selection order were
supplied by me, not discovered by the search.

**Results:** on the known example, both histogram and path programs fit training;
path-first selection predicts the known answer, while histogram-first selection
fails. On 400 previously used public training tasks, the path language fits
none, and combined accuracy stays 60/400. Its one-dimensional output restriction
already excludes 388 of those tasks. This demonstrates a bounded negative result
and ambiguity on one inspected example, not general discovery of path abstraction
or a justified preference for path over histogram.

The frozen extension failed its public-training coverage criterion and is
closed. Stable core, 003 objects, and accepted mathematics are unchanged.
[Protocol](protocol.md) was published at `8ed404a` before scientific runs;
runnable code and the portable resource wrapper are preserved at
`cd7e58b2513a408a698c016907e68131f5ef2951`. The original wrapper could not start
because `/usr/bin/time` was absent; Python child-resource measurement replaced
it before any scientific execution. The language and selection rule did not
change. All five targeted tests passed in debug and release builds.

## Observations

| Corpus / policy | Training-fitting tasks | Test-correct tasks | Oracle-correct tasks |
|---|---:|---:|---:|
| 400 public training tasks: grid d2 / objects / histogram control | 63 | 60 | 61 |
| Same: append path fallback | 63 | 60 | 61 |
| Same: path alone | 0 | 0 | 0 |
| Known `7b5033c1`: histogram alone | 1 | 0 | 0 |
| Known `7b5033c1`: path alone | 1 | 1 | 1 |
| Known `7b5033c1`: control then path | 1 | 0 | 1 |
| Known `7b5033c1`: path then control | 1 | 1 | 1 |

Oracle means one training-fitting program gets every test output correct; it is
not a deployable selector. The known example's test answer was already inspected
in 006. Its success is a mechanism check, not an independent generalisation gain.
The 400 public tasks were used in previous studies, not an untouched holdout.

Direct reports: [pilot](pilot/report.md), [full](full/report.md),
[development](development/report.md). Twelve workers, serial timing runs:
pilot 0.467s / 9,216 KiB peak RSS; full 3.079s / 9,088 KiB; development 0.114s /
9,216 KiB. Full enumeration tested 575,520 histogram/path ASTs with 576,192
training-example checks. Core/003 reference counts reproduced exactly. No task
qualified for the registered depth-3 audit; no depth-3 comparison was performed.
The sole histogram fit, `d631b094`, was already solved by core `countRow`.

A post-run [output-shape diagnostic](output-shape-diagnostic.json) finds only
12/400 tasks have exclusively one-dimensional training outputs. The frozen
renderer excludes the other 388 before graph structure even matters. Thus this
negative result rejects this small end-to-end grammar on this corpus, not path
reasoning generally. Choosing the mechanism from an ARC-AGI-2 witness did not
establish its applicability to the older public-training corpus.

## The unresolved selection problem

For `7b5033c1`, both languages have 20 ASTs fitting both training examples.
Each has two distinct query predictions: 18 aliases give its main prediction,
and two `UniqueColour` aliases give the same incorrect 12-cell subsequence.
The two 25-cell main predictions differ:

- Histogram: `1×8, 3×5, 8×6, 4×6`.
- Path: `1×5, 3×3, 8×6, 4×6, 1×3, 3×2` (the supplied answer).

The selected path witness is
`modal-4-mono / All / Path4 / first / column`.
The histogram family cannot emit separated runs of one colour at all. Varying
its colour constants, counts, or the order of its single-colour runs cannot
express the required answer. Preserving adjacency and traversal order changes
the representation. It is not merely replacing constants with parameters.

But expressibility is insufficient: first-fit control still chooses a fitting
histogram. Prioritising path succeeds on this inspected example, without proving
that this priority is a sound general rule. Both families have the same 18:2
query-answer frequency split; under a uniform prior over their fitting ASTs,
their answer entropies are identical. Neither training consistency, number of
fits, nor that entropy diagnostic distinguishes the correct family here.

The controlled synthetic test shows what additional evidence could do: a
training path containing a repeated, noncontiguous colour eliminates every
histogram fit while preserving path fits. This uses a supplied labelled example;
it is not an available oracle in ordinary ARC.

## What this says about the motivating theory

No anti-unification algorithm or MDL selector was evaluated. Syntactic
anti-unification can enlarge an instance family under substitution without
showing that the chosen instance predicts the intended answer. Here the useful
repair introduces traversal semantics that the histogram renderer cannot
express. Anti-unifying whole subprograms into a hole would still leave the
semantics of that hole to be learned.

Similarly, exact-fit MDL would need a specified code length for histogram and
path programs, including the representation. Training error ties. The present
family order is not an MDL code, and alias counts are not description lengths.
No claim follows that MDL necessarily chooses path, or that this experiment
refutes MDL. The concrete open problem is how an evidence-based prior or further
constraints can distinguish these fitting explanations without using the test
answer.

## Reproduction and next question

Check out `cd7e58b2513a408a698c016907e68131f5ef2951` and use the commands in the
retained protocol. For the development run, fetch 006's pinned inputs and write
its `7b5033c1` data object to a task JSON, then pass its absolute path to
`python3 experiments/007-path-order/run.py development TASK.json` from `symarc`.
The data's canonical SHA-256 was checked against 006's manifest:
`c2883ff0c29933800c27e34d8e5f6e3a399d024ebc255737535c5128baedcb1b`.

Each retained run manifest records source/data/binary hashes, command, revision,
clean working tree, resource usage and artifact hashes. Reports, per-task scores,
selections and the known-witness pools are retained here. Other pools are
regenerable at that revision; their hashes remain in the manifests. Local paths
in original manifests describe the execution environment, not portable inputs.

Before another operator sweep, establish multiple explicit witnesses in the
chosen study corpus and check renderer applicability. For the original repair
research, a useful next question is to catalogue which incorrect/correct program
pairs remain observationally equivalent on training data, and which extra
constraint or counterexample would separate each pair. Keep representation
coverage and selection evidence as separate claims.
