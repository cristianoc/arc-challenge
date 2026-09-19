# Constant-to-feature replacement did not repair these programs

We implemented a bounded automatic repair search over the **120 external
Python programs**. It replaced one integer in a comparison or `range` bound
with one of twelve supplied input features. Training outputs filtered the
candidates. All query predictions were frozen and hashed before scoring them
against intended outputs.

**None of the 5,316 tested mutations fixed a previously wrong test answer.**
This remains true even if an oracle chooses the best training-fitting mutation.
The experiment therefore found a repair-language coverage failure before it
could test a useful method for selecting repairs.

This is a narrower step than discovering the relational repairs in 012. It
implements literal-to-feature binding; it does **not** discover path traversal,
per-colour counting scope, marker interpretation, or rank-based selection.
These are the same public tasks used in earlier audits, not an untouched
holdout. The feature list was fixed before this run; it was chosen by the
researcher, not learned.

## What was implemented

For each program, enumerate at most the first 16 eligible source locations:
integer literals 2–30 occurring inside comparison operands or `range` arguments.
At one location at a time, insert an input-derived scalar. The twelve features
are dimensions and their one-less variants, colour counts and populations,
4-connected monochromatic object count, and maximum object area. The
[protocol](protocol.md) gives exact definitions and CPU limits.

Keep every candidate that exactly fits all training outputs, including the
original program. A prediction is called *unanimous* only if every fitting
candidate is defined on every test input and predicts the same output tuple.
There is no ranking, majority vote, MDL score or task-specific priority.

The original is deliberately retained. If it fits training and predicts a test
answer, a different answer cannot be unanimous. That logical limitation was
recorded **before** running: the experiment's informative measurement is whether
correct repairs exist in the space, and how much ambiguity the space creates.
It does not empirically discover that consistency fails to distinguish two
training-fitting programs.

## Results

| Measurement | Result |
|---|---:|
| Programs examined | 120 |
| Programs with eligible mutation sites | 85 |
| Programs with no eligible site | 35 |
| Programs reaching the 16-site cap | 7 |
| Mutations tested | 5,316 |
| Mutations fitting every training pair | 1,421 |
| Fitting mutations whose feature equals the old literal on every training input | 97 |
| Original programs fitting training | 116 |
| Original programs solving all test pairs | 27 |
| Training-fitting, test-wrong originals with any correct fitting mutation | **0 / 89** |
| Training-failing originals with any correct fitting mutation | **0 / 4** |
| Already-correct originals with a correct fitting mutation | 14 |

A mutation retaining training fit is not automatically a useful generalisation:
97 have the original literal value on every training input. Others can leave
behaviour unchanged because an altered guard has the same truth value, a branch
is inactive, or another part of the program masks the edit. We did not measure
which of those explanations accounts for each of the remaining mutations.

| Candidate agreement on all query inputs | Tasks | Correct unanimous answers |
|---|---:|---:|
| All fitting candidates defined and unanimous | 88 | 23 |
| Defined candidates disagree, none undefined | 23 | — |
| At least one fitting candidate undefined | 5 | — |
| No training-fitting candidate | 4 | — |

Thus **65 unanimous answers are wrong**. This is agreement inside a restricted
candidate space, not evidence that the task's intended rule has been identified.
The 35 tasks with no mutation site can have only the original candidate; their
agreement is vacuous. The strict policy abstains on 32 tasks and produces no
new correct answer. Four originally correct tasks become ambiguous.

## Connection to the three previous case studies

| Case | Mutations / fitting mutations | Distinct defined query answers | Result |
|---|---:|---:|---|
| `7b5033c1`: histogram → path order | 0 / 0 | 1 | No eligible literal; traversal is absent from this repair space. |
| `8f215267`: local lookup → global per-colour counts | 24 / 7 | 1 | All fitting candidates retain the original wrong answer. |
| `97d7923e`: position → marker-indicated rank | 60 / 19 | 5 | Several competing answers, none correct. |

This makes the structural gap concrete. Generalising some constants can enlarge
the set of candidate behaviours without introducing the required computation.
The result concerns these twelve features, these sites, and single-site edits;
it is not a negative result about all parameterisation or all anti-unification.

The literal and feature variants can be represented by a shared program with
one hole. However, choosing a computation to fill that hole is an additional
problem. No anti-unification algorithm was run, and replacing a literal by a
feature is not claimed to create a semantically more general program. No MDL
or Kolmogorov-complexity comparison was made.

## Checks and timeouts

Five pre-run controls passed: a successful width-binding repair, explicit
ambiguity, an undefined candidate blocking unanimity, component counting, and
site enumeration. All source/data hashes matched 006's pinned manifest. Every
original training-fit and test-correctness result matches the earlier baseline
when errors are correctly treated as failures, rather than truthy strings.

Thirteen mutations reached the 0.25 CPU-second fitting limit. Only these were
rerun at 5 CPU seconds per phase; all thirteen still timed out. All belong to
`2b83f449`, where the original code has this structure:

```python
if row[c] == 7:
    start = c
    while c < w and row[c] == 7:
        c += 1
    # ... repainting based on c - start ...
    continue
```

Changing just one of the two colour comparisons can make the outer guard true
and the inner guard false. Then `c` never changes, and `continue` repeats the
same state. [Verification evidence](verification.json) records a concrete
training cell and the two unequal comparison values for every timeout. The
input row is not modified by this loop. These candidates cannot fit all
training pairs, independently of the CPU limit. This also illustrates a
limitation of untyped, single-site edits: equal literals may encode a shared
role whose uses must change together. Globally changing every equal integer
would in turn risk conflating unrelated roles.

Main run: 12 workers, 8.24s. Timeout sensitivity: 12 workers, 10.96s. No speed
comparison is claimed; the reference original had a larger CPU allowance.

## Evidence and disposition

Close this bounded repair grammar without integration. The relevant next work
is the separate [minimal-sufficient-distinction study](../../013-minimal-sufficient-distinction/README.md),
which asks whether training supplies positive support for the distinctions a
program may forget. This negative result does not establish that study's
criterion; it only rules out this particular constant-lifting space as a useful
repair mechanism for the audited failures.

Two studies were concurrently registered with numeric prefix 013. Their full
slugs identify separate records; neither study's files or results replace the
other's.

- [Frozen protocol](protocol.md).
- [Per-task scores and totals](scores.json).
- [Complete compact candidate evidence](candidates.json).
- [Run provenance and prediction hash](run.json).
- [Targeted timeout results](sensitivity.json).
- [Baseline, lossless encoding and loop checks](verification.json).

`candidates.json` retains all candidates, input features, source/data hashes,
and predictions. To avoid duplicate grids it uses an `answers` array; each
candidate row begins with an index into `schemas`, followed by that schema's
field values. `site` indexes the task's sites; `feature` indexes the top-level
feature list; `predictions` and the task's `answer` index `answers`. Nulls remain
null. Expansion was checked for exact equality with the frozen prediction
file, whose SHA-256 is in the run manifest.

Runnable main experiment was committed before execution at `cf2e54f`.
The sensitivity procedure is available at `91603a2`. Completed live runner code
was removed; the stable solver and accepted mathematics were unchanged. From
a checkout of `91603a2`, with the pinned 006 corpus:

```sh
python3 symarc/experiments/013-constant-lifting/run.py controls
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 symarc/experiments/013-constant-lifting/run.py predict --corpus /path/to/corpus.json --out /tmp/arc013
python3 symarc/experiments/013-constant-lifting/run.py score --corpus /path/to/corpus.json --out /tmp/arc013
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 symarc/experiments/013-constant-lifting/sensitivity.py --corpus /path/to/corpus.json --out /tmp/arc013
```
