# Retained results

[baseline/](baseline/README.md) contains the current stable solver's reference
results: directly emitted Markdown reports and reproducibility manifests tied
to a source commit. This is the comparison point for new experiments.

- `results/baseline/`: retained baseline evidence, committed with the project.
- `experiments/RESULTS.md`: compact findings and decisions from experiments.
- `out/`: disposable raw runs, ignored by Git.

Refresh the baseline deliberately when an accepted core change warrants it.
Keep one current baseline here; Git preserves prior versions. An experiment
must cite the baseline source commit/hash it actually used, not just this path.
