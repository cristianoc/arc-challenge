# Stable core and experiments

- `src/` is the stable solver library and CLI. Preserve its default behavior
  during experiment work; put trial policies in `experiments/<id>/`.
- Experiments import the core or invoke its CLI. Never copy the solver. Keep
  experimental Cargo packages independent of the core build.
- Follow `experiments/README.md`: register the question before running it,
  record decisive results and disposition in `experiments/RESULTS.md`, and
  retain that compact entry when pruning code or raw outputs.
- Integrate promising work as a separate evidence-backed change. Move accepted
  code/tests into core and remove the experimental copy. Do not accumulate
  abandoned feature flags or duplicate implementations.
- `MATH.md` and `SymArc/Theory.lean` describe the current accepted model only.
  Keep experimental mathematics local until accepted; retain only a compact
  finding about rejected ideas in the ledger.
- Check core changes with `cargo test --release`, `cargo build --release`, and
  `python3 tests/check.py`. Check mathematical changes with `lake build`.
  Run timing comparisons serially. No extra permission step is imposed here.
- Older projects outside `symarc/` are outside this cleanup/research scope.
