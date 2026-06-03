# Changelog

## 0.2.3 - 2026-06-02

### Changed

- Fixed Apple Silicon MPS compatibility for BayesFlow/Keras Torch workflows by
  enabling PyTorch's CPU fallback for unsupported MPS operations and by failing
  unavailable `mps` or `cuda` requests during workflow construction.
- Optimized hierarchical random posterior sampling so
  `HierarchicalWorkflow.sample_random_posterior(...)` uses BayesFlow
  `ancestral_sample(...)` with subject data and group posterior draws passed as
  separate condition sets.
- Standardized all hierarchical random-workflow BayesFlow inputs:
  `inference_variables`, `summary_variables`, and `inference_conditions`.
- Updated random recovery diagnostics to process datasets in bounded chunks and
  forward `sample_batch_size` to BayesFlow posterior sampling.

### Validation

- Confirmed the release checks pass with `uv run pytest`,
  `uv run ruff check .`, `uv run black --check .`, `uv run mkdocs build`, and
  `uv build`.

## 0.2.1 - 2026-05-30

### Added

- Added `posterior_to_dataframe(...)` for converting posterior sample
  dictionaries into tidy analysis tables.
- Added documentation for posterior dataframe conversion utilities.

### Changed

- Kept workflow `sample_*` methods dictionary-based so raw posterior keys
  remain available for downstream workflow steps.
- Updated hierarchical workflows so `keep_subject_truth=None` saves all
  stochastic subject-level truth, while `keep_subject_truth=[]` saves none.
- Updated the development plan for the 0.2.1 sampling contract and the 0.3.0
  `MultiConditionWorkflow` direction.

## 0.2.0 - 2026-05-30

### Changed

- Refined the package metadata for the `0.2.0` release.
- Simplified the README into a package usage entry point for installed users,
  with a short overview, installation commands, a minimal workflow example, and
  links to the full documentation.
- Updated the documentation deployment workflow so GitHub Pages deploys
  automatically only from `main`.

### Documentation

- Clarified the public package positioning as reusable BayesFlow workflows for
  simulator-based Bayesian amortized model inference.
- Pointed users to the hosted documentation for detailed workflow examples,
  API reference pages, evaluation metrics, and diagnostic plots.
- Summarized the current milestone cleanup around public workflow contracts,
  simulator helpers, evaluation metrics, diagnostic plots, examples, runtime
  defaults, and backend/device documentation.

### Removed

- Removed long development, training, evaluation, and contributor guidance from
  the README so package users can quickly find installation and documentation
  links.
