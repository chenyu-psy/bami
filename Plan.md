# Development Plan

## Purpose

This file tracks reviewable development work for `bami`. Each item should be
small enough to inspect, test, and explain to psychology researchers who use the
package in their own modeling projects.

The current priority is to review the existing code and polish the website.
That means making the current public workflow clearer, better documented, and
easier to teach. New modeling features and research prototypes belong to the
next milestone, after the current code and documentation are stable.

## Current Milestone: Review Current Code

This milestone focuses on the package that already exists. Do not add new
public features unless a small change is required to fix current behavior,
documentation, or tests.

### 1. Stabilize Existing Workflow Contracts

- Review `SimpleWorkflow` and `HierarchicalWorkflow` for consistent argument
  names, return values, and error messages.
- Check fixed and flexible trial-count behavior.
- Check aggregate and trial-level observation contracts.
- Make validation messages explain what researchers should change in their
  model setup.
- Add or refine tests for workflow behavior that is already public.

### 2. Review Existing Simulators and Input Helpers

- Review SDM, ezDM, M3, and circular simulator helpers.
- Review input formats, padding, and validation helpers.
- Improve docstrings, comments, data-shape explanations, and examples where
  they make the existing data flow easier to teach.
- Add or refine tests for current behavior such as parameter validation,
  probability normalization, output shape, optional jitter, optional scaling,
  padding, and trial-count features.

### 3. Review Existing Evaluation Tools

Completed for metrics, legacy recovery, and model diagnostic plots.

- Evaluation metrics/API cleanup is complete. Public evaluation metrics now
  keep only dataframe-first helpers and scalar metrics:
  `aggregate_data`, `estimate_recovery`, `compute_corr`, `compute_ccc`, and
  `compute_rmse`.
- Posterior sampling is exposed through workflow/model methods, not through
  evaluation metrics.
- Old BayesFlow/brms diagnostic wrappers, placeholder metrics, and public
  recovery wrappers have been removed from evaluation.
- Old evaluation plotting functions and table-contract validators have been
  removed.
- New diagnostic plots live in `bami.evaluation.diagnostics`, with shallow
  model methods for ordinary use:
  `model.plot_parameter_recovery(...)`,
  `model.plot_population_recovery(...)`, and
  `model.plot_random_recovery(...)`.
- The old `bami.recovery` module has been removed. Future recovery workflows,
  if needed, should be redesigned from model-level sampling and dataframe-first
  evaluation helpers.

### 4. Polish Website Structure

- Review `mkdocs.yml`, `docs/index.md`, API pages, and examples.
- Make standard workflows easier to find before advanced internals.
- Keep `README.md` focused on the package overview and first successful use.
- Keep `docs/` pages organized by task: setup, simulation, training,
  evaluation, diagnostics, and examples.
- Hide or de-emphasize advanced inference navigation if it distracts from
  normal user workflows. Keep the source docs in place so internal references
  do not break.

### 5. Improve Existing Examples

Completed for the current simple SDM and hierarchical ezDM examples.

- The examples now show complete workflows from model construction through
  simulation, training/loading, posterior sampling, dataframe recovery tables,
  `aggregate_data(...)`, `estimate_recovery(...)`, and diagnostic plots.
- Expected array shapes are shown after major simulation and sampling steps.
- Examples use current model-level APIs rather than old recovery or evaluation
  wrappers.
- Future example reviews should focus on runtime defaults, copy-paste testing
  on fresh environments, and whether additional model families need the same
  complete workflow treatment.

### 6. Run Checks and Record Remaining Gaps

- Current checks pass for the latest code and documentation review:
  `uv run pytest`, `uv run ruff check .`, `uv run black --check .`, and
  `uv run mkdocs build`.
- `uv run mkdocs build` prints the upstream Material for MkDocs 2.0 warning,
  but the site still builds successfully.
- Record future failures as concrete follow-up tasks with the exact command
  that failed.
- Keep unresolved issues separate from next-milestone feature ideas.

### 7. Review Runtime Compatibility and Defaults

- Runtime defaults have been reviewed and updated. `train_workflow(...)` now
  defaults to `initial_epochs=5`, `n_batch=2000`, `workers=1`, and
  `max_queue_size=4`, while preserving `max_epochs=100`, `batch_size=32`,
  `validation_data=200`, `patience=5`, and `min_delta=0.1`.
- `SimpleWorkflow.train_workflow(...)`, `HierarchicalWorkflow.train_workflow(...)`,
  and inherited random-workflow training settings now share the same defaults.
- Workflow documentation now explains which training arguments mainly affect
  total computation, memory use, validation stability, and concurrent
  simulation/prefetching.
- Backend/device support is now documented explicitly: current `bami` workflows
  use the BayesFlow/Keras Torch backend, and `torch_device` controls only Torch
  device selection. TensorFlow and JAX backends are not part of the current
  tested workflow contract.
- Static cross-platform portability audit has been recorded instead of
  requiring real macOS/Linux/Windows hardware runs. The audit checked for
  hard-coded absolute user paths, OS-specific shell commands, platform-specific
  Python branches, multiprocessing/fork assumptions, and Torch device
  assumptions.
- The current audit found no macOS-only runtime dependency. `mps` is checked
  only when users explicitly request `torch_device="mps"`, and unavailable
  accelerators fall back to CPU. The Torch backend is an intentional current
  support boundary rather than an accidental platform dependency.
- `configure_torch_device(...)` now validates device names before calling
  Torch, so unsupported values such as `"gpu"` produce a bami-level error that
  explains the supported Torch device choices.

## Current Milestone Acceptance Criteria

- Existing public APIs are simple, explicit, and teachable.
- Existing docs guide users through standard workflows before advanced
  internals.
- Examples match current code and show expected array or dataframe shapes.
- Runtime defaults are reasonable for CPU-only and lower-performance research
  laptops.
- Backend and device selection behavior is documented clearly enough that users
  do not need to inspect internals.
- Known platform compatibility gaps are documented with exact commands or
  manual checks still needed.
- Touched functions have useful docstrings.
- Comments explain assumptions or domain reasoning instead of restating the
  code.
- Checks pass, or remaining failures are recorded with exact commands and next
  actions.
- No new public features are added unless required to fix current behavior.

## Current Milestone Remaining Gaps

- Record future platform-specific findings here if users or CI expose behavior
  that was not visible in the static portability audit.

## Milestone 0.2.1: Workflow-Wide API Alignment

Completed for version `0.2.1`.

This milestone completes workflow-wide changes that were identified while
planning `MultiConditionWorkflow`. These changes affect existing workflows and
should be implemented before adding the new multi-condition workflow.

### 1. Posterior Sampling and Dataframe Conversion

- Keep public `sample_*` methods dictionary-based so downstream workflow logic
  can reuse raw BayesFlow keys when needed.
- Preserve this contract for existing public workflow sampling methods before
  adding `MultiConditionWorkflow`: `SimpleWorkflow.sample_posterior(...)`,
  `HierarchicalWorkflow.sample_group_posterior(...)`, and
  `HierarchicalWorkflow.sample_random_posterior(...)`.
- Add a general `bami.utils.posterior_to_dataframe(...)` helper for users who
  want tidy posterior tables for analysis, plotting, or export.
- Sampling dictionaries may include raw keys such as `theta_raw`,
  `theta_mu_raw`, `theta_log_sigma`, `theta_subj_raw`, and `theta_z`.
  `posterior_to_dataframe(..., include_raw=False)` should hide those by
  default and show researcher-facing public-scale values.
- Use the minimal posterior dataframe columns:
  `dataset`, `draw`, `level`, `param`, `basis`, `quantity`, and `value`.
- `value` is on the user-interpretable scale by default. Raw-space values are
  opt-in through `include_raw=True`, using readable `quantity` labels such as
  `raw`, `mu_raw`, `log_sigma`, and `z` rather than extra dataframe columns.
- Use `basis="global"` for simple or non-condition-specific parameters.
- Use `basis="subject:<slot>"` for existing random-effect subject samples.
- Represent correlation rows with `param="cor"`, `quantity="corr"`, and a
  readable `basis` such as `drift_A1:B1__drift_A1:B2` when correlations are
  available in later workflows.

### 2. Subject Truth Defaults

- Update hierarchical truth handling before `MultiConditionWorkflow` so
  `keep_subject_truth=None` saves all stochastic subject-level truth,
  `keep_subject_truth=["a", "c"]` saves only listed parameters, and
  `keep_subject_truth=[]` saves none.
- Apply this rule to existing `HierarchicalWorkflow` first. `SimpleWorkflow`
  does not have subject-level truth.
- Update examples, diagnostics, and tests that currently assume subject truth
  is opt-in.

### 3. Documentation and Regression Checks

- Update docs and examples to show that `sample_*` methods return dictionaries,
  and that `posterior_to_dataframe(...)` is the analysis/reporting conversion
  step.
- Keep internal diagnostics working with the existing dictionary sample
  contract.
- Full checks passed after this milestone:
  `uv run pytest`, `uv run ruff check .`, `uv run black --check .`, and
  `uv run mkdocs build`. `pytest` reported only the existing Keras/Torch NumPy
  deprecation warnings, `black --check` reported the existing Python 3.14 /
  Python 3.15 target warning, and `mkdocs build` reported the existing Material
  for MkDocs 2.0 warning.

## Milestone 0.2.2: BayesFlow-Ind Compatibility and Validation

This milestone matches `bami` to the `2026-bayesflow-Ind` project workflow,
then debugs and validates the current package before new 0.3.0 features begin.

### 1. Match the Target Workflow

- Review the relevant `2026-bayesflow-Ind` usage patterns and expected workflow
  behavior.
- Compare those expectations against the current `bami` simple, hierarchical,
  training, sampling, posterior-dataframe, and subject-truth contracts.
- Record any mismatch as a concrete task with affected files, expected
  behavior, and a minimal test case.

### 2. Debug and Fix Compatibility Bugs

- Fix bugs exposed by matching `bami` to `2026-bayesflow-Ind`.
- Prefer small, focused fixes that preserve the 0.2.1 workflow-wide contracts.
- Do not start `MultiConditionWorkflow` implementation while compatibility bugs
  from this milestone remain unresolved.
- Fixed Apple Silicon MPS training compatibility for BayesFlow/Keras Torch
  workflows. When users request `torch_device="mps"` and MPS is available,
  `configure_torch_device(...)` now enables PyTorch's CPU fallback for
  unsupported MPS operations. This handles Keras orthogonal initializer QR
  operations such as `aten::linalg_qr.out`, while still selecting MPS for
  supported operations.
- The fallback environment variable is initialized during `bami` package import
  so it is present before BayesFlow, Keras, or Torch initialize their backend.
- Unavailable `mps` and `cuda` requests still fall back to CPU, but now print a
  short user-facing message explaining the selected fallback device.

### 3. Validation

- Add regression tests for every compatibility bug that is fixed.
- Validate at least one representative simple workflow and one representative
  hierarchical workflow from the target usage pattern.
- Added runtime regression tests for available MPS with CPU fallback,
  unavailable MPS fallback, unavailable CUDA fallback, explicit CPU selection,
  and invalid device-name errors.
- Verified the MPS fallback path with a small Keras `Orthogonal()` initializer
  smoke test after `configure_torch_device("mps")`; the initializer completed
  with `PYTORCH_ENABLE_MPS_FALLBACK=1`.
- Run the full check set after fixes:
  `uv run pytest`, `uv run ruff check .`, `uv run black --check .`, and
  `uv run mkdocs build`.
- Record any unresolved compatibility gap with exact reproduction steps before
  starting 0.3.0 work.

## Milestone 0.3.0: MultiConditionWorkflow

Start this milestone only after 0.2.1 workflow-wide API alignment and 0.2.2
BayesFlow-Ind compatibility validation are complete.

This milestone adds `MultiConditionWorkflow` for aggregate multi-condition
hierarchical models. The workflow is parallel to `HierarchicalWorkflow`, not an
extension of `SimpleWorkflow`. It uses cell-specific coding, similar to
`0 + condition`, and estimates group means, subject-level SDs, and selected
subject-level raw-effect correlations.

### 1. Workflow Scope and Data Contract

- Add a new public `MultiConditionWorkflow`.
- Support only `observation="aggregate"` in the first implementation.
- Keep `SimpleWorkflow` separate because simple workflows do not have a
  subject-level correlation structure.
- Keep simulator calls condition-agnostic. The workflow chooses parameter
  values for each condition cell, calls
  `simulator(**params, n_trials=..., rng=...)`, and stores the returned row in
  the matching condition cell.
- Preserve the condition axis in simulated data:
  `sim["data"].shape == (n_datasets, n_subjects, n_condition_cells,
  feature_width)`.
- For BayesFlow summaries, flatten each subject's condition-by-feature matrix
  in fixed condition order, then summarize the exchangeable subject rows with a
  set network.
- Treat subjects as exchangeable. Do not treat condition cells as exchangeable.

### 2. Condition Design

- Name the workflow argument `condition`.
- Allow users to provide manual cells:
  `condition=[{"A": "A1", "B": "B1"}, {"A": "A1", "B": "B2"}]`.
- Add a `condition_factors(...)` helper under `bami.utils` to generate full
  factorial cells, for example
  `condition=condition_factors({"A": ["A1", "A2"], "B": ["B1", "B2"]})`.
- `condition_factors(...)` returns a small `ConditionDesign` object.
- `condition=` accepts either a `ConditionDesign` or a manually written list of
  cell dictionaries. The workflow normalizes both forms to `ConditionDesign`.
- Internally store `cells`, `cell_names`, and `factors` on `ConditionDesign`
  because these data are small and are needed for validation, parameter naming,
  and dataframe conversion.
- Use `:` between factor levels in cell labels, for example `A1:B2`. Do not
  use `_` inside condition cell labels.

### 3. Effects Syntax

- Use one `effects` argument to define both condition basis and correlation
  mode:
  `effects=["a ~ A | none", "c ~ A:B | levels",
  "a + c + slope ~ A:B | pairs", "bias + asy ~ A:B | block"]`.
- `none` means the parameter varies by the basis, but subject-level effects are
  independent.
- `levels` means the same parameter is correlated across basis levels or cells.
- `pairs` means different parameters are correlated within the same basis level
  or cell.
- `block` means one full joint correlation block over parameters by basis
  levels or cells.
- Parameters not listed in `effects` default to full-cell basis with `none`.
- The right-hand side of `~` defines the basis. `A:B` means the factorial cell
  basis for factors `A` and `B`.
- `block` already includes the corresponding `levels` and `pairs` structure, so
  overlapping `block` and `pairs` or `levels` declarations are invalid.
- Add `validate_effect_terms(...)` to parse and validate effects. Conflicts
  raise `ValueError`; harmless duplicate or mergeable terms issue
  `warnings.warn(...)`; the final contract deduplicates pairs and blocks.

### 4. Priors and Correlations

- Each parameter-by-basis-level effect gets its own group mean and
  subject-level SD.
- Each level uses the same `priors[param]` template, but the resulting group
  parameters are distinct.
- Estimate correlations as group-level parameters on the raw subject-effect
  scale.
- Add correlation-specific LKJ support outside the scalar prior parser.
- Default `corr_prior` to `"lkj(1)"` and use it globally for all `levels`,
  `pairs`, and `block` structures.
- `none` does not use `corr_prior`.

### 5. Parameter Naming

- Use brms-like flat keys with `_` separating semantic fields and `:`
  separating condition factor levels.
- Internal group inference variables use
  `param_basis_quantity_scale`, for example
  `drift_A1:B2_mu_raw` and `drift_A1:B2_sigma_log`.
- Posterior transforms may add public keys such as `drift_A1:B2_mu` and
  `drift_A1:B2_sigma`.
- Correlation keys use `cor_left__right`, for example
  `cor_drift_A1:B1__drift_A1:B2`.
- Correlation keys do not include `_raw`; documentation should state that
  correlations are raw-scale correlations.

### 6. Subject Truth and Simulator Mapping

- Build on the 0.2.1 hierarchical truth rule:
  `keep_subject_truth=None` saves all stochastic subject-level truth,
  `keep_subject_truth=["a", "c"]` saves only listed parameters, and
  `keep_subject_truth=[]` saves none.
- For `MultiConditionWorkflow`, save subject truth as arrays:
  `sim["<param>_subj"].shape == (n_datasets, n_subjects, n_condition_cells)`.
- Broadcast lower-dimensional basis parameters to full condition cells when
  saving subject truth so truth arrays align with `sim["data"]`.
- Broadcast lower-dimensional basis parameters into simulator calls. For
  example, with full cells `A1:B1`, `A1:B2`, `A2:B1`, and `A2:B2`, an
  `a ~ A` effect passes `a_A1` to both `A1` cells and `a_A2` to both `A2`
  cells.
- Different parameters may use different bases.

### 7. Summary Network

- Add a small `MultiConditionSummary` wrapper instead of pre-flattening the
  simulated data before BayesFlow sees it.
- The first aggregate path takes data shaped
  `batch x subjects x condition_cells x features`, flattens each subject's
  condition-by-feature matrix in fixed condition order, and summarizes the
  subject axis with `DeepSet`.
- Keep the wrapper structure extensible for later trial and flexible designs:
  future trial support can add a trial encoder before condition flattening, and
  future flexible support can add subject or trial masks without changing the
  public workflow contract.
- Do not treat condition cells as exchangeable in the summary network.

### 8. Posterior Sampling and Dataframe Conversion

- Build on the 0.2.1 sampling contract. Public
  `MultiConditionWorkflow.sample_group_posterior(...)` should return a
  dictionary with raw group keys and public-scale keys, matching existing
  workflows.
- Extend `posterior_to_dataframe(...)` or a small companion helper so
  multi-condition posterior dictionaries can be converted into tidy tables with
  columns `dataset`, `draw`, `level`, `param`, `basis`, `quantity`, and
  `value`.
- Use `basis` to identify condition cells or effect bases such as `A1:B2`.
- `value` is on the user-interpretable scale by default; raw-space values stay
  opt-in through `include_raw=True`.
- Represent correlation rows with `param="cor"`, `quantity="corr"`, and a
  readable `basis` such as `drift_A1:B1__drift_A1:B2`.

### 9. First Implementation Tests

- First implementation tests should focus on contracts and simulation, not
  BayesFlow training smoke tests or full documentation examples.
- Add contract tests for `condition_factors(...)`, manual condition
  normalization, `ConditionDesign`, and fixed cell-name ordering.
- Add parser and validation tests for valid and invalid `effects` terms,
  including unknown parameters, unknown factors, conflicting bases, duplicate
  mergeable terms, and invalid `block` overlap.
- Add LKJ tests for valid positive-definite correlation matrices and invalid
  correlation prior strings.
- Add simulation tests for `sim["data"]` shape, flat group truth keys, subject
  truth shape, lower-dimensional basis broadcasting into simulator calls, and
  the guarantee that simulators do not receive condition metadata.
- Add dataframe schema tests for the explicit posterior conversion utility,
  not for `sample_*` return values.

## Future Backlog: New Features and Research Prototypes

These items are intentionally deferred beyond the focused 0.3.0
`MultiConditionWorkflow` milestone.

### 1. Reorganize Parameter API Ownership

- Keep the package organized by responsibility: `workflows`, `parameters`,
  `evaluation`, `inference`, `simulators`, `inputs`, and `recovery`.
- Evaluate whether a separate user-facing `parameters` module is still needed
  now that model-level sampling methods exist. Do not add a new layer unless it
  removes real user confusion or duplication.
- Add `model.recover(...)` only if recovery workflows need the model object as
  their natural first argument.
- Do not add plot or scalar-metric wrappers to workflow classes. Plotting,
  diagnostics, and metrics should keep operating on arrays or data frames in
  `evaluation`.
- Redesign `bami.recovery` from dataframe-first inputs and current model-level
  posterior sampling. Do not restore the old evaluation-owned `bf_*` or
  `estimate_*` recovery wrappers.

### 2. Adjust API Reference Around New Ownership

- Continue checking API reference pages for old examples that reach through
  `model.workflow` for common tasks.
- Keep user-facing docs on model-level sampling methods. Do not reintroduce
  `bami.evaluation.sample_posterior(workflow, ...)`.

### 3. Design `sample_random_posterior`

- Validate the BayesFlow-only random workflow route on real examples and
  recovery-style checks.
- Keep the user-facing flow explicit:
  `model.train_random_workflow(file=..., overwrite=False)`,
  `group_samples = model.sample_group_posterior(...)`, then
  `model.sample_random_posterior(observed_data, group_samples)`.
- `sample_random_posterior(...)` requires explicit `group_samples`; it should
  not estimate group parameters internally from subject-level `observed_data`.
- Preserve the simulator-first BayesFlow workflow and avoid analytic likelihood
  requirements.
- The likelihood-weighted posthoc path has been removed. Future subject-level
  recovery tables should be built from random-effect posterior draws.
- Move PyMC compatibility to a future advanced compatibility investigation, not
  an active implementation target for this milestone.

### 4. Multi-Condition Subject Workflows

- Design workflows for subjects with multiple condition-specific datasets.
- Distinguish shared subject parameters estimated from multiple conditions from
  condition-specific parameters with explicit covariance or correlation
  structure.
- Define simulator, prior, posterior-table, and data-shape contracts before
  writing a public API.

### 5. BayesFlow CompositionalWorkflow Evaluation

- Evaluate `CompositionalWorkflow` only as a candidate route for
  multi-condition or multi-dataset evidence composition.
- Compare it with the current `BasicWorkflow` hierarchy approach before making
  public API changes.
- Do not assume `CompositionalWorkflow` estimates cross-condition correlations
  by itself. The simulator and prior must represent any joint structure.

### 6. Alternative Summary Networks

- Track `SetTransformer` or related networks as future evaluation targets for
  exchangeable observations.
- Keep `DeepSet` as the default unless benchmarks show a clear benefit for
  typical psychology-modeling examples.
- Require a short design note or small benchmark before changing defaults or
  exposing new public network configuration.

### 7. Advanced BayesFlow Features

- Consider ensembles, wrappers, and other advanced BayesFlow APIs only when
  they solve a real `bami` user problem.
- Do not expose new BayesFlow options only because they exist.
- Prefer clear user-facing workflows over broad compatibility layers.

### 8. New Shared Simulator Abstractions

- Add shared helpers only if current-code review shows repeated logic that
  harms readability.
- Keep any helper small, documented, and tested.
- Avoid a broad utility layer unless it removes real duplication.

### 9. Future PyMC Compatibility

- Revisit PyMC only after the BayesFlow-only `train_random_workflow(...)` and
  `sample_random_posterior(...)` route is validated.
- Treat PyMC as an optional advanced bridge for users who specifically need
  MCMC diagnostics or composition with PyMC models.
- Do not require users to write analytic PyMC likelihoods for standard `bami`
  workflows.
- Do not add PyMC as a required dependency.

## Completed Work

### Evaluation Metrics API Cleanup

Completed.

- Replaced posterior-specific `summarize_*` helpers with
  `aggregate_data(...)`, which summarizes one or more numeric columns by
  optional grouping columns.
- Added `estimate_recovery(...)` as the single dataframe-first recovery metric
  helper for paired simulated and estimated long tables.
- Removed public `sample_posterior` from `bami.evaluation`; posterior sampling
  remains model-owned through `SimpleWorkflow.sample_posterior(...)`,
  `HierarchicalWorkflow.sample_group_posterior(...)`, and
  `HierarchicalWorkflow.sample_random_posterior(...)`.
- Removed old `bf_*` and brms placeholder metrics from evaluation exports and
  documentation.
- Deleted the legacy `evaluation.metrics.bayesflow` implementation and moved
  the remaining internal posterior sampling helper to
  `bami.workflows._sampling`.
- Reorganized the Evaluation metrics documentation around data aggregation,
  recovery metrics, and scalar metrics, and increased the MkDocs TOC depth so
  function names appear in the page table of contents.
- `uv run pytest tests/test_evaluation_scalar_metrics.py` passed.
- `uv run pytest tests/test_evaluation_contracts.py tests/test_evaluation_plots.py`
  passed as regression coverage only; the plotting API and docs still need a
  separate design/readability review.
- `uv run pytest tests/test_evaluation_metrics_bayesflow.py` passed after
  narrowing it to workflow sampling and random-posterior behavior.
- `uv run ruff check src/bami/workflows src/bami/evaluation tests/test_evaluation_metrics_bayesflow.py`
  passed.
- `uv run mkdocs build` passed with the upstream Material for MkDocs 2.0
  warning only.

### Evaluation Diagnostics Plot Cleanup

Completed.

- Deleted the old public `bami.evaluation.plots` API and the old recovery and
  diagnostic table-contract validators.
- Added `bami.evaluation.diagnostics` as the home for diagnostic plotting
  logic, with workflow methods kept as shallow user-facing aliases.
- Added `plot_parameter_recovery(...)` for simple workflows and
  `plot_population_recovery(...)` for hierarchical population parameters.
  These scatter plots use one shared `metrics` argument, defaulting to
  `"corr"`, and show requested metrics in panel titles.
- Added `plot_random_recovery(...)` for hierarchical random parameters. It now
  computes one dataset-level metric per parameter and plots the metric
  distribution with box plots and jittered points.
- Removed user-facing sampling controls and visual style controls from the
  diagnostic plot API. Diagnostics use a conservative internal sampling batch
  size and fixed plotting style.
- Updated `docs/api/evaluation/diagnostics.md` so each diagnostic function is
  documented before its example, and the page explains that model comparison
  should use explicit dataframes plus `estimate_recovery(...)`.
- `uv run pytest tests/test_evaluation_metrics_bayesflow.py` passed.
- `uv run pytest tests/test_evaluation_scalar_metrics.py` passed.
- `uv run ruff check src tests` passed.
- `uv run mkdocs build` passed with the upstream Material for MkDocs 2.0
  warning only.

### Legacy Recovery Module Removal

Completed.

- Deleted the old `bami.recovery` module and its public
  `simulate` / `recover` / `summarize` workflow.
- Removed the recovery API page and navigation entry.
- Deleted the obsolete group recovery tests.
- Current recovery-style summaries should be built explicitly from
  `model.simulate(...)`, `model.sample_*`, `aggregate_data(...)`, and
  `estimate_recovery(...)`.

### Complete Workflow Examples

Completed.

- Expanded the simple SDM example into a full workflow covering model setup,
  validation simulation, saved workflow training/loading, posterior sampling,
  long-format recovery tables, `aggregate_data(...)`, `estimate_recovery(...)`,
  and `model.plot_parameter_recovery(...)`.
- Expanded the hierarchical ezDM example into a full workflow covering group
  and random-effect training, `sample_group_posterior(...)`,
  `sample_random_posterior(...)`, group-level and subject-level recovery
  tables, and population/random diagnostic plots.
- Updated the website landing page to describe these as complete workflow
  examples rather than short examples.
- Updated the README evaluation section to mention model diagnostic plots.
- `uv run mkdocs build` passed with the upstream Material for MkDocs 2.0
  warning only.
- `git diff --check` passed.

### Model-Owned Posterior Sampling APIs

Completed.

- Added `model.simulate(...)` wrappers so examples no longer need to call
  `model.workflow.simulate(...)` for ordinary use.
- Added `SimpleWorkflow.sample_posterior(...)` for simple parameter posterior
  draws and `HierarchicalWorkflow.sample_group_posterior(...)` for group-level
  hierarchical draws.
- Kept `HierarchicalWorkflow.sample_posterior(...)` absent so group and
  subject-level outputs stay explicit.
- Later evaluation cleanup removed the temporary `summarize_*` helpers and the
  public `bami.evaluation.sample_posterior(...)` compatibility helper.
- Updated README, API pages, and examples to use model-level simulation and
  posterior sampling.
- `uv run pytest` passed with the existing Keras/Torch NumPy deprecation
  warnings only.
- `uv run ruff check src tests` passed.
- Touched-file `uv run black --check ...` passed.
- `uv run mkdocs build` passed with the upstream Material for MkDocs 2.0
  warning only.

### BayesFlow-Only Random-Effect Workflow Prototype

Completed.

- Added `HierarchicalWorkflow.train_random_workflow(...)` as a separate
  subject-level random-effect workflow trained on standardized deviations
  such as `theta_z`.
- `train_workflow(...)` now stores its effective training config on the model;
  `train_random_workflow(..., inherit_config=True)` inherits that config and
  applies explicit overrides.
- Added `HierarchicalWorkflow.sample_random_posterior(observed_data,
  group_samples, ...)`, using explicit group posterior draws from
  `model.sample_group_posterior(...)`.
- Random posterior sampling combines paired group draws and sampled
  standardized deviations into public subject-level parameters while
  preserving output shape `(n_datasets, n_samples, n_subjects)`.
- Trial-level random sampling now follows the model's trial design: fixed
  trial mismatches warn but continue, and flexible trial inputs are padded and
  given `active_trial` masks internally when needed.
- Removed the public `summarize_subject_posterior(...)` helper, legacy posthoc
  internals, and likelihood-distribution helpers used only by posthoc
  candidate weighting.
- Kept exchangeable hierarchy individual recovery function names as
  placeholders that point users to `sample_random_posterior(...)` until the
  recovery tables are rebuilt.
- `uv run pytest tests/test_evaluation_metrics_bayesflow.py` passed.
- `uv run ruff check src tests` passed.
- Touched-file `uv run black --check ...` passed.
- `uv run mkdocs build` passed with the upstream Material for MkDocs 2.0
  warning only.

### Advanced Docs and Unified Training API

Completed.

- Moved input-format and inference details under `API Reference > Advanced` so
  standard workflow pages stay easier to scan before advanced internals.
- Removed the separate public `fit_workflow()` helper and made
  `model.train_workflow(...)` the only user-facing training entry point.
- Moved shared training behavior into internal workflow helpers so
  `SimpleWorkflow` and `HierarchicalWorkflow` both delegate to the same saved
  workflow, device-selection, and training-loop implementation.
- Updated examples and docs to use researcher-facing saved workflow language
  such as `saved_workflows/...` instead of checkpoint-oriented examples.
- Renamed the saved-workflow tests around `train_workflow()` behavior and
  updated tiny SDM training tests to use the workflow method directly.
- `uv run pytest tests/test_train_workflow_saved_workflow.py` passed.
- `uv run pytest tests/test_sdm_fixed_simple.py` passed.
- `uv run mkdocs build` passed with the upstream Material for MkDocs 2.0
  warning only.
- Full `uv run pytest` passed with the existing Keras/Torch NumPy deprecation
  warnings only.

### Workflow and Simulator Website Polish

Completed.

- Reorganized the workflow reference so researchers first see workflow choice,
  setup order, prior structure, and minimal examples before constructor
  details.
- Added clearer prior documentation for the current `mean` / `sd` / `link`
  format, including fixed simulator constants and supported link functions.
- Simplified the SDM simulator API around continuous signed circular errors in
  radians: `simulate_sdm_simple(c, kappa, n_trials=100, rng=None)`.
- Removed the old SDM degree-bin user interface, including `GRID_SIZE`,
  `grid_size`, `error_scale`, `jitter`, `sdm_probs`, and degree/index helper
  exports.
- Updated SDM examples and fixtures to use `obs_names=["error_rad"]`, with
  simulated and observed SDM data documented as radians in `[-pi, pi]`.
- Aligned simulator navigation with the M3 and ezDM pages by exposing SDM as
  `SDM` and featuring only the workflow-facing simulator function.
- Full `uv run pytest` passed with the existing Keras/Torch NumPy deprecation
  warnings only.
- `uv run mkdocs build` passed with the upstream Material for MkDocs 2.0
  warning only.

### BayesFlow 2.0.12 Compatibility

Completed.

- Dependency range now targets `bayesflow>=2.0.12,<2.1` in `pyproject.toml`
  and `uv.lock`.
- Local environment verified BayesFlow `2.0.12` with the Torch backend.
- Current defaults remain the fixed `DeepSet` and `CouplingFlow` setup.
- No public `summary_network` or `inference_network` arguments were added.
- `SetTransformer`, `CompositionalWorkflow`, PyMC wrappers, ensembles, and
  other advanced BayesFlow features were not adopted during this compatibility
  pass.
- Targeted compatibility tests passed for workflow construction, tiny SDM
  training, posterior sampling, diagnostics, and checkpoint helper behavior.
- Full `uv run pytest` passed with the existing Keras/Torch NumPy deprecation
  warnings only.
- `uv run ruff check src tests` passed.
- `uv run black --check src tests` only reported formatting in
  `src/bami/simulators/m3.py` and `tests/fixtures_model_specs.py`; no behavior
  changes were needed for BayesFlow compatibility.

Useful sources:

- PyPI: <https://pypi.org/project/bayesflow/>
- GitHub releases: <https://github.com/bayesflow-org/bayesflow/releases>
- BayesFlow user guide: <https://bayesflow.org/v2.0.12/user_guide/index.html>

## Development Workflow

For each change:

1. Start from a focused issue or task.
2. Read nearby code, docs, and tests before editing.
3. Make the smallest clear change that solves the task.
4. Add or update tests for the behavior touched.
5. Run the local checks:

```bash
uv run pytest
uv run ruff check src tests
uv run black --check src tests
uv run mkdocs build
```

If Black reports formatting changes, run:

```bash
uv run black src tests
```

Then re-run the checks.

## Readability Checklist

- Is the main data flow visible from top to bottom?
- Are function names compact but understandable?
- Does each touched function have a useful docstring?
- Do comments explain assumptions or domain reasoning instead of restating the
  code?
- Are tests named after the behavior they protect?
- Can a psychology researcher understand the example without learning extra
  software-engineering patterns first?
- Does the website guide users through the standard workflow before advanced
  internals?

## Release Notes To Track

- Public API changes.
- New or changed workflow behavior.
- New examples or documentation pages.
- Any change in required Python version or dependencies.
- Known limitations that users should account for in research projects.
