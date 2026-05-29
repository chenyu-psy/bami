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

- Review the current simple SDM and hierarchical ezDM examples.
- Ensure examples match the current API and can be copied into a research
  project with minimal editing.
- Show expected array or dataframe shapes after major steps.
- Prefer examples that demonstrate existing supported workflow patterns before
  introducing advanced internals.

### 6. Run Checks and Record Remaining Gaps

- Run tests, lint, format checks, and the website build.
- Record any remaining failures as concrete follow-up tasks with the exact
  command that failed.
- Keep unresolved issues separate from next-milestone feature ideas.

### 7. Review Runtime Compatibility and Defaults

- Check package behavior on common researcher systems: macOS, Linux, and
  Windows where feasible.
- Check whether users can choose an appropriate computation backend or device
  without editing package internals.
- Review backend/device documentation so CPU-only users, Apple Silicon users,
  and GPU users know what to expect.
- Audit default training settings for low-performance laptops and CPU-only
  environments. Defaults should be safe and teachable, even if advanced users
  later increase epochs, batch sizes, or worker counts.
- Add small smoke tests or documented manual checks for backend/device setup
  when full cross-platform CI is not available.
- Record any unsupported platform, backend, or device limitation explicitly in
  the docs instead of leaving users to infer it from errors.

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

- Run or document cross-platform smoke checks for macOS, Linux, and Windows.
- Review backend/device selection for BayesFlow/Keras/Torch and decide whether
  `bami` needs a documented user-facing setting beyond current
  `torch_device` training arguments.
- Review whether default training values such as epochs, batches, batch size,
  workers, and queue size are friendly to CPU-only and low-memory machines.
- Add documentation that explains recommended settings for low-performance
  laptops versus faster GPU machines.
- Record any checks that cannot be run locally, including the exact command or
  CI setup needed later.

## Next Milestone: New Features and Research Prototypes

These items are intentionally deferred. They should start only after the current
code review and website polish milestone is complete.

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
