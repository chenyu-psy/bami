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

### 3. Review Existing Evaluation and Recovery Tools

- Check recovery tables, scalar metrics, diagnostics, and plotting helpers.
- Improve consistency of parameter naming and table contracts.
- Add or refine tests for current edge cases such as missing parameters, fixed
  parameters, and small simulated datasets.
- Keep plots readable without requiring users to customize many arguments.

### 4. Polish Website Structure

- Review `mkdocs.yml`, `docs/index.md`, API pages, and examples.
- Make standard workflows easier to find before advanced internals.
- Keep `README.md` focused on the package overview and first successful use.
- Keep `docs/` pages organized by task: setup, simulation, training,
  evaluation, recovery, and examples.
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

## Current Milestone Acceptance Criteria

- Existing public APIs are simple, explicit, and teachable.
- Existing docs guide users through standard workflows before advanced
  internals.
- Examples match current code and show expected array or dataframe shapes.
- Touched functions have useful docstrings.
- Comments explain assumptions or domain reasoning instead of restating the
  code.
- Checks pass, or remaining failures are recorded with exact commands and next
  actions.
- No new public features are added unless required to fix current behavior.

## Next Milestone: New Features and Research Prototypes

These items are intentionally deferred. They should start only after the current
code review and website polish milestone is complete.

### 1. Reorganize Parameter API Ownership

- Keep the package organized by responsibility: `workflows`, `parameters`,
  `evaluation`, `inference`, `simulators`, `inputs`, and `recovery`.
- Add a user-facing `parameters` layer for posterior-sampling functions.
  `sample_group_posterior(...)` should replace the current user-facing
  `sample_posterior(...)` name while keeping the old name as a compatibility
  alias.
- Add thin workflow wrappers only when the model is the natural first argument:
  `model.simulate(...)`, `model.sample_group_posterior(...)`,
  `model.recover(...)`, and later `model.train_random_likelihood(...)` and
  `model.sample_random_posterior(...)`.
- Do not add plot or scalar-metric wrappers to workflow classes. Plotting,
  diagnostics, and metrics should keep operating on arrays or data frames in
  `evaluation`.
- Add `summarize_group_parameters(...)` and `summarize_random_parameters(...)`
  to `evaluation`. These functions should consume posterior samples or draws
  and return data frames with `estimate` (posterior mean), `sd`, `ci_lower`,
  `ci_upper`, and `ess` when ESS is available.
- Keep the current `summarize_subject_posterior(...)` behavior for now, but
  mark it as an upcoming legacy/outdated path. Do not remove it until the
  BayesFlow/PyMC random-posterior route is validated.
- Reserve the public names `sample_random_posterior(...)` and
  `model.sample_random_posterior(...)`, but do not implement PyMC behavior in
  this cleanup pass.

### 2. Adjust API Reference Around New Ownership

- Update the Parameters page to feature group-level posterior sampling through
  `sample_group_posterior(...)` and to reserve space for future random-effect
  posterior sampling.
- Update the Evaluation page to document `summarize_group_parameters(...)` and
  `summarize_random_parameters(...)` as data-frame summaries of posterior
  draws.
- Move `summarize_subject_posterior(...)` out of the main Parameters workflow
  path and label it as a legacy/approximate posthoc helper that remains
  available while the random-posterior sampling design is evaluated.
- Update examples so users can avoid reaching through `model.workflow` for
  common tasks. Prefer `model.simulate(...)` and
  `model.sample_group_posterior(...)` in user-facing docs.
- Keep compatibility notes brief and researcher-facing. Explain old names only
  where needed to help existing users migrate.

### 3. Design `sample_random_posterior`

- Evaluate the BayesFlow-native route before implementing PyMC-specific code:
  train a subject-level neural likelihood or ratio estimator from the existing
  simulator, then use PyMC only as the optional sampler over random effects.
- Keep PyMC optional. Users should only need the PyMC extra when they call
  `sample_random_posterior(...)`.
- Prefer the user-facing flow
  `model.train_random_likelihood(file=..., overwrite=False)` followed by
  `model.sample_random_posterior(...)`.
- Preserve the simulator-first BayesFlow workflow. Do not require users to
  write analytic PyMC likelihoods for normal `bami` workflows.
- Treat the existing likelihood-weighted posthoc path as a fast approximate
  route until the neural likelihood/ratio plus PyMC design is proven reliable
  and teachable.

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

## Completed Work

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
