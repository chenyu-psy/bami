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

### 1. PyMC-Based Posthoc Estimation

- Prototype an optional `PyMCPosthocEstimator` separately from the current
  lightweight posthoc path.
- Keep the current NumPy/Scipy posthoc path as the default for fast recovery
  checks and multiprocessing-friendly diagnostics.
- Do not add PyMC as a required dependency. If this project moves forward, use
  an optional dependency group and keep imports isolated.
- Start with one small M3 or SDM-style subject-level example.
- Compare PyMC posthoc output with the current likelihood-weighted posthoc
  output for model clarity, diagnostics, runtime cost, and agreement of
  estimates.

### 2. Multi-Condition Subject Workflows

- Design workflows for subjects with multiple condition-specific datasets.
- Distinguish shared subject parameters estimated from multiple conditions from
  condition-specific parameters with explicit covariance or correlation
  structure.
- Define simulator, prior, posterior-table, and data-shape contracts before
  writing a public API.

### 3. BayesFlow CompositionalWorkflow Evaluation

- Evaluate `CompositionalWorkflow` only as a candidate route for
  multi-condition or multi-dataset evidence composition.
- Compare it with the current `BasicWorkflow` hierarchy approach before making
  public API changes.
- Do not assume `CompositionalWorkflow` estimates cross-condition correlations
  by itself. The simulator and prior must represent any joint structure.

### 4. Alternative Summary Networks

- Track `SetTransformer` or related networks as future evaluation targets for
  exchangeable observations.
- Keep `DeepSet` as the default unless benchmarks show a clear benefit for
  typical psychology-modeling examples.
- Require a short design note or small benchmark before changing defaults or
  exposing new public network configuration.

### 5. Advanced BayesFlow Features

- Consider ensembles, wrappers, and other advanced BayesFlow APIs only when
  they solve a real `bami` user problem.
- Do not expose new BayesFlow options only because they exist.
- Prefer clear user-facing workflows over broad compatibility layers.

### 6. New Shared Simulator Abstractions

- Add shared helpers only if current-code review shows repeated logic that
  harms readability.
- Keep any helper small, documented, and tested.
- Avoid a broad utility layer unless it removes real duplication.

## Completed Work

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
