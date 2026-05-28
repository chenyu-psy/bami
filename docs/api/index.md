# Package index

This page groups the main `bami` functions and classes by the part of an
analysis where they are usually used.

## Workflows

Start here when building a BayesFlow workflow around a simulator.

[`SimpleWorkflow`](workflows/simple.md)
: Build a non-hierarchical workflow where one prior draw generates one dataset.

[`HierarchicalWorkflow`](workflows/hierarchical.md)
: Build a group-level workflow where each simulated dataset contains multiple subjects.

## Inputs

Use these functions when workflow input rows need explicit trial-count encoding.

[`aggregate_summary`](inputs/formats.md)
: Preserve aggregate summary features and append encoded `n_trials`.

[`proportions`](inputs/formats.md)
: Preserve proportion features and append encoded `n_trials`.

[`counts`](inputs/formats.md)
: Preserve count rows and optionally append encoded `n_trials`.

## Simulators

Use these functions to simulate cognitive-model summaries or trial rows for
workflows. Model-specific helpers are documented on each simulator page.

[`simulate_sdm_simple`](simulators/sdm-circular.md)
: Simulate trial-level circular errors for a standard SDM workflow.

[`simulate_ezdm_simple`](simulators/ezdm.md)
: Simulate one ezDM summary row with `pc`, `mrt`, and `vrt`.

[`simulate_m3_custom`](simulators/m3.md)
: Simulate response-count vectors for M3 workflows.

## Training

Use this function when a configured workflow should be fitted with shared
training settings and optional checkpoint loading or saving.

[`fit_workflow`](training/fit-workflow.md)
: Fit a workflow and optionally reuse or save a `.keras` checkpoint.

## Evaluation

Use these functions after training to turn posterior samples into recovery and
diagnostic tables.

[`compute_corr`, `compute_ccc`, and `compute_rmse`](evaluation/metrics.md)
: Compute scalar agreement and error metrics for true vs. estimated values.

[`sample_posterior`](evaluation/metrics.md)
: Sample posterior draws and apply public-space transforms when available.

[`estimate_population_recovery`](evaluation/metrics.md)
: Convert posterior samples into population-level recovery rows.

[`estimate_fixed_individual_recovery`](evaluation/metrics.md)
: Convert fixed-subject posterior samples into individual recovery rows.

[`estimate_flex_individual_recovery`](evaluation/metrics.md)
: Estimate individual recovery for flexible hierarchical workflows.

[`bf_pop_recovery`](evaluation/metrics.md)
: Sample and tabulate BayesFlow population recovery in one call.

[`bf_ind_recovery`](evaluation/metrics.md)
: Sample and tabulate BayesFlow individual recovery in one call.

[`bf_calibration`, `bf_coverage`, and `bf_zscore`](evaluation/metrics.md)
: Compute diagnostic metric tables from BayesFlow defaults.

[`plot_population_recovery`](evaluation/plots-contracts.md)
: Plot population-level true vs. estimated parameter recovery.

[`plot_individual_recovery`](evaluation/plots-contracts.md)
: Plot individual recovery correlations by dataset and parameter.

[`validate_recovery_contract`](evaluation/plots-contracts.md)
: Check the shared recovery table schema before plotting or saving.

## Recovery workflows

Use these functions for group-generated recovery checks.

[`simulate`](recovery.md)
: Simulate group-level recovery datasets.

[`recover`](recovery.md)
: Estimate population and individual recovery rows for one fitted workflow.

[`summarize`](recovery.md)
: Summarize recovery rows into tables and figures.
