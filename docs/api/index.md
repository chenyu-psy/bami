# Package index

This page groups the main `bami` functions and classes by the part of an
analysis where they are usually used.

## Workflows

Start here when building a BayesFlow workflow around a simulator.

[`SimpleWorkflow` and `HierarchicalWorkflow`](workflows.md)
: Build non-hierarchical or group-level workflows around a simulator.

## Simulators

Use these functions to simulate cognitive-model summaries or trial rows for
workflows. Model-specific helpers are documented on each simulator page.

[`simulate_sdm_simple`](simulators/sdm-circular.md)
: Simulate trial-level circular errors for a standard SDM workflow.

[`simulate_ezdm_simple`](simulators/ezdm.md)
: Simulate one ezDM summary row with `pc`, `mrt`, and `vrt`.

[`simulate_m3_custom`](simulators/m3.md)
: Simulate response-count vectors for M3 workflows.

## Parameters

Use these functions after training when you want population-level or
individual-level parameter estimates.

[`sample_posterior`](parameters.md)
: Sample population or group-level posterior draws and apply public-scale
parameter transforms when available.

[`summarize_subject_posterior`](parameters.md)
: Summarize individual-level parameter posteriors from observed subject counts.

## Evaluation

Use these functions to check recovery, diagnostics, and model performance.

[`compute_corr`, `compute_ccc`, and `compute_rmse`](evaluation/metrics.md)
: Compute scalar agreement and error metrics for true vs. estimated values.

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

## Advanced

These pages are useful when a workflow needs custom input encoding, lower-level
inference utilities, or saved-workflow and runtime details.

### Input

Use these functions when workflow input rows need explicit trial-count encoding.

[`aggregate_summary`](inputs/formats.md)
: Preserve aggregate summary features and append encoded `n_trials`.

[`proportions`](inputs/formats.md)
: Preserve proportion features and append encoded `n_trials`.

[`counts`](inputs/formats.md)
: Preserve count rows and optionally append encoded `n_trials`.

### Inference

Use these helpers when you need prior/link utilities, likelihood building
blocks, or saved-workflow and runtime controls.

[`Prior utilities`](inference/priors-transforms.md)
: Define priors and transform parameters between raw and public scales.

[`Likelihood distributions`](inference/distributions.md)
: Use distribution helpers for likelihood-oriented workflows.

[`Saved workflows and runtime`](inference/checkpoints-runtime.md)
: Load, save, and inspect fitted workflow runtime details.
