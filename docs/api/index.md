# Package index

This page groups the main `bami` functions and classes by the part of an
analysis where they are usually used.

## Workflows

Start here when building a BayesFlow workflow around a simulator.

[`SimpleWorkflow` and `HierarchicalWorkflow`](workflows.md)
: Build non-hierarchical or group-level workflows around a simulator.

## Simulators

A simulator is a function that converts model parameters into simulated
behavioral performance or observation rows for a workflow. The pages below
document only the built-in simulators shipped with `bami`.

You can also write your own simulator. See the
[simulators article](../articles/simulators.md) for a step-by-step custom ezDM
example.

[`simulate_sdm_simple`](simulators/sdm-circular.md)
: Simulate trial-level circular errors for a standard SDM workflow.

[`simulate_ezdm_simple`](simulators/ezdm.md)
: Simulate one ezDM summary row with `pc`, `mrt`, and `vrt`.

[`simulate_m3_custom`](simulators/m3.md)
: Simulate response-count vectors for M3 workflows.

## Utils

Use these functions to reshape workflow outputs into analysis-ready tables.

[`posterior_to_dataframe`](utils.md)
: Convert posterior sample dictionaries into tidy dataframes for summaries,
plotting, or export.

[`aggregate_data`](utils.md)
: Aggregate one or more numeric columns by optional grouping columns.

## Evaluation

Use these functions to check recovery and model performance.

[`compute_corr`, `compute_ccc`, and `compute_rmse`](evaluation/metrics.md)
: Compute scalar agreement and error metrics for true vs. estimated values.

[`estimate_recovery`](evaluation/metrics.md)
: Compute grouped recovery metrics from simulated and estimated value tables.

[`plot_parameter_recovery`, `plot_population_recovery`, and `plot_random_recovery`](evaluation/diagnostics.md)
: Plot model-level recovery diagnostics for one fitted workflow.

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

Use these helpers when you need prior/link utilities or saved-workflow and
runtime controls.

[`Prior utilities`](inference/priors-transforms.md)
: Define priors and transform parameters between raw and public scales.

[`Saved workflows and runtime`](inference/checkpoints-runtime.md)
: Load, save, and inspect fitted workflow runtime details.
