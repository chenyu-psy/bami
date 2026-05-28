"""Tests for the public recovery workflow API."""

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from bami import recovery

TOY_PARAMS = ["theta", "alpha"]


class _GroupWorkflow:
    """Workflow stub that simulates group-generated recovery data."""

    def simulate(self, n_datasets: int) -> dict[str, np.ndarray]:
        """Create deterministic group and subject truth arrays.

        Parameters
        ----------
        n_datasets : int
            Number of group-level datasets to generate.

        Returns
        -------
        dict[str, np.ndarray]
            Simulated data with population and subject-level truth.
        """

        n_subjects = 4
        payload = {
            "data": np.zeros((n_datasets, n_subjects, 5), dtype=np.float32),
        }
        for base_param in TOY_PARAMS:
            pop_truth = np.linspace(0.2, 0.8, n_datasets, dtype=np.float32)
            payload[f"{base_param}_mu"] = pop_truth
            payload[f"{base_param}_subj"] = np.stack(
                [pop_truth + 0.03 * subj for subj in range(n_subjects)],
                axis=1,
            )
            for subj in range(n_subjects):
                payload[f"{base_param}_subj_{subj}"] = pop_truth + 0.03 * subj
        return payload


class _GroupGenerator:
    """Object exposing the simulation workflow expected by recovery.simulate."""

    def __init__(self):
        """Attach a deterministic group workflow."""

        self.workflow = _GroupWorkflow()


class _FitWorkflow:
    """Workflow stub that returns deterministic posterior samples."""

    def sample(
        self,
        *,
        num_samples: int,
        conditions: dict[str, np.ndarray],
        **kwargs,
    ) -> dict[str, np.ndarray]:
        """Return posterior samples aligned with the condition batch.

        Parameters
        ----------
        num_samples : int
            Number of posterior samples per dataset.
        conditions : dict[str, np.ndarray]
            BayesFlow-style conditions with a ``data`` tensor.
        **kwargs
            Ignored sampling options.

        Returns
        -------
        dict[str, np.ndarray]
            Posterior sample arrays keyed by population and subject parameters.
        """

        n_rows = int(np.asarray(conditions["data"]).shape[0])
        base_est = np.linspace(0.21, 0.81, n_rows, dtype=np.float32)
        samples = {}
        for base_param in TOY_PARAMS:
            samples[base_param] = np.tile(base_est[:, None], (1, num_samples))
            samples[f"{base_param}_mu"] = np.tile(base_est[:, None], (1, num_samples))
            for subj in range(4):
                subj_est = base_est + 0.03 * subj
                samples[f"{base_param}_subj_{subj}"] = np.tile(
                    subj_est[:, None],
                    (1, num_samples),
                )
        return samples


class _FitModel:
    """Minimal fitted model exposing a BayesFlow-like workflow."""

    data_width = 5

    def __init__(self):
        """Attach a deterministic fitted workflow."""

        self.workflow = _FitWorkflow()


def test_simulate_returns_group_payload():
    """Simulate should return group recovery payload metadata and data."""

    payload = recovery.simulate(generator=_GroupGenerator(), n_reps=7)

    assert payload["n_reps"] == 7
    assert payload["n_subjects"] == 4
    assert payload["sim_data"]["data"].shape == (7, 4, 5)


def test_recover_fixed_simple_returns_labeled_rows():
    """Recover should estimate population and individual rows for simple fits."""

    payload = recovery.simulate(generator=_GroupGenerator(), n_reps=3)
    population_rows, individual_rows = recovery.recover(
        fit_name="fixed_simple",
        fit_model=_FitModel(),
        sim_data=payload["sim_data"],
        base_params=TOY_PARAMS,
        posterior_samples=5,
    )

    assert set(population_rows["fit_model"]) == {"fixed_simple"}
    assert set(individual_rows["fit_model"]) == {"fixed_simple"}
    assert set(population_rows["level"]) == {"population"}
    assert set(individual_rows["level"]) == {"individual"}
    assert population_rows.shape[0] == 3 * len(TOY_PARAMS)
    assert individual_rows.shape[0] == 3 * 4 * len(TOY_PARAMS)


def test_recover_fixed_hierarchy_returns_labeled_rows():
    """Recover should estimate rows for hierarchical fits using one sample call."""

    payload = recovery.simulate(generator=_GroupGenerator(), n_reps=3)
    population_rows, individual_rows = recovery.recover(
        fit_name="fixed_hierarchy",
        fit_model=_FitModel(),
        sim_data=payload["sim_data"],
        base_params=TOY_PARAMS,
        posterior_samples=5,
    )

    assert set(population_rows["fit_model"]) == {"fixed_hierarchy"}
    assert set(individual_rows["fit_model"]) == {"fixed_hierarchy"}
    assert population_rows.shape[0] == 3 * len(TOY_PARAMS)
    assert individual_rows.shape[0] == 3 * 4 * len(TOY_PARAMS)


def test_recover_requires_base_params():
    """Recover should not infer model-specific parameter names."""

    payload = recovery.simulate(generator=_GroupGenerator(), n_reps=1)

    with pytest.raises(ValueError, match="base_params"):
        recovery.recover(
            fit_name="fixed_simple",
            fit_model=_FitModel(),
            sim_data=payload["sim_data"],
            base_params=[],
            posterior_samples=5,
        )


def test_summarize_returns_tables_and_figures():
    """Summarize should return diagnostics and figures without writing files."""

    payload = recovery.simulate(generator=_GroupGenerator(), n_reps=4)
    pop_simple, ind_simple = recovery.recover(
        fit_name="fixed_simple",
        fit_model=_FitModel(),
        sim_data=payload["sim_data"],
        base_params=TOY_PARAMS,
        posterior_samples=5,
    )
    pop_hier, ind_hier = recovery.recover(
        fit_name="fixed_hierarchy",
        fit_model=_FitModel(),
        sim_data=payload["sim_data"],
        base_params=TOY_PARAMS,
        posterior_samples=5,
    )
    population_rows = pd.concat([pop_simple, pop_hier], ignore_index=True)
    individual_rows = pd.concat([ind_simple, ind_hier], ignore_index=True)

    outputs = recovery.summarize(
        population_rows=population_rows,
        individual_rows=individual_rows,
        base_params=TOY_PARAMS,
        quantile_bias_params=["theta_mu"],
        fit_models=["fixed_simple", "fixed_hierarchy"],
    )

    assert {
        "population_summary",
        "population_diagnostics",
        "individual_correlations",
        "population_figure",
        "individual_figure",
        "param_quantile_bias",
    }.issubset(outputs)
    assert {"ccc", "n"}.issubset(outputs["population_summary"].columns)
    assert {"r", "n"}.issubset(outputs["individual_correlations"].columns)
    assert isinstance(outputs["population_figure"], matplotlib.figure.Figure)
    assert isinstance(outputs["individual_figure"], matplotlib.figure.Figure)

    plt.close(outputs["population_figure"])
    plt.close(outputs["individual_figure"])


def test_summarize_requires_base_params():
    """Summarize should fail clearly when plot parameters are omitted."""

    rows = pd.DataFrame(
        {
            "level": ["population"],
            "param": ["theta_mu"],
            "true_value": [0.2],
            "est_value": [0.21],
            "dataset_id": [0],
            "model": ["bayesflow"],
            "fit_model": ["fixed_simple"],
        }
    )

    with pytest.raises(ValueError, match="base_params"):
        recovery.summarize(
            population_rows=rows,
            individual_rows=rows.assign(level="individual", subject_id=0),
            base_params=[],
        )
