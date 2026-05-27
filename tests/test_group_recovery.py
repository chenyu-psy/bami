"""Tests for group-generated model-comparison recovery."""

from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from bami.recovery.group import (
    BASE_PARAMS,
    examine_group_recovery,
    prepare_flex_subject_recovery_data,
    group_model_conditions,
    generate_group_datasets,
    load_group_generated_data,
    load_group_recovery_rows,
    save_group_recovery_rows,
    summarize_edge_param_quantile_bias,
    summarize_population_recovery_diagnostics,
)


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
        for base_param in BASE_PARAMS:
            pop_truth = np.linspace(0.2, 0.8, n_datasets, dtype=np.float32)
            payload[f"{base_param}_mu"] = pop_truth
            for subj in range(n_subjects):
                payload[f"{base_param}_subj_{subj}"] = pop_truth + 0.03 * subj
        return payload


class _GroupModel:
    """Model stub exposing a group workflow."""

    def __init__(self):
        """Attach a deterministic group workflow."""

        self.workflow = _GroupWorkflow()


class _FlexModel:
    """Minimal flex model that pads fixed subject count data."""

    def counts_to_data(self, counts: np.ndarray) -> tuple[np.ndarray, dict]:
        """Convert one dataset's counts to padded flex-hierarchy data.

        Parameters
        ----------
        counts : np.ndarray
            Subject-by-response count table.

        Returns
        -------
        tuple[np.ndarray, dict]
            One flex-formatted dataset and empty metadata.
        """

        n_subjects = counts.shape[0]
        data = np.zeros((1, n_subjects, 7), dtype=np.float32)
        data[0, :, :5] = counts
        data[0, :, 5] = counts.sum(axis=1)
        data[0, :, 6] = 1.0
        return data, {}


class _FlexSummaryModel:
    """Minimal flex summary model that expects explicit trial counts."""

    data_width = 3

    class _ObsSpec:
        """Small observation spec stub with encoded n enabled."""

        add_n = True

    obs_spec = _ObsSpec()

    def counts_to_data(self, rows: np.ndarray) -> tuple[np.ndarray, dict]:
        """Convert summary rows plus n_trials to padded flex data."""

        n_subjects = rows.shape[0]
        data = np.zeros((1, n_subjects, 5), dtype=np.float32)
        data[0, :, :4] = rows
        data[0, :, 4] = 1.0
        return data, {}


def test_generate_group_datasets_uses_configurable_repetitions(tmp_path: Path):
    """Group generation should store the requested comparison dataset count.

    Parameters
    ----------
    tmp_path : Path
        Pytest temporary directory fixture.

    Returns
    -------
    None
        Asserts payload metadata and data dimensions.
    """

    out_path = generate_group_datasets(
        group_generator=_GroupModel(),
        n_reps=7,
        seed=2026,
        artifact_dir=tmp_path,
    )

    with out_path.open("rb") as file_obj:
        payload = pickle.load(file_obj)

    assert payload["n_reps"] == 7
    assert payload["n_subjects"] == 4
    assert payload["sim_data"]["data"].shape == (7, 4, 5)

    loaded = load_group_generated_data(out_path)
    assert loaded["n_reps"] == payload["n_reps"]
    assert loaded["sim_data"]["data"].shape == payload["sim_data"]["data"].shape


def test_generate_group_datasets_accepts_artifact_prefix(tmp_path: Path):
    """Group generation should support ezDM-style artifact prefixes."""

    out_path = generate_group_datasets(
        group_generator=_GroupModel(),
        n_reps=2,
        seed=2026,
        artifact_dir=tmp_path,
        artifact_prefix="22_ezDM",
    )

    assert out_path.name == "22_ezDM_group_generated_data.pkl"


def test_group_model_conditions_append_fixed_n_for_flex_summaries():
    """Flex summary workflows should receive base summaries plus n_trials."""

    sim_data = {
        "data": np.ones((2, 3, 3), dtype=np.float32),
    }

    conditions = group_model_conditions(
        fit_name="flex_hierarchy",
        fit_model=_FlexSummaryModel(),
        sim_data=sim_data,
        n_trials=100,
    )

    assert conditions["data"].shape == (2, 3, 5)
    assert np.allclose(conditions["data"][:, :, 3], 100)
    assert np.allclose(conditions["data"][:, :, 4], 1)


def test_flex_test_data_accepts_column_vector_subject_truth():
    """Flex adapter should flatten subject truth arrays from generated data."""

    n_datasets = 3
    n_subjects = 2
    sim_data = {
        "data": np.ones((n_datasets, n_subjects, 5), dtype=np.float32),
    }
    for base_param in BASE_PARAMS:
        for subject_id in range(n_subjects):
            values = np.arange(n_datasets, dtype=np.float32).reshape(-1, 1)
            sim_data[f"{base_param}_subj_{subject_id}"] = values + subject_id

    out = prepare_flex_subject_recovery_data(_FlexModel(), sim_data)

    assert out["data"].shape == (n_datasets, n_subjects, 7)
    assert out["raw_counts"].shape == (n_datasets, n_subjects, 5)
    assert np.allclose(out["raw_counts"], sim_data["data"])
    assert out["a_subj"].shape == (n_datasets, n_subjects)
    assert np.allclose(out["a_subj"][:, 0], [0.0, 1.0, 2.0])
    assert np.allclose(out["a_subj"][:, 1], [1.0, 2.0, 3.0])


def test_flex_test_data_accepts_array_subject_truth():
    """Flex adapter should prefer current array-form subject truth.

    Returns
    -------
    None
        Asserts ``a_subj`` array truth is copied into the padded flex payload.
    """

    n_datasets = 3
    n_subjects = 2
    sim_data = {
        "data": np.ones((n_datasets, n_subjects, 5), dtype=np.float32),
    }
    for base_param in BASE_PARAMS:
        sim_data[f"{base_param}_subj"] = np.asarray(
            [
                [0.1, 0.2],
                [0.3, 0.4],
                [0.5, 0.6],
            ],
            dtype=np.float32,
        )

    out = prepare_flex_subject_recovery_data(_FlexModel(), sim_data)

    assert out["data"].shape == (n_datasets, n_subjects, 7)
    assert out["raw_counts"].shape == (n_datasets, n_subjects, 5)
    assert np.allclose(out["raw_counts"], sim_data["data"])
    assert out["a_subj"].shape == (n_datasets, n_subjects)
    assert np.allclose(out["a_subj"], sim_data["a_subj"])


def test_population_recovery_diagnostics_include_scale_and_bias():
    """Population diagnostics should expose compression-relevant metrics."""

    rows = pd.DataFrame(
        {
            "fit_model": ["flex_hierarchy"] * 4,
            "param": ["ra_mu"] * 4,
            "true_value": [0.1, 0.3, 0.7, 0.9],
            "est_value": [0.2, 0.35, 0.65, 0.8],
        }
    )

    summary = summarize_population_recovery_diagnostics(rows)

    assert summary.shape[0] == 1
    assert summary["fit_model"].iloc[0] == "flex_hierarchy"
    assert summary["param"].iloc[0] == "ra_mu"
    assert summary["slope"].iloc[0] < 1.0
    assert {"ccc", "r", "bias", "mae", "rmse", "n"}.issubset(summary.columns)


def test_edge_param_quantile_bias_detects_center_compression():
    """Quantile bias should show low-end overestimation and high-end underestimation."""

    rows = pd.DataFrame(
        {
            "fit_model": ["flex_hierarchy"] * 8,
            "param": ["rc_mu"] * 8,
            "true_value": [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9],
            "est_value": [0.2, 0.28, 0.34, 0.42, 0.58, 0.66, 0.72, 0.8],
        }
    )

    bias = summarize_edge_param_quantile_bias(rows, n_quantiles=4)

    assert bias.shape[0] == 4
    assert bias["bias"].iloc[0] > 0.0
    assert bias["bias"].iloc[-1] < 0.0


def test_save_group_recovery_rows_writes_step22_artifacts(tmp_path: Path):
    """Recovery-row saving should write the artifacts consumed by Step 23.

    Parameters
    ----------
    tmp_path : Path
        Pytest temporary directory fixture.

    Returns
    -------
    None
        Asserts pickle and CSV paths are written and loadable.
    """

    population_rows = pd.DataFrame(
        {
            "level": ["population"],
            "param": ["a_mu"],
            "true_value": [0.2],
            "est_value": [0.21],
            "dataset_id": [0],
            "model": ["bayesflow"],
            "fit_model": ["fixed_simple"],
        }
    )
    individual_rows = pd.DataFrame(
        {
            "level": ["individual"],
            "param": ["a"],
            "true_value": [0.2],
            "est_value": [0.21],
            "dataset_id": [0],
            "subject_id": [0],
            "model": ["bayesflow"],
            "fit_model": ["fixed_simple"],
        }
    )

    paths = save_group_recovery_rows(
        population_rows=population_rows,
        individual_rows=individual_rows,
        artifact_dir=tmp_path,
    )
    loaded_population, loaded_individual = load_group_recovery_rows(
        population_rows_path=paths["population_rows_pkl"],
        individual_rows_path=paths["individual_rows_pkl"],
    )

    assert (
        paths["population_rows_pkl"].name == "11_M3_group_population_recovery_rows.pkl"
    )
    assert (
        paths["individual_rows_pkl"].name == "11_M3_group_individual_recovery_rows.pkl"
    )
    assert paths["population_rows_csv"].exists()
    assert paths["individual_rows_csv"].exists()
    pd.testing.assert_frame_equal(loaded_population, population_rows)
    pd.testing.assert_frame_equal(loaded_individual, individual_rows)


def test_save_group_recovery_rows_accepts_artifact_prefix(tmp_path: Path):
    """Recovery-row saving should support 22_ezDM artifact names."""

    population_rows = pd.DataFrame(
        {
            "level": ["population"],
            "param": ["v_mu"],
            "true_value": [0.2],
            "est_value": [0.21],
            "dataset_id": [0],
            "model": ["bayesflow"],
            "fit_model": ["fixed_simple"],
        }
    )
    individual_rows = pd.DataFrame(
        {
            "level": ["individual"],
            "param": ["v"],
            "true_value": [0.2],
            "est_value": [0.21],
            "dataset_id": [0],
            "subject_id": [0],
            "model": ["bayesflow"],
            "fit_model": ["fixed_simple"],
        }
    )

    paths = save_group_recovery_rows(
        population_rows=population_rows,
        individual_rows=individual_rows,
        artifact_dir=tmp_path,
        artifact_prefix="22_ezDM",
    )

    assert paths["population_rows_pkl"].name == (
        "22_ezDM_group_population_recovery_rows.pkl"
    )
    assert paths["individual_rows_pkl"].name == (
        "22_ezDM_group_individual_recovery_rows.pkl"
    )


def test_examine_group_recovery_writes_population_and_individual_outputs(
    tmp_path: Path,
):
    """Group recovery should summarize all four fit models at both levels.

    Parameters
    ----------
    tmp_path : Path
        Pytest temporary directory fixture.

    Returns
    -------
    None
        Asserts row counts and per-dataset individual correlations.
    """

    n_datasets = 3
    n_subjects = 4
    fit_models = ["fixed_simple", "flex_simple", "fixed_hierarchy", "flex_hierarchy"]
    population_rows = []
    individual_rows = []

    for base_param in BASE_PARAMS:
        pop_truth = np.linspace(0.2, 0.8, n_datasets, dtype=np.float32)
        for fit_name in fit_models:
            for dataset_id, true_val in enumerate(pop_truth):
                population_rows.append(
                    {
                        "level": "population",
                        "param": f"{base_param}_mu",
                        "true_value": float(true_val),
                        "est_value": float(true_val + 0.01),
                        "dataset_id": dataset_id,
                        "model": "bayesflow",
                        "fit_model": fit_name,
                    }
                )

                for subject_id in range(n_subjects):
                    subj_truth = float(true_val + 0.04 * subject_id)
                    individual_rows.append(
                        {
                            "level": "individual",
                            "param": base_param,
                            "true_value": subj_truth,
                            "est_value": subj_truth + 0.01,
                            "dataset_id": dataset_id,
                            "subject_id": subject_id,
                            "model": "bayesflow",
                            "fit_model": fit_name,
                        }
                    )

    row_paths = save_group_recovery_rows(
        population_rows=pd.DataFrame(population_rows),
        individual_rows=pd.DataFrame(individual_rows),
        artifact_dir=tmp_path,
    )

    (
        population_rows,
        population_summary,
        individual_rows,
        individual_correlations,
        output_paths,
    ) = examine_group_recovery(
        population_rows_path=row_paths["population_rows_pkl"],
        individual_rows_path=row_paths["individual_rows_pkl"],
        result_dir=tmp_path,
    )

    assert population_rows.shape[0] == len(fit_models) * n_datasets * len(BASE_PARAMS)
    assert population_summary.shape[0] == len(fit_models) * len(BASE_PARAMS)
    assert "ccc" in population_summary.columns
    assert "r" not in population_summary.columns
    assert np.all(population_summary["ccc"] < 1.0)
    assert individual_rows.shape[0] == len(fit_models) * n_datasets * n_subjects * len(
        BASE_PARAMS
    )
    assert individual_correlations.shape[0] == len(fit_models) * n_datasets * len(
        BASE_PARAMS
    )
    assert "ccc" not in individual_correlations.columns
    assert np.allclose(individual_correlations["r"], 1.0)
    for output_path in output_paths.values():
        assert output_path.exists()
    assert "population_diagnostics" in output_paths
    assert "edge_param_quantile_bias" in output_paths
