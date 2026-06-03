"""Tests for generic simple workflow builders."""

import warnings

import numpy as np
import pytest

from bami.inputs import aggregate_summary, counts, proportions
from bami.inference.priors import log_sigma_key, mu_raw_key, raw_key
from bami.workflows import (
    HierarchicalWorkflow,
    SimpleWorkflow,
)
from bami.workflows.hierarchical import (
    MaskedEquivariantSetEncoder,
    MaskedNestedSummary,
)
from bami.workflows.simple import _suppress_singleton_softmax_warning


def _toy_simulator(theta: float, n_trials: int) -> np.ndarray:
    """Simulate one toy fixed-simple data row.

    Parameters
    ----------
    theta
        Public toy parameter.
    n_trials
        Fixed trial count supplied by the workflow.
    Returns
    -------
    numpy.ndarray
        One row with a parameter-derived feature and trial-count feature.
    """

    return np.array([theta, n_trials], dtype=np.float32)


def _toy_summary_simulator(theta: float, n_trials: int) -> np.ndarray:
    """Simulate one aggregate row that does not contain trial count.

    Parameters
    ----------
    theta
        Public toy parameter.
    n_trials
        Trial count accepted for workflow compatibility.
    Returns
    -------
    numpy.ndarray
        One aggregate summary feature.
    """

    return np.array([theta], dtype=np.float32)


def _toy_trial_simulator(theta: float, n_trials: int) -> np.ndarray:
    """Simulate one toy trial-data array.

    Parameters
    ----------
    theta
        Public toy parameter.
    n_trials
        Number of trial rows to return.
    Returns
    -------
    numpy.ndarray
        Trial rows with one feature per trial.
    """

    return np.full((n_trials, 1), theta, dtype=np.float32)


def _to_numpy(value) -> np.ndarray:
    """Convert a backend tensor to NumPy for test comparisons."""

    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _trigger_singleton_softmax_warning() -> None:
    """Run the Keras operation that warns for singleton attention axes."""

    import keras

    keras.ops.softmax(keras.ops.zeros((1, 4, 4, 1)), axis=3)


def test_singleton_softmax_warning_filter_is_scoped():
    """Aggregate warning filter should hide only the known DeepSet warning."""

    with warnings.catch_warnings(record=True) as caught_disabled:
        warnings.simplefilter("always")
        with _suppress_singleton_softmax_warning(False):
            _trigger_singleton_softmax_warning()

    assert any("softmax over axis" in str(item.message) for item in caught_disabled)

    with warnings.catch_warnings(record=True) as caught_enabled:
        warnings.simplefilter("always")
        with _suppress_singleton_softmax_warning(True):
            _trigger_singleton_softmax_warning()
            warnings.warn("ordinary warning", UserWarning)

    messages = [str(item.message) for item in caught_enabled]
    assert messages == ["ordinary warning"]


def test_input_format_log_range_scales_trial_count():
    """Log-range n encoding should map range endpoints to -1 and 1."""

    input_format = aggregate_summary(n_range=(50, 200))

    assert np.isclose(input_format.transform_n(50), -1.0)
    assert np.isclose(input_format.transform_n(200), 1.0)
    assert input_format.to_dict()["kind"] == "aggregate_summary"


def test_input_format_presets_encode_expected_widths():
    """Input presets should preserve rows and append n only when requested."""

    row = np.array([0.2, 0.4], dtype=np.float32)

    aggregate = aggregate_summary(n_range=(10, 20)).encode(row, 10)
    proportion_row = proportions(n_range=(10, 20)).encode(row, 20)
    count_row = counts().encode(row, 10)
    counts_with_n = counts(add_n=True, n_range=(10, 20)).encode(row, 10)

    assert aggregate.shape == (3,)
    assert proportion_row.shape == (3,)
    assert count_row.shape == (2,)
    assert counts_with_n.shape == (3,)


def _build_toy_workflow() -> SimpleWorkflow:
    """Build a minimal simulator-first workflow for fast unit tests.

    Returns
    -------
    SimpleWorkflow
        Workflow using one identity-linked parameter and two data features.
    """

    return SimpleWorkflow(
        name="toy",
        param_names=["theta"],
        priors={"theta": {"mean": 0.0, "sd": 1.0, "link": "identity"}},
        simulator=_toy_simulator,
        observation="aggregate",
        data_width=2,
        n_trials=7,
        summary_dim=4,
        n_coupling_layers=2,
    )


def test_fixed_simple_workflow_simulates_fixed_width_rows():
    """Generic workflow simulation should produce batch x one-row x width data."""

    np.random.seed(2026)
    model = _build_toy_workflow()
    sim = model.workflow.simulate(5)

    assert sim["data"].shape == (5, 1, 2)
    assert np.all(sim["data"][:, :, 1] == 7)
    assert model.workflow.workflow_family == "fixed_simple"
    assert model.workflow.workflow_level == "simple"


def test_fixed_simple_workflow_infers_raw_prior_keys():
    """Inference variables should come from dictionary-valued prior specs."""

    model = _build_toy_workflow()
    adapter_text = str(model.workflow.adapter)

    assert raw_key("theta") in adapter_text


def test_fixed_simple_workflow_transforms_public_parameters():
    """Raw posterior samples should gain public parameter keys."""

    model = _build_toy_workflow()
    samples = {raw_key("theta"): np.array([[-1.0, 1.0]])}

    out = model.convert_posterior(samples)

    assert np.array_equal(out["theta"], samples[raw_key("theta")])


def test_fixed_simple_workflow_reuses_validation_data():
    """Validation data should be reused when the requested size is unchanged."""

    model = _build_toy_workflow()
    first = model._resolve_validation_data(3)
    second = model._resolve_validation_data(3)
    third = model._resolve_validation_data(4)

    assert first is second
    assert first is not third
    assert third["data"].shape == (4, 1, 2)


def test_simple_workflow_accepts_contract():
    """Contract input should keep simulator settings grouped when useful."""

    contract = {
        "name": "toy",
        "param_names": ["theta"],
        "priors": {"theta": {"mean": 0.0, "sd": 1.0, "link": "identity"}},
        "simulator": _toy_simulator,
        "data_width": 2,
    }

    model = SimpleWorkflow(
        contract=contract,
        observation="aggregate",
        n_trials=7,
        summary_dim=4,
        n_coupling_layers=2,
    )

    assert model.workflow.model_name == "toy"


def test_simple_workflow_requires_explicit_observation():
    """Observation contract should be a visible modeling decision."""

    with pytest.raises(ValueError, match="observation='aggregate'"):
        SimpleWorkflow(
            name="toy",
            param_names=["theta"],
            priors={"theta": {"mean": 0.0, "sd": 1.0, "link": "identity"}},
            simulator=_toy_simulator,
            data_width=2,
            n_trials=7,
            summary_dim=4,
            n_coupling_layers=2,
        )


def test_simple_workflow_rejects_unknown_observation():
    """Only aggregate and trial observation contracts should be accepted."""

    with pytest.raises(ValueError, match="aggregate.*trial"):
        SimpleWorkflow(
            name="toy",
            param_names=["theta"],
            priors={"theta": {"mean": 0.0, "sd": 1.0, "link": "identity"}},
            simulator=_toy_simulator,
            observation="summary",
            data_width=2,
            n_trials=7,
            summary_dim=4,
            n_coupling_layers=2,
        )


def test_simple_workflow_trial_observation_outputs_trial_rows():
    """Trial observation should use trials as the set dimension."""

    np.random.seed(2026)
    model = SimpleWorkflow(
        name="toy",
        param_names=["theta"],
        priors={"theta": {"mean": 0.0, "sd": 1.0, "link": "identity"}},
        simulator=_toy_trial_simulator,
        observation="trial",
        obs_names=["response"],
        n_trials=6,
        summary_dim=4,
        n_coupling_layers=2,
    )

    sim = model.workflow.simulate(3)

    assert sim["data"].shape == (3, 6, 1)
    assert model.workflow.observation == "trial"
    assert model.workflow.obs_names == ["response"]


def test_simple_workflow_flex_trial_observation_adds_active_mask():
    """Flexible trial observations should be padded and marked with a mask."""

    np.random.seed(2026)
    model = SimpleWorkflow(
        name="toy",
        param_names=["theta"],
        priors={"theta": {"mean": 0.0, "sd": 1.0, "link": "identity"}},
        simulator=_toy_trial_simulator,
        observation="trial",
        obs_names=["response"],
        n_trials=None,
        n_trials_range=(4, 8),
        summary_dim=4,
        n_coupling_layers=2,
    )

    sim = model.workflow.simulate(6)
    mask = sim["data"][:, :, 1]

    assert sim["data"].shape == (6, 7, 2)
    assert np.all((mask == 0) | (mask == 1))
    assert np.all(mask.sum(axis=1) >= 4)
    assert np.all(mask.sum(axis=1) < 8)
    assert model.workflow.obs_names == ["response", "active_trial"]


def test_simple_workflow_trial_shape_error_mentions_observation():
    """Trial simulator shape errors should identify the selected contract."""

    def bad_trial_simulator(theta: float, n_trials: int) -> np.ndarray:
        """Return an invalid aggregate-style row for a trial workflow."""

        del theta, n_trials
        return np.array([1.0, 2.0], dtype=np.float32)

    model = SimpleWorkflow(
        name="toy",
        param_names=["theta"],
        priors={"theta": {"mean": 0.0, "sd": 1.0, "link": "identity"}},
        simulator=bad_trial_simulator,
        observation="trial",
        obs_names=["response"],
        n_trials=6,
        summary_dim=4,
        n_coupling_layers=2,
    )

    with pytest.raises(ValueError, match="observation='trial'"):
        model.workflow.simulate(1)


def test_simple_workflow_draws_trial_counts_from_range():
    """SimpleWorkflow should support flexible trial-count simulation."""

    np.random.seed(2026)
    model = SimpleWorkflow(
        name="toy",
        param_names=["theta"],
        priors={"theta": {"mean": 0.0, "sd": 1.0, "link": "identity"}},
        simulator=_toy_simulator,
        observation="aggregate",
        data_width=2,
        n_trials=None,
        n_trials_range=(5, 9),
        summary_dim=4,
        n_coupling_layers=2,
    )

    sim = model.workflow.simulate(12)
    trial_counts = sim["data"][:, :, 1]

    assert sim["data"].shape == (12, 1, 2)
    assert np.all(trial_counts >= 5)
    assert np.all(trial_counts < 9)
    assert model.workflow.workflow_family == "flex_simple"
    assert model.workflow.trial_design == "flex"


def test_simple_workflow_input_format_encodes_flex_summary_n():
    """Flex simple aggregate rows should include encoded n through input_format."""

    np.random.seed(2026)
    model = SimpleWorkflow(
        name="toy",
        param_names=["theta"],
        priors={"theta": {"mean": 0.0, "sd": 1.0, "link": "identity"}},
        simulator=_toy_summary_simulator,
        observation="aggregate",
        data_width=1,
        n_trials=None,
        n_trials_range=(5, 9),
        input_format=aggregate_summary(n_range=(5, 8)),
        summary_dim=4,
        n_coupling_layers=2,
    )

    sim = model.workflow.simulate(12)
    encoded_n = sim["data"][:, :, -1]
    manual_data, _ = model._prepare_observed_counts(
        np.array([[0.5, 5.0]], dtype=np.float32)
    )

    assert sim["data"].shape == (12, 1, 2)
    assert np.all(encoded_n >= -1.0)
    assert np.all(encoded_n <= 1.0)
    assert manual_data.shape == (1, 1, 2)
    assert np.isclose(manual_data[0, 0, -1], -1.0)


def test_simple_workflow_rejects_ambiguous_trial_design():
    """Trial design should be exactly one of fixed count or count range."""

    with pytest.raises(ValueError, match="either n_trials or n_trials_range"):
        SimpleWorkflow(
            name="toy",
            param_names=["theta"],
            priors={"theta": {"mean": 0.0, "sd": 1.0, "link": "identity"}},
            simulator=_toy_simulator,
            observation="aggregate",
            data_width=2,
            n_trials=7,
            n_trials_range=(5, 9),
            summary_dim=4,
            n_coupling_layers=2,
        )


def test_hierarchical_workflow_simulates_fixed_group_rows():
    """Fixed hierarchical workflow should keep subject-row count fixed."""

    model = HierarchicalWorkflow(
        name="toy_hier",
        priors={"theta": {"mean": "normal(0, 0.1)", "sd": 1.0, "link": "identity"}},
        simulator=_toy_simulator,
        observation="aggregate",
        data_width=2,
        n_subjects=3,
        n_trials=7,
        summary_dim=4,
        n_coupling_layers=2,
    )

    sim = model.workflow.simulate(4)

    assert sim["data"].shape == (4, 3, 2)
    assert np.all(sim["data"][:, :, 1] == 7)
    assert model.workflow.workflow_level == "hierarchical"
    assert model.workflow.workflow_family == "fixed_hierarchical"
    assert model.workflow.observation == "aggregate"


def test_hierarchical_trial_observation_outputs_nested_trial_rows():
    """Fixed trial hierarchy should preserve subject and trial dimensions."""

    np.random.seed(2026)
    model = HierarchicalWorkflow(
        name="toy_hier",
        priors={"theta": {"mean": "normal(0, 0.1)", "sd": 1.0, "link": "identity"}},
        simulator=_toy_trial_simulator,
        observation="trial",
        obs_names=["response"],
        n_subjects=3,
        n_trials=7,
        summary_dim=4,
        n_coupling_layers=2,
    )

    sim = model.workflow.simulate(4)

    assert sim["data"].shape == (4, 3, 7, 1)
    assert model.workflow.observation == "trial"
    assert model.workflow.obs_names == ["response"]


def test_hierarchical_trial_summary_accepts_bayesflow_stage_arg():
    """Nested trial summary should expose BayesFlow's metric-call interface."""

    np.random.seed(2026)
    model = HierarchicalWorkflow(
        name="toy_hier",
        priors={"theta": {"mean": "normal(0, 0.1)", "sd": 1.0, "link": "identity"}},
        simulator=_toy_trial_simulator,
        observation="trial",
        obs_names=["response"],
        n_subjects=3,
        n_trials=7,
        summary_dim=4,
        n_coupling_layers=2,
    )
    sim = model.workflow.simulate(4)

    metrics = model.summary_network.compute_metrics(
        sim["data"],
        stage="training",
    )

    assert "outputs" in metrics
    assert tuple(metrics["outputs"].shape) == (4, 4)
    assert isinstance(model.summary_network, MaskedNestedSummary)
    assert model.summary_network.has_trial_mask is False
    assert model.summary_network.include_count_features is False


def test_masked_nested_summary_ignores_padded_trial_values():
    """Inactive trial rows should not change the group summary."""

    network = MaskedNestedSummary(summary_dim=4, has_trial_mask=True)
    data = np.array(
        [
            [
                [[0.1, 1.0], [0.2, 1.0], [9.0, 0.0]],
                [[0.3, 1.0], [8.0, 0.0], [7.0, 0.0]],
            ],
            [
                [[0.4, 1.0], [0.5, 1.0], [6.0, 0.0]],
                [[0.6, 1.0], [0.7, 1.0], [5.0, 0.0]],
            ],
        ],
        dtype=np.float32,
    )
    changed = data.copy()
    changed[changed[:, :, :, 1] == 0, 0] = 1000.0

    first = _to_numpy(network(data, training=False))
    second = _to_numpy(network(changed, training=False))

    assert np.allclose(first, second, atol=1e-6)


def test_masked_nested_summary_ignores_padded_subject_values():
    """Subjects with no active trials should not change the group summary."""

    network = MaskedNestedSummary(summary_dim=4, has_trial_mask=True)
    data = np.array(
        [
            [
                [[0.1, 1.0], [0.2, 1.0], [0.3, 1.0]],
                [[9.0, 0.0], [8.0, 0.0], [7.0, 0.0]],
                [[0.4, 1.0], [0.5, 1.0], [0.6, 1.0]],
            ],
        ],
        dtype=np.float32,
    )
    changed = data.copy()
    changed[:, 1, :, 0] = 1000.0

    first = _to_numpy(network(data, training=False))
    second = _to_numpy(network(changed, training=False))

    assert np.allclose(first, second, atol=1e-6)


def test_masked_nested_summary_accepts_fixed_no_mask_inputs():
    """Fixed trial hierarchy inputs should not need an active-trial column."""

    network = MaskedNestedSummary(summary_dim=4, has_trial_mask=False)
    data = np.zeros((2, 3, 5, 1), dtype=np.float32)
    metrics = network.compute_metrics(data, stage="training")

    assert "outputs" in metrics
    assert tuple(metrics["outputs"].shape) == (2, 4)


def test_masked_nested_summary_accepts_flex_mask_inputs():
    """Flex trial hierarchy inputs should accept active-trial mask columns."""

    network = MaskedNestedSummary(summary_dim=4, has_trial_mask=True)
    data = np.zeros((2, 3, 5, 2), dtype=np.float32)
    data[:, :, :3, 1] = 1.0

    metrics = network.compute_metrics(data, stage="training")

    assert "outputs" in metrics
    assert tuple(metrics["outputs"].shape) == (2, 4)


def test_masked_nested_summary_uses_context_blocks():
    """Nested summaries should use generic masked equivariant set encoders."""

    network = MaskedNestedSummary(summary_dim=4, has_trial_mask=True)

    assert isinstance(network.trial_encoder, MaskedEquivariantSetEncoder)
    assert isinstance(network.subject_encoder, MaskedEquivariantSetEncoder)
    assert len(network.trial_encoder.element_layers) >= 2
    assert len(network.trial_encoder.context_layers) >= 2
    assert len(network.trial_encoder.post_layers) >= 2
    assert len(network.subject_encoder.element_layers) >= 2
    assert len(network.subject_encoder.context_layers) >= 2
    assert len(network.subject_encoder.post_layers) >= 2


def test_masked_nested_summary_is_trial_permutation_invariant():
    """Reordering trials within subjects should not change the group summary."""

    network = MaskedNestedSummary(summary_dim=4, has_trial_mask=True)
    data = np.array(
        [
            [
                [[0.1, 1.0], [0.2, 1.0], [9.0, 0.0], [0.3, 1.0]],
                [[0.4, 1.0], [8.0, 0.0], [0.5, 1.0], [7.0, 0.0]],
            ],
        ],
        dtype=np.float32,
    )
    permuted = data[:, :, [3, 0, 2, 1], :]

    first = _to_numpy(network(data, training=False))
    second = _to_numpy(network(permuted, training=False))

    assert np.allclose(first, second, atol=1e-6)


def test_masked_nested_summary_is_subject_permutation_invariant():
    """Reordering subjects within a dataset should not change the summary."""

    network = MaskedNestedSummary(summary_dim=4, has_trial_mask=True)
    data = np.array(
        [
            [
                [[0.1, 1.0], [0.2, 1.0], [0.3, 1.0]],
                [[0.4, 1.0], [0.5, 1.0], [0.6, 1.0]],
                [[9.0, 0.0], [8.0, 0.0], [7.0, 0.0]],
            ],
        ],
        dtype=np.float32,
    )
    permuted = data[:, [1, 2, 0], :, :]

    first = _to_numpy(network(data, training=False))
    second = _to_numpy(network(permuted, training=False))

    assert np.allclose(first, second, atol=1e-6)


def test_masked_set_encoder_count_features_track_active_count():
    """Count-aware encoders should expose active set size after pooling."""

    summary = np.zeros((2, 4), dtype=np.float32)
    mask = np.array(
        [
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )

    encoded = _to_numpy(
        MaskedEquivariantSetEncoder._append_count_features(summary, mask)
    )

    assert encoded.shape == (2, 6)
    assert np.allclose(encoded[:, :4], 0.0)
    assert encoded[0, 4] == 0.5
    assert encoded[1, 4] == 1.0
    assert encoded[0, 5] < encoded[1, 5]
    assert encoded[1, 5] == 1.0


def test_hierarchical_flex_trial_observation_adds_trial_masks():
    """Flex trial hierarchy should pad trials and mark active trial rows."""

    np.random.seed(2026)
    model = HierarchicalWorkflow(
        name="toy_hier",
        priors={"theta": {"mean": "normal(0, 0.1)", "sd": 1.0, "link": "identity"}},
        simulator=_toy_trial_simulator,
        observation="trial",
        obs_names=["response"],
        n_subjects_range=(2, 5),
        n_trials_range=(4, 8),
        summary_dim=4,
        n_coupling_layers=2,
    )

    sim = model.workflow.simulate(6)
    mask = sim["data"][:, :, :, 1]
    n_subjects = sim["n_subjects"].reshape(-1)

    assert sim["data"].shape == (6, 4, 7, 2)
    assert np.all((mask == 0) | (mask == 1))
    assert np.all(n_subjects >= 2)
    assert np.all(n_subjects < 5)
    assert model.workflow.obs_names == ["response", "active_trial"]
    assert isinstance(model.summary_network, MaskedNestedSummary)
    assert model.summary_network.has_trial_mask is True
    assert model.summary_network.include_count_features is True
    assert model.summary_network.trial_encoder.include_count_features is True
    assert model.summary_network.subject_encoder.include_count_features is True


def test_hierarchical_workflow_pads_flexible_subject_rows():
    """Flexible subject designs should include padded rows and an active mask."""

    np.random.seed(2026)
    model = HierarchicalWorkflow(
        name="toy_hier",
        priors={"theta": {"mean": "normal(0, 0.1)", "sd": 1.0, "link": "identity"}},
        simulator=_toy_simulator,
        observation="aggregate",
        data_width=2,
        n_subjects_range=(1, 5),
        n_trials_range=(5, 9),
        include_trial_feature=True,
        summary_dim=4,
        n_coupling_layers=2,
    )

    sim = model.workflow.simulate(12)
    n_subjects = sim["n_subjects"].reshape(-1)

    assert sim["data"].shape == (12, 4, 4)
    assert np.all(n_subjects >= 1)
    assert np.all(n_subjects < 5)
    for dataset_id, n_subj in enumerate(n_subjects):
        active = sim["data"][dataset_id, :, -1] > 0.5
        assert int(active.sum()) == int(n_subj)
    assert model.workflow.workflow_family == "flex_hierarchical"


def test_hierarchical_workflow_input_format_encodes_subject_summary_n():
    """Flex hierarchy aggregate rows should include encoded n plus mask."""

    np.random.seed(2026)
    model = HierarchicalWorkflow(
        name="toy_hier",
        priors={"theta": {"mean": "normal(0, 0.1)", "sd": 1.0, "link": "identity"}},
        simulator=_toy_summary_simulator,
        observation="aggregate",
        data_width=1,
        n_subjects_range=(2, 5),
        n_trials_range=(10, 15),
        input_format=aggregate_summary(n_range=(10, 14)),
        summary_dim=4,
        n_coupling_layers=2,
    )

    sim = model.workflow.simulate(6)
    manual_data, _ = model._prepare_observed_counts(
        np.array([[0.2, 10.0], [0.4, 14.0]], dtype=np.float32)
    )

    assert sim["data"].shape == (6, 4, 3)
    assert model.workflow.input_format_metadata["kind"] == "aggregate_summary"
    assert np.all(sim["data"][:, :, 1] >= -1.0)
    assert np.all(sim["data"][:, :, 1] <= 1.0)
    assert manual_data.shape == (1, 4, 3)
    assert np.isclose(manual_data[0, 0, 1], -1.0)
    assert np.isclose(manual_data[0, 1, 1], 1.0)


def test_hierarchical_workflow_draws_default_group_and_subject_values():
    """Generic hierarchy workflow should use the shared prior format directly."""

    rng = np.random.default_rng(2026)
    priors = {
        "theta": {"mean": "normal(0, 0.1)", "sd": 0.2, "link": "identity"},
        "scale": {"mean": 0.0, "sd": "uniform(0.1, 0.3)", "link": "log"},
        "bias": 0.0,
    }
    model = HierarchicalWorkflow(
        name="toy_hier",
        priors=priors,
        simulator=_toy_simulator,
        observation="aggregate",
        data_width=2,
        n_subjects=2,
        n_trials=5,
        summary_dim=4,
        n_coupling_layers=2,
    )

    group_params = model._draw_independent_group_prior(rng)
    subject_params = model._draw_independent_subject_params(group_params, rng)

    assert model.param_names == [mu_raw_key("theta"), log_sigma_key("scale")]
    assert "theta_mu" in group_params
    assert "scale_sigma" in group_params
    assert set(subject_params) == {"theta", "scale", "bias"}
    assert subject_params["scale"] > 0
    assert subject_params["bias"] == 0.0
