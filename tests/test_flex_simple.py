"""Tests for the flexible-response simple model."""

import numpy as np

from fixtures_model_specs import M3_SPEC, m3_activation
from bami.simulators.m3 import simulate_m3_custom
from bami.workflows import SimpleWorkflow


def test_flex_simple_simulator_adds_total_response_feature():
    """Flex simple data should include counts plus total responses."""

    np.random.seed(2026)
    model = SimpleWorkflow(
        name=M3_SPEC["model_name"],
        param_names=["a", "c", "ra", "rc"],
        priors=M3_SPEC["priors"],
        simulator=simulate_m3_custom,
        observation="aggregate",
        simulator_kwargs={
            "activation_fn": m3_activation,
            "n_options": M3_SPEC["n_options"],
            "rule": M3_SPEC["rule"],
        },
        data_width=len(M3_SPEC["activation_contract"]["order"]),
        n_trials=None,
        n_trials_range=(5, 9),
        include_trial_feature=True,
    )

    sim = model.workflow.simulate(10)
    data = sim["data"]
    totals = data[:, :, -1]
    count_sums = data[:, :, :5].sum(axis=-1)

    assert data.shape == (10, 1, 6)
    assert np.all(totals >= 5)
    assert np.all(totals < 9)
    assert np.array_equal(totals, count_sums)
