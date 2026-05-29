"""Tests for the fixed-trial simple ezDM workflow."""

import numpy as np

from fixtures_model_specs import EZDM_SPEC
from bami.inference.priors import apply_link, raw_key
from bami.simulators.ezdm import simulate_ezdm_simple
from bami.workflows import SimpleWorkflow


def _build_ezdm_model(n_trials: int = 50) -> SimpleWorkflow:
    """Build ezDM by passing its simulator into the generic workflow.

    Parameters
    ----------
    n_trials
        Fixed trial count for the simulated summary row.

    Returns
    -------
    SimpleWorkflow
        Generic simple workflow initialized with ezDM settings.
    """

    return SimpleWorkflow(
        name=EZDM_SPEC["model_name"],
        param_names=["v", "a", "t0"],
        priors=EZDM_SPEC["priors"],
        simulator=simulate_ezdm_simple,
        observation="aggregate",
        simulator_kwargs={"s": EZDM_SPEC["scaling"]},
        data_width=len(EZDM_SPEC["summary_contract"]["order"]),
        n_trials=n_trials,
        summary_dim=4,
        n_coupling_layers=2,
    )


def test_ezdm_fixed_simple_simulator_outputs_summary_features():
    """Fixed simple simulation should output pc, mrt, and vrt."""

    np.random.seed(2026)
    model = _build_ezdm_model()
    sim = model.workflow.simulate(8)
    data = sim["data"]

    assert data.shape == (8, 1, 3)
    assert np.all(data[:, :, 0] > 0)
    assert np.all(data[:, :, 0] < 1)
    assert np.all(data[:, :, 1] > 0)
    assert np.all(data[:, :, 2] > 0)


def test_ezdm_fixed_simple_infers_raw_parameter_keys():
    """The workflow should infer raw v/a/t0 keys from the priors."""

    model = _build_ezdm_model()
    adapter_text = str(model.workflow.adapter)

    assert raw_key("v") in adapter_text
    assert raw_key("a") in adapter_text
    assert raw_key("t0") in adapter_text


def test_ezdm_fixed_simple_transform_returns_public_parameters():
    """Posterior transforms should add public-scale ezDM parameters."""

    model = _build_ezdm_model()
    samples = {
        raw_key("v"): np.array([[-0.1, 0.1]]),
        raw_key("a"): np.array([[np.log(0.12), np.log(0.16)]]),
        raw_key("t0"): np.array([[np.log(0.25), np.log(0.35)]]),
    }

    out = model.convert_posterior(samples)

    expected_v = apply_link(samples[raw_key("v")], EZDM_SPEC["priors"]["v"]["link"])
    assert np.array_equal(out["v"], expected_v)
    assert np.all(out["a"] > 0)
    assert np.all(out["t0"] > 0)


def test_ezdm_fixed_simple_uses_generic_simple_workflow():
    """ezDM should use the generic simple workflow without an ezDM preset."""

    model = _build_ezdm_model()

    assert isinstance(model, SimpleWorkflow)
    assert model.workflow.workflow_family == "fixed_simple"
    assert model.workflow.model_name == "ezDM"


def test_ezdm_workflow_simulator_expansion_matches_preset_summary():
    """Workflow simulator expansion should match the preset pc/mrt/vrt order."""

    params = {"v": 0.25, "a": 1.4, "t0": 0.28}

    np.random.seed(2026)
    model = _build_ezdm_model(n_trials=50)
    from_workflow = model._simulator_fn(
        **params,
        n_trials=50,
        rng=np.random,
        **model.simulator_kwargs,
    )

    np.random.seed(2026)
    from_simulator = simulate_ezdm_simple(
        **params,
        n_trials=50,
        s=EZDM_SPEC["scaling"],
        rng=np.random,
    )

    assert from_workflow.shape == (3,)
    assert np.allclose(from_workflow, from_simulator)
