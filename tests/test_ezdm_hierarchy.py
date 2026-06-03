"""Tests for generic hierarchical ezDM workflows."""

import numpy as np

from fixtures_model_specs import EZDM_SPEC
from bami.inputs import aggregate_summary
from bami.inference.priors import (
    mu_raw_key,
)
from bami.simulators.ezdm import simulate_ezdm_simple
from bami.workflows import HierarchicalWorkflow


def _build_ezdm_hierarchy(**design_kwargs) -> HierarchicalWorkflow:
    """Build a small generic ezDM hierarchy for smoke tests.

    Parameters
    ----------
    **design_kwargs
        Fixed or flexible subject/trial settings passed to
        ``HierarchicalWorkflow``.

    Returns
    -------
    HierarchicalWorkflow
        Generic hierarchical workflow using the ezDM summary simulator.
    """

    return HierarchicalWorkflow(
        name=EZDM_SPEC["model_name"],
        priors=EZDM_SPEC["priors"],
        simulator=simulate_ezdm_simple,
        observation="aggregate",
        simulator_kwargs={"s": EZDM_SPEC["scaling"]},
        data_width=len(EZDM_SPEC["summary_contract"]["order"]),
        summary_dim=4,
        n_coupling_layers=2,
        **design_kwargs,
    )


def test_ezdm_fixed_hierarchy_simulates_subject_summary_rows():
    """Fixed ezDM hierarchy should produce subject rows with pc/mrt/vrt."""

    np.random.seed(2026)
    model = _build_ezdm_hierarchy(n_subjects=3, n_trials=20)
    sim = model.workflow.simulate(5)

    assert sim["data"].shape == (5, 3, 3)
    assert np.all(sim["data"][:, :, 0] > 0)
    assert np.all(sim["data"][:, :, 0] < 1)
    assert model.workflow.workflow_family == "fixed_hierarchical"


def test_ezdm_flex_hierarchy_uses_input_format_n_feature_and_mask():
    """Flex ezDM hierarchy should append scaled trial count and active mask."""

    np.random.seed(2026)
    model = _build_ezdm_hierarchy(
        n_subjects_range=(2, 5),
        n_trials_range=(10, 15),
        input_format=aggregate_summary(n_range=(10, 14)),
    )
    sim = model.workflow.simulate(6)
    n_subjects = sim["n_subjects"].reshape(-1)

    assert sim["data"].shape == (6, 4, 5)
    assert np.all(sim["data"][:, :, 3] >= -1.0)
    assert np.all(sim["data"][:, :, 3] <= 1.0)
    assert np.all(n_subjects >= 2)
    assert np.all(n_subjects < 5)
    for dataset_id, n_subj in enumerate(n_subjects):
        active = sim["data"][dataset_id, :, -1] > 0.5
        assert int(active.sum()) == int(n_subj)
    assert model.workflow.workflow_family == "flex_hierarchical"


def test_ezdm_hierarchy_uses_generic_prior_helpers():
    """Generic workflow should infer ezDM hierarchy prior pieces."""

    np.random.seed(2026)
    model = _build_ezdm_hierarchy(n_subjects=3, n_trials=20)
    group_params = model._draw_independent_group_prior(np.random)
    subject_params = model._draw_independent_subject_params(group_params, np.random)

    assert model.param_names == [mu_raw_key("v"), mu_raw_key("a"), mu_raw_key("t0")]
    assert {"v_mu", "a_mu", "t0_mu"}.issubset(group_params)
    assert {"v_sigma", "a_sigma", "t0_sigma"}.issubset(group_params)
    assert {"v", "a", "t0"} == set(subject_params)
