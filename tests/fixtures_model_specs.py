"""Small model specs used by package tests.

These fixtures keep package tests independent from analysis-project notebooks
while preserving the public prior and observation contracts needed to exercise
workflow construction.
"""

M3_SPEC = {
    "model_name": "M3",
    "n_options": [1, 3, 1, 3, 4],
    "rule": "softmax",
    "priors": {
        "a": {"mean": "normal(0.8, 0.5)", "sd": 0.2, "link": "log"},
        "c": {"mean": "normal(0.8, 0.5)", "sd": 0.2, "link": "log"},
        "ra": {"mean": "logistic(0, 0.75)", "sd": 0.5, "link": "logit"},
        "rc": {"mean": "logistic(0, 0.75)", "sd": 0.5, "link": "logit"},
        "b": 0,
    },
    "activation_contract": {
        "names": ["correct", "other", "dist", "other_dist", "new"],
        "order": ["correct", "other", "dist", "other_dist", "new"],
    },
}


def m3_activation(a, c, ra, rc, b=0):
    """Return standard M3 activation scores for test workflows.

    Parameters
    ----------
    a, c, ra, rc, b
        Public-scale M3 parameters.

    Returns
    -------
    list[float]
        Activation scores in ``correct``, ``other``, ``dist``, ``other_dist``,
        and ``new`` order.
    """

    return [
        a + c + b,
        a + b,
        ra * a + rc * c + b,
        ra * a + b,
        b,
    ]

EZDM_SPEC = {
    "model_name": "ezDM",
    "priors": {
        "v": {"mean": "normal(0, 0.6)", "sd": 0.15, "link": "log"},
        "a": {"mean": "normal(0.2, 0.4)", "sd": 0.15, "link": "log"},
        "t0": {"mean": "logistic(-2, 0.5)", "sd": 0.15, "link": "logit"},
    },
    "summary_contract": {
        "names": ["pc", "mrt", "vrt"],
        "order": ["pc", "mrt", "vrt"],
    },
    "scaling": 1,
}

SDM_SPEC = {
    "model_name": "SDM",
    "priors": {
        "c": {"mean": "normal(1, 0.35)", "sd": 0.15, "link": "log"},
        "kappa": {"mean": "normal(1.2, 0.35)", "sd": 0.15, "link": "log"},
    },
    "trial_contract": {
        "names": ["error"],
        "order": ["error"],
        "unit": "signed degrees",
        "range": "[-180, 180)",
    },
}
