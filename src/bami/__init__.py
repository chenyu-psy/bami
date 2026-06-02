"""Bayesian amortized model inference workflows.

The package is organized by responsibility:
workflows: model-agnostic BayesFlow workflow builders
inputs: input-format helpers for workflow data rows
simulators: preset model-family simulator functions
inference: priors, transforms, saved workflows, and runtime helpers
evaluation: reusable plotting and metric contracts for model assessment
data_ops: tabular data transformation helpers
data_shapes: future data-shape validation and padding helpers
utils: general user-facing utility helpers
"""

import os

# PyTorch checks this flag when the MPS backend is initialized. Set it before
# BayesFlow or Torch can be imported so unsupported MPS ops can fall back to CPU.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from bami.utils import posterior_to_dataframe

__all__ = [
    "workflows",
    "inputs",
    "simulators",
    "inference",
    "evaluation",
    "data_ops",
    "data_shapes",
    "utils",
    "posterior_to_dataframe",
]
