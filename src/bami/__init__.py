"""Bayesian amortized model inference workflows.

The package is organized by responsibility:
- workflows: model-agnostic BayesFlow workflow builders
- inputs: input-format helpers for workflow data rows
- simulators: preset model-family simulator functions
- training: workflow fitting with shared checkpoint behavior
- inference: priors, transforms, checkpoints, runtime, and posthoc helpers
- recovery: parameter-recovery analysis helpers
- evaluation: reusable plotting and metric contracts for model assessment
- data_ops: tabular data transformation helpers
- data_shapes: future data-shape validation and padding helpers
"""

__all__ = [
    "workflows",
    "inputs",
    "simulators",
    "training",
    "inference",
    "recovery",
    "evaluation",
    "data_ops",
    "data_shapes",
]
