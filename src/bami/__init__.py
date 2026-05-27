"""Bayesian amortized model inference workflows.

The package is organized by responsibility:
- workflows: model-agnostic BayesFlow workflow builders
- simulators: preset model-family simulator functions
- inference: priors, transforms, checkpoints, runtime, and posthoc helpers
- recovery: parameter-recovery analysis helpers
- evaluation: reusable plotting and metric contracts for model assessment
- data_ops: tabular data transformation helpers
- data_shapes: future data-shape validation and padding helpers
"""

__all__ = [
    "workflows",
    "simulators",
    "inference",
    "recovery",
    "evaluation",
    "data_ops",
    "data_shapes",
]
