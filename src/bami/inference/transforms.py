"""Posterior-sample transforms for project workflows.

These helpers expose the raw-to-public parameter transforms from the inference
namespace so model workflows do not need to import from legacy model package.
"""

from bami.inference.priors import (
    transform_hierarchical_samples,
    transform_simple_samples,
)

__all__ = ["transform_hierarchical_samples", "transform_simple_samples"]
