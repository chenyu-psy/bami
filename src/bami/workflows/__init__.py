"""Model-agnostic workflow building blocks.

The package exposes small workflow classes. Model families provide simulator
functions and prior settings from outside this package so the workflow layer can
remain reusable.
"""

from .contracts import validate_observation
from .hierarchical import HierarchicalWorkflow
from .simple import SimpleWorkflow

__all__ = [
    "HierarchicalWorkflow",
    "SimpleWorkflow",
    "validate_observation",
]
