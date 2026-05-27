"""Model-agnostic workflow building blocks.

The package exposes small workflow classes and contract validators. Model
families provide simulator functions and prior settings from outside this
package so the workflow layer can remain reusable.
"""

from .contracts import validate_observation, validate_workflow_contract
from .hierarchical import HierarchicalWorkflow
from .obs_spec import ObsSpec
from .simple import SimpleWorkflow

__all__ = [
    "HierarchicalWorkflow",
    "ObsSpec",
    "SimpleWorkflow",
    "validate_observation",
    "validate_workflow_contract",
]
