"""Preset model simulators for reusable workflow construction.

The functions exported here are convenience generators. Future workflows can
also accept user-supplied simulator functions with the same simple
parameter-to-data role.
"""

from .ezdm import simulate_ezdm_simple
from .m3 import prop_m3, simulate_m3_custom
from .sdm import (
    simulate_sdm_simple,
)

__all__ = [
    "prop_m3",
    "simulate_ezdm_simple",
    "simulate_m3_custom",
    "simulate_sdm_simple",
]
