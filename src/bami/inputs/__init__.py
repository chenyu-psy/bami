"""Workflow input-format helpers.

Use these functions when aggregate workflow inputs need to carry trial-count
information explicitly.
"""

from .formats import InputFormat, aggregate_summary, counts, proportions

__all__ = [
    "InputFormat",
    "aggregate_summary",
    "counts",
    "proportions",
]
