"""Validation helpers for workflow observation choices."""

from __future__ import annotations

VALID_OBSERVATIONS = {"aggregate", "trial"}


def validate_observation(observation: str | None, workflow_name: str) -> str:
    """Validate an explicit workflow observation contract.

    Args:
        observation: Candidate observation contract.
        workflow_name: Name used in error messages, for example
            ``"SimpleWorkflow"``.

    Returns:
        str: Validated observation contract. Supported values are
            ``"aggregate"`` and ``"trial"``.
    """

    if observation is None:
        raise ValueError(
            f"{workflow_name} requires observation='aggregate' or "
            "observation='trial'."
        )
    checked = str(observation).strip().lower()
    if checked not in VALID_OBSERVATIONS:
        raise ValueError(
            "observation must be either 'aggregate' or 'trial', "
            f"not {observation!r}."
        )
    return checked
