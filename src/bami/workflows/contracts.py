"""Validation helpers for model-agnostic workflow contracts.

Workflow contracts describe the small amount of model-specific information a
generic fixed, flexible, or hierarchical workflow needs. The contract keeps
simulator functions visible while avoiding hard-coded dependencies on M3, SDM,
or ezDM classes.
"""

from __future__ import annotations

from collections.abc import Mapping

REQUIRED_CONTRACT_FIELDS = (
    "name",
    "param_names",
    "priors",
    "simulator",
    "data_width",
)

VALID_OBSERVATIONS = {"aggregate", "trial"}


def validate_workflow_contract(contract: Mapping) -> dict:
    """Validate and normalize a workflow contract.

    Args:
        contract: Mapping with model-specific workflow information. Required
            fields are ``name``, ``param_names``, ``priors``, ``simulator``, and
            ``data_width``.

    Returns:
        dict: A shallow normalized copy of the contract.
    """

    if not isinstance(contract, Mapping):
        raise TypeError("workflow contract must be a mapping.")

    missing = [field for field in REQUIRED_CONTRACT_FIELDS if field not in contract]
    if missing:
        raise ValueError(f"workflow contract is missing required fields: {missing}")

    out = dict(contract)
    out["name"] = _check_name(out["name"])
    out["param_names"] = _check_param_names(out["param_names"])
    out["data_width"] = _check_data_width(out["data_width"])
    _check_priors(out["priors"])
    _check_callable(out["simulator"], "simulator")

    return out


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


def _check_name(name) -> str:
    """Validate the model name in a workflow contract.

    Args:
        name:
            Candidate model name.

    Returns:
        str: Non-empty model name.
    """

    checked = str(name).strip()
    if checked == "":
        raise ValueError("workflow contract name must be non-empty.")
    return checked


def _check_param_names(param_names) -> list[str]:
    """Validate parameter names in a workflow contract.

    Args:
        param_names:
            Candidate sequence of parameter names.

    Returns:
        list[str]: Non-empty parameter-name list.
    """

    if isinstance(param_names, str):
        raise ValueError("param_names must be a sequence of names, not one string.")

    try:
        checked = [str(name).strip() for name in param_names]
    except TypeError as exc:
        raise ValueError("param_names must be a sequence of names.") from exc

    if not checked or any(name == "" for name in checked):
        raise ValueError("param_names must contain at least one non-empty name.")
    return checked


def _check_priors(priors) -> None:
    """Validate that priors are mapping-like.

    Args:
        priors:
            Prior specification supplied to the workflow contract.

    Returns:
        None: Raises an error when the prior specification is not mapping-like.
    """

    if not isinstance(priors, Mapping):
        raise ValueError("priors must be a mapping.")


def _check_data_width(data_width) -> int:
    """Validate subject data row width.

    Args:
        data_width:
            Candidate number of values returned by ``simulator``.

    Returns:
        int: Positive integer row width.
    """

    checked = int(data_width)
    if checked < 1:
        raise ValueError("data_width must be at least 1.")
    return checked


def _check_callable(value, field_name: str) -> None:
    """Validate that a contract field is callable.

    Args:
        value:
            Candidate callable.
        field_name:
            Field name used in error messages.

    Returns:
        None: Raises an error when ``value`` is not callable.
    """

    if not callable(value):
        raise ValueError(f"{field_name} must be callable.")
