"""brms metric interface placeholders for shared evaluation outputs.

These functions define stable interfaces and output schema for future brms
integration. They currently raise ``NotImplementedError`` intentionally.
"""

from __future__ import annotations

import pandas as pd


def brms_pop_recovery(*args, **kwargs) -> pd.DataFrame:
    """Return brms population recovery in the shared long-format contract.

    Parameters
    ----------
    *args, **kwargs
        Placeholder arguments for future brms extraction inputs.

    Returns
    -------
    pd.DataFrame
        Recovery contract table with ``level='population'``.

    Raises
    ------
    NotImplementedError
        brms integration is planned but not implemented in this revision.
    """

    raise NotImplementedError("BRMS population recovery is not implemented yet.")


def brms_ind_recovery(*args, **kwargs) -> pd.DataFrame:
    """Return brms individual recovery in the shared long-format contract.

    Parameters
    ----------
    *args, **kwargs
        Placeholder arguments for future brms extraction inputs.

    Returns
    -------
    pd.DataFrame
        Recovery contract table with ``level='individual'``.

    Raises
    ------
    NotImplementedError
        brms integration is planned but not implemented in this revision.
    """

    raise NotImplementedError("BRMS individual recovery is not implemented yet.")


def brms_calibration(*args, **kwargs) -> pd.DataFrame:
    """Return brms calibration metric table in the shared diagnostic format.

    Parameters
    ----------
    *args, **kwargs
        Placeholder arguments for future brms extraction inputs.

    Returns
    -------
    pd.DataFrame
        Diagnostic table with columns ``param``, ``metric``, and ``value``.

    Raises
    ------
    NotImplementedError
        brms integration is planned but not implemented in this revision.
    """

    raise NotImplementedError("BRMS calibration diagnostics are not implemented yet.")


def brms_coverage(*args, **kwargs) -> pd.DataFrame:
    """Return brms coverage metric table in the shared diagnostic format.

    Parameters
    ----------
    *args, **kwargs
        Placeholder arguments for future brms extraction inputs.

    Returns
    -------
    pd.DataFrame
        Diagnostic table with columns ``param``, ``metric``, and ``value``.

    Raises
    ------
    NotImplementedError
        brms integration is planned but not implemented in this revision.
    """

    raise NotImplementedError("BRMS coverage diagnostics are not implemented yet.")


def brms_zscore(*args, **kwargs) -> pd.DataFrame:
    """Return brms z-score/contraction metrics in shared diagnostic format.

    Parameters
    ----------
    *args, **kwargs
        Placeholder arguments for future brms extraction inputs.

    Returns
    -------
    pd.DataFrame
        Diagnostic table with columns ``param``, ``metric``, and ``value``.

    Raises
    ------
    NotImplementedError
        brms integration is planned but not implemented in this revision.
    """

    raise NotImplementedError("BRMS z-score diagnostics are not implemented yet.")
