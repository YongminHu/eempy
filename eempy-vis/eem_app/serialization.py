"""
JSON-safe serialization helpers for Dash dcc.Store payloads.

Dash stores must be JSON-serializable. These helpers convert common scientific Python
objects (NumPy arrays, pandas DataFrames) into JSON-safe structures, and back.

Conventions used in this codebase
---------------------------------
- NumPy arrays: nested lists; NaN encoded as None.
- DataFrames: represented as [columns] + values (a header row + body rows).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


Jsonable = Union[None, bool, int, float, str, List["Jsonable"], Dict[str, "Jsonable"]]


def ndarray_to_jsonable(arr: Any) -> Jsonable:
    """Convert an array-like to nested lists with NaN encoded as None.

    Parameters
    ----------
    arr : array-like
        Input array. Can be a NumPy array or anything convertible via ``np.asarray``.

    Returns
    -------
    jsonable : Jsonable
        Nested list representation suitable for JSON serialization.
    """
    a = np.asarray(arr)
    if a.ndim == 0:
        x = a.item()
        if isinstance(x, float) and np.isnan(x):
            return None
        return x  # type: ignore[return-value]
    # Convert to Python lists, then replace NaNs with None recursively
    lst = a.tolist()

    def _replace(x):
        if isinstance(x, list):
            return [_replace(v) for v in x]
        if isinstance(x, float) and np.isnan(x):
            return None
        return x

    return _replace(lst)


def jsonable_to_ndarray(obj: Any, dtype: Any = float) -> np.ndarray:
    """Convert a JSON-safe nested list back into a NumPy array.

    Parameters
    ----------
    obj : Any
        JSON-safe representation produced by ``ndarray_to_jsonable``.
    dtype : Any, default float
        Target dtype.

    Returns
    -------
    arr : np.ndarray
        NumPy array, with None converted back to NaN.
    """
    def _replace(x):
        if isinstance(x, list):
            return [_replace(v) for v in x]
        if x is None:
            return np.nan
        return x

    return np.asarray(_replace(obj), dtype=dtype)


def df_to_header_rows(df: pd.DataFrame) -> List[List[Any]]:
    """Serialize a pandas DataFrame as ``[columns] + values``.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to serialize.

    Returns
    -------
    header_rows : list of list
        A list where the first row is column names, followed by data rows.
        Index is NOT included; store it separately if needed.
    """
    return [df.columns.tolist()] + df.values.tolist()


def header_rows_to_df(data: Sequence[Sequence[Any]], index: Optional[Sequence[Any]] = None) -> pd.DataFrame:
    """Deserialize ``[columns] + values`` into a pandas DataFrame.

    Parameters
    ----------
    data : sequence of sequences
        Stored representation where ``data[0]`` are column names and ``data[1:]`` are rows.
    index : sequence, optional
        Index to attach to the returned DataFrame.

    Returns
    -------
    df : pandas.DataFrame
        Reconstructed DataFrame.
    """
    if data is None:
        raise ValueError("data is None")
    if len(data) == 0:
        return pd.DataFrame(index=index)
    cols = list(data[0])
    rows = list(data[1:]) if len(data) > 1 else []
    return pd.DataFrame(rows, columns=cols, index=index)


def eem_dataset_to_store(
    *,
    eem_stack: np.ndarray,
    ex_range: np.ndarray,
    em_range: np.ndarray,
    index: Optional[Sequence[Any]] = None,
    ref: Optional[pd.DataFrame] = None,
    cluster: Optional[Sequence[Any]] = None,
) -> Dict[str, Any]:
    """Serialize the minimal EEMDataset fields for dcc.Store.

    Parameters
    ----------
    eem_stack : np.ndarray
        EEM stack with shape ``(n_samples, n_ex, n_em)``.
    ex_range : np.ndarray
        Excitation axis.
    em_range : np.ndarray
        Emission axis.
    index : sequence, optional
        Sample identifiers.
    ref : pandas.DataFrame, optional
        Reference/metadata table aligned with samples.
    cluster : sequence, optional
        Cluster labels aligned with samples.

    Returns
    -------
    data : dict
        JSON-safe dict that mirrors the convention used in the app.
    """
    return {
        "eem_stack": ndarray_to_jsonable(eem_stack),
        "ex_range": np.asarray(ex_range).tolist(),
        "em_range": np.asarray(em_range).tolist(),
        "index": list(index) if index is not None else None,
        "ref": df_to_header_rows(ref) if ref is not None else None,
        "cluster": list(cluster) if cluster is not None else None,
    }


def eem_dataset_from_store(
    data: Dict[str, Any],
    *,
    dataset_factory: Optional[Callable[..., Any]] = None,
) -> Any:
    """Deserialize EEMDataset-like data from store.

    Parameters
    ----------
    data : dict
        Dict produced by ``eem_dataset_to_store`` (or equivalent in the legacy app).
    dataset_factory : callable, optional
        If provided, called as ``dataset_factory(eem_stack=..., ex_range=..., em_range=..., ref=..., index=..., cluster=...)``.
        If not provided, returns a plain dict with NumPy/pandas objects.

    Returns
    -------
    eem_dataset : object or dict
        Dataset object if ``dataset_factory`` is provided; otherwise a dict.
    """
    eem_stack = jsonable_to_ndarray(data.get("eem_stack"))
    ex_range = np.asarray(data.get("ex_range", []), dtype=float)
    em_range = np.asarray(data.get("em_range", []), dtype=float)
    index = data.get("index", None)
    ref_data = data.get("ref", None)
    ref = header_rows_to_df(ref_data, index=index) if ref_data is not None else None
    cluster = data.get("cluster", None)

    if dataset_factory is not None:
        return dataset_factory(
            eem_stack=eem_stack,
            ex_range=ex_range,
            em_range=em_range,
            ref=ref,
            index=index,
            cluster=cluster,
        )

    return {
        "eem_stack": eem_stack,
        "ex_range": ex_range,
        "em_range": em_range,
        "index": index,
        "ref": ref,
        "cluster": cluster,
    }


def model_payload_to_store(
    *,
    components: np.ndarray,
    score: pd.DataFrame,
    fmax: pd.DataFrame,
    index: Optional[Sequence[Any]] = None,
    ref: Optional[pd.DataFrame] = None,
    fitting_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Serialize a fitted decomposition model payload for dcc.Store.

    This matches the schema used in the legacy Dash app for ``parafac-models`` and ``nmf-models``.

    Parameters
    ----------
    components : np.ndarray
        Component EEMs with shape ``(n_components, n_ex, n_em)``.
    score : pandas.DataFrame
        Sample scores (columns are components).
    fmax : pandas.DataFrame
        Component amplitudes table (often NNLS-Fmax).
    index : sequence, optional
        Sample identifiers aligned with ``score``/``fmax``.
    ref : pandas.DataFrame, optional
        Reference/metadata aligned with samples.
    fitting_params : dict, optional
        Any additional fitting parameters to store (must be JSON-serializable).

    Returns
    -------
    payload : dict
        JSON-safe model payload.
    """
    index = list(index) if index is not None else list(score.index) if score is not None else None
    return {
        "components": ndarray_to_jsonable(components),
        "score": df_to_header_rows(score) if score is not None else None,
        "Fmax": df_to_header_rows(fmax) if fmax is not None else None,
        "index": index,
        "ref": df_to_header_rows(ref) if ref is not None else None,
        "fitting_params": fitting_params or {},
    }


def model_payload_from_store(
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    """Deserialize a model payload from store into NumPy/pandas objects.

    Parameters
    ----------
    payload : dict
        Dict produced by ``model_payload_to_store`` (or equivalent in the legacy app).

    Returns
    -------
    out : dict
        Dict with keys ``components``, ``score``, ``fmax``, ``index``, ``ref``, ``fitting_params``.
    """
    index = payload.get("index", None)
    components = jsonable_to_ndarray(payload.get("components"))
    score_data = payload.get("score", None)
    fmax_data = payload.get("Fmax", None)
    ref_data = payload.get("ref", None)

    score = header_rows_to_df(score_data, index=index) if score_data is not None else None
    fmax = header_rows_to_df(fmax_data, index=index) if fmax_data is not None else None
    ref = header_rows_to_df(ref_data, index=index) if ref_data is not None else None

    return {
        "components": components,
        "score": score,
        "fmax": fmax,
        "index": index,
        "ref": ref,
        "fitting_params": payload.get("fitting_params", {}) or {},
    }
