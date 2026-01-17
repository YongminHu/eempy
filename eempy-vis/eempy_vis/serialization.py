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
from eempy.eem_processing import EEMDataset, PARAFAC, EEMNMF


def eem_dataset_to_serializable(
        eem_dataset: EEMDataset,
) -> Dict[str, Any]:
    """Serialize the minimal EEMDataset fields for dcc.Store.

    Parameters
    ----------
    eem_dataset : EEMDataset
        EEM dataset.

    Returns
    -------
    data : dict
        JSON-safe dict that mirrors the convention used in the app.
    """
    return {
        "eem_stack": [
            [[None if np.isnan(x) else x for x in subsublist] for subsublist in sublist]
            for sublist in eem_dataset.eem_stack.tolist()
        ],
        "ex_range": eem_dataset.ex_range.tolist(),
        "em_range": eem_dataset.em_range.tolist(),
        "index": eem_dataset.index,
        "ref": [eem_dataset.ref.columns.tolist()] + eem_dataset.ref.values.tolist() if eem_dataset.ref is not None
        else None,
        "cluster": None,
    }


def eem_dataset_from_serializable(
        data_dict: Dict[str, Any],
) -> EEMDataset:
    """Deserialize EEMDataset-like data from store.

    Parameters
    ----------
    data : dict
        Dict produced by ``eem_dataset_to_store``.

    Returns
    -------
    eem_dataset : EEMDataset
        The converted EEM dataset.
    """

    return EEMDataset(
        eem_stack=np.array(
            [[[np.nan if x is None else x for x in subsublist] for subsublist in sublist] for sublist
             in data_dict['eem_stack']]),
        ex_range=np.array(data_dict['ex_range']),
        em_range=np.array(data_dict['em_range']),
        index=data_dict['index'],
        ref=pd.DataFrame(data_dict['ref'][1:], columns=data_dict['ref'][0],
                         index=data_dict['index'])
        if data_dict['ref'] is not None else None,
    )

