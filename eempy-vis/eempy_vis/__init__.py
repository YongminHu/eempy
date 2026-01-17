"""
Legacy alias module.

Use ``eempy_vis.__init__`` (``eempy_vis/__init__.py``) for package initialization.
This module re-exports common symbols for convenience.
"""

from .config import (EXTERNAL_STYLESHEETS, COLORS, MARKER_SHAPES, HELP_ICON_STYLE, ROW_FLEX_STYLE,
                     SECTION_TITLE_STYLE, CARD_STYLE, SMALL_TEXT_STYLE)
from .ids import IDS, ID_TO_CONST, CONST_TO_ID
from .serialization import (
    ndarray_to_jsonable,
    jsonable_to_ndarray,
    df_to_header_rows,
    header_rows_to_df,
    eem_dataset_to_serializable,
    eem_dataset_from_serializable,
)
