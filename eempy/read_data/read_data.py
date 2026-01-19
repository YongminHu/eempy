"""
Functions for importing files
Author: Yongmin Hu (yongminhu@outlook.com)
Last update: 2026-01
"""

import os
import re
import warnings

import numpy as np
import pandas as pd
import json
from datetime import datetime
from typing import Union, Tuple, List, Optional
from ..eem_processing import eem_interpolation, eem_cutting, process_eem_stack, EEMDataset
from ..utils import dichotomy_search
from scipy.interpolate import interp1d


def read_eem(
        file_path: str,
        index_pos: Union[Tuple[int, int], List[int], None] = None,
        data_format: str = "default",
        as_timestamp: bool = False,
        timestamp_format: Optional[str] = None,
        delimiter: Optional[str] = None,
        file_first_row: str = "ex",
):
    """
    Import EEM from file (tabular format).

    Parameters
    ----------
    file_path : str
        Filepath to the EEM file.
    index_pos : None or tuple/list with two elements
        Start/end positions of index in filename (1-based start, end inclusive as in original code intent).
        Example: (4, 13) extracts basename[3:13].
    data_format : str, {'default'}
        Format of the EEM file. By passing ``'default'``, the following tabular format is supported:
        - The first row contains excitation wavelengths (Ex).
        - The top left may be blank or non-numeric; any non-numeric tokens are ignored.
        - The first column of subsequent rows contains emission wavelengths (Em).
        - The remaining cells are fluorescence intensities for each (Ex, Em) pair.

        Schematic layout::

            <blank or label>   Ex_1   Ex_2   Ex_3   ...   Ex_n
            Em_1               I_11   I_12   I_13   ...   I_1N
            Em_2               I_21   I_22   I_23   ...   I_2N
            ...
            Em_m               I_m1   I_m2   I_m3   ...   I_mn

        Where I_nm is the intensity at excitation Ex_n and emission Em_m.

    as_timestamp : bool
        Whether to parse extracted index as datetime.
    timestamp_format : str
        Datetime strptime format if as_timestamp is True.
        Rules can be seen on https://docs.python.org/3/library/datetime.html#format-codes
    delimiter : Optional[str] = None
        Field delimiter. If None (default), split on arbitrary whitespace (tabs/spaces). If your file uses a specific
        delimiter (e.g., comma or semicolon), pass ``delimiter=','`` or ``delimiter=';'``, etc.
    file_first_row : {"ex","em"}
        Whether the first row contains Ex or Em wavelengths.

    Returns
    -------
    intensity : np.ndarray (2d)
        EEM matrix with shape (n_ex, n_em). Rows correspond to excitation wavelengths, columns to emission
        wavelengths. The smallest excitation & emission wavelengths correspond to intensity[-1, 0],
        with excitation wavelength increases from bottom to top and emission wavelength increases from left to
        right.
    ex_range : np.ndarray (1d)
        Sorted excitation wavelengths (ascending).
    em_range : np.ndarray (1d)
        Sorted emission wavelengths (ascending).
    index : str | datetime | None
        Extracted index (optionally parsed as datetime).
    """
    # ---- validate layout selection ----
    file_first_row = file_first_row.lower()
    if file_first_row not in ("ex", "em"):
        raise ValueError('file_first_row must be "ex" or "em".')

    # ---- index parsing ----
    if index_pos:
        index = os.path.basename(file_path)[index_pos[0] - 1: index_pos[1]]
        if as_timestamp:
            if not timestamp_format:
                raise ValueError("timestamp_format must be provided when as_timestamp=True.")
            index = datetime.strptime(index, timestamp_format)
    else:
        index = None

    if data_format != "default":
        raise ValueError('Unsupported data_format. Currently only "default" is supported.')

    def _split(line: str) -> List[str]:
        parts = line.split(delimiter) if delimiter is not None else line.split()
        return [p.strip() for p in parts if p.strip() != ""]

    def _parse_header_wavelengths(line: str) -> np.ndarray:
        # Parse wavelengths across the first row (usually after a blank/label cell)
        if delimiter is None:
            nums = re.findall(r"\d+(?:\.\d+)?", line)
            return np.array(list(map(float, nums)), dtype=float)
        else:
            fields = _split(line)
            if len(fields) >= 2:
                fields = fields[1:]  # drop first cell (blank/label)
            out = []
            for f in fields:
                try:
                    out.append(float(f))
                except ValueError:
                    pass
            return np.array(out, dtype=float)

    # ---- read file ----
    with open(file_path, "r") as of:
        firstline = of.readline()
        header_vals = _parse_header_wavelengths(firstline)
        if header_vals.size == 0:
            raise ValueError("Could not parse wavelength header from the first line.")

        idx_vals: List[float] = []
        rows: List[List[float]] = []

        for line in of:
            parts = _split(line)
            if not parts:
                continue
            try:
                idx = float(parts[0])
            except ValueError:
                continue  # skip non-data lines

            try:
                intens = list(map(float, parts[1:]))
            except ValueError:
                raise ValueError(f"Non-numeric intensity value encountered in line: {line!r}")

            idx_vals.append(idx)
            rows.append(intens)

    if not rows:
        raise ValueError("No EEM data rows were found in the file.")

    data = np.array(rows, dtype=float)  # (n_rows, n_cols)
    idx_arr = np.array(idx_vals, dtype=float)
    hdr_arr = np.array(header_vals, dtype=float)

    if data.ndim != 2:
        raise ValueError(f"Parsed data is not 2D (got shape {data.shape}).")
    if data.shape[0] != idx_arr.size:
        raise ValueError("Row count mismatch between parsed index column and data matrix.")
    if data.shape[1] != hdr_arr.size:
        raise ValueError(
            f"Header/data dimension mismatch: header has {hdr_arr.size} values, "
            f"but data rows have {data.shape[1]} columns."
        )

    # ---- interpret file axes and convert to a single internal layout: (em rows, ex cols) ----
    # Current parsing produces: rows correspond to file_first_col, cols correspond to file_first_row
    if file_first_row == "ex":
        ex_raw = hdr_arr
        em_raw = idx_arr
        data_em_ex = data  # rows=em, cols=ex (because first_col is em)
    elif file_first_row == "em":
        em_raw = hdr_arr
        ex_raw = idx_arr
        data_em_ex = data.T  # convert from rows=ex, cols=em -> rows=em, cols=ex

    # ---- sort wavelengths ascending and reorder ----
    ex_order = np.argsort(ex_raw)
    em_order = np.argsort(em_raw)

    if np.unique(ex_raw).size != ex_raw.size:
        warnings.warn("Duplicate excitation wavelengths detected; sorting will not resolve duplicates.")
    if np.unique(em_raw).size != em_raw.size:
        warnings.warn("Duplicate emission wavelengths detected; sorting will not resolve duplicates.")

    ex_range = ex_raw[ex_order]  # ascending
    em_range = em_raw[em_order]  # ascending

    data_em_ex_sorted = data_em_ex[np.ix_(em_order, ex_order)]  # (n_em, n_ex) with em/ex ascending

    # ---- standard output: intensity is (n_ex, n_em) with em along columns ----
    # Transpose to (ex rows, em cols)
    intensity = data_em_ex_sorted.T  # (n_ex, n_em)

    # ---- enforce "matrix has origin at lower-left" in array indexing ----
    intensity = intensity[::-1, :]

    return intensity, ex_range, em_range, index


def read_eem_dataset(folder_path: str, mandatory_keywords=None, optional_keywords=None, data_format: str = 'default',
                     index_pos: Union[Tuple, List, None] = None, as_timestamp=False, timestamp_format=None,
                     delimiter=None, file_first_row="ex",
                     custom_filename_list: Union[Tuple, List, None] = None, wavelength_alignment=False,
                     interpolation_method: str = 'linear'):
    """
    Import EEM dataset from files.

    Parameters
    ----------
    folder_path: str
        The path to the folder containing EEMs.
    mandatory_keywords: list of str
        Keywords for searching EEM files whose filenames contain all the mandatory keywords.
    optional_keywords: list of str
        Keywords for searching EEM files whose filenames contain any of the optional keywords.
    data_format: str, {'default'}
        Format of the EEM file. By passing ``'default'``, the following tabular format is supported:
        - The first row contains excitation wavelengths (Ex).
        - The top left may be blank or non-numeric; any non-numeric tokens are ignored.
        - The first column of subsequent rows contains emission wavelengths (Em).
        - The remaining cells are fluorescence intensities for each (Ex, Em) pair.

        Schematic layout::

            <blank or label>   Ex_1   Ex_2   Ex_3   ...   Ex_n
            Em_1               I_11   I_12   I_13   ...   I_1N
            Em_2               I_21   I_22   I_23   ...   I_2N
            ...
            Em_m               I_m1   I_m2   I_m3   ...   I_mn

        Where I_nm is the intensity at excitation Ex_n and emission Em_m.
    index_pos: str
        The starting and ending positions of index in filenames. For example, if you want to read the index "2024_01_01"
        from the file with the name "EEM_2024_01_01_PEM.dat", a tuple (4, 13) should be passed to this parameter.
    as_timestamp: bool
        Whether to read the index as timestamps.
    timestamp_format: str
        Datetime strptime format if as_timestamp is True.
        Rules can be seen on https://docs.python.org/3/library/datetime.html#format-codes
    delimiter : Optional[str] = None
        Field delimiter. If None (default), split on arbitrary whitespace (tabs/spaces). If your file uses a specific
        delimiter (e.g., comma or semicolon), pass ``delimiter=','`` or ``delimiter=';'``, etc.
    file_first_row : {"ex","em"}
        Whether the first row contains Ex or Em wavelengths.
    custom_filename_list: list or None
        If a list is passed, only the EEM files whose filenames are specified in the list will be imported.
    wavelength_alignment: bool
        Align the ex/em ranges of the EEMs. This is useful if the EEMs are measured with different ex/em ranges.
        Note that ex/em will be aligned according to the ex/em ranges with the smallest intervals among all the
        imported EEMs.
    interpolation_method: str
        The interpolation method used for aligning ex/em. It is only useful if wavelength_alignment=True.

    Returns
    -------
    eem_stack: np.ndarray (3d)
        EEM stack with shape (n_sample, n_ex, n_em). For each EEM with shape (n_ex, n_em), rows correspond to
        excitation wavelengths, columns to emission wavelengths. The smallest excitation & emission wavelengths
        correspond to intensity[-1, 0], with excitation wavelength increases from bottom to top and emission
        wavelength increases from left to right.
    ex_range: np.ndarray (1d)
        Sorted excitation wavelengths (ascending).
    em_range: np.ndarray (1d)
        Sorted emission wavelengths (ascending).
    indexes: str | datetime | None
        Extracted index (optionally parsed as datetime).
    """
    # ---- resolve file list ----
    filename_list = get_filelist(folder_path, mandatory_keywords, optional_keywords) \
        if custom_filename_list is None else list(custom_filename_list)

    # Fast exits / clearer errors
    if not filename_list:
        raise ValueError("No EEM files found (filename_list is empty).")

    # Prebind for speed/readability
    join_path = os.path.join
    _read_eem = read_eem

    intensity_list = []
    indexes = []
    ex_range_list = []
    em_range_list = []

    # For alignment: find overlapping wavelength window across all samples
    if wavelength_alignment:
        ex_min = -np.inf
        ex_max = np.inf
        em_min = -np.inf
        em_max = np.inf

    # ---- read all EEMs (single pass) ----
    for fname in filename_list:
        path = join_path(folder_path, fname)

        intensity, ex_range, em_range, index = _read_eem(
            path,
            data_format=data_format,
            index_pos=index_pos,
            as_timestamp=as_timestamp,
            timestamp_format=timestamp_format,
            delimiter=delimiter,
            file_first_row=file_first_row,
        )

        indexes.append(index)
        intensity_list.append(intensity)
        ex_range_list.append(ex_range)
        em_range_list.append(em_range)

        if wavelength_alignment:
            # Overlap across all samples (intersection of ranges)
            ex_min = max(ex_min, float(np.min(ex_range)))
            ex_max = min(ex_max, float(np.max(ex_range)))
            em_min = max(em_min, float(np.min(em_range)))
            em_max = min(em_max, float(np.max(em_range)))

    # If no alignment, return immediately using the last-read ex_range/em_range (same as original behavior)
    if not wavelength_alignment:
        try:
            eem_stack = np.asarray(intensity_list)
        except ValueError as e:
            # keep the original spirit but make it deterministic
            raise ValueError(f"Check data dimension consistency across files: {filename_list}") from e
        return eem_stack, ex_range, em_range, indexes

    # ---- wavelength alignment ----
    if not (np.isfinite(ex_min) and np.isfinite(ex_max) and np.isfinite(em_min) and np.isfinite(em_max)):
        raise ValueError("Failed to determine finite overlap window for wavelength alignment.")
    if ex_min > ex_max or em_min > em_max:
        raise ValueError(
            f"No overlapping wavelength region across EEMs: "
            f"ex_min={ex_min}, ex_max={ex_max}, em_min={em_min}, em_max={em_max}"
        )

    # Choose "optimal" target grids: within the overlap window, pick the longest cut grid encountered
    # (matches original logic: uses the range with the most points after cutting to overlap)
    ex_range_opt = np.zeros(1)
    em_range_opt = np.zeros(1)

    for ex_range_i, em_range_i in zip(ex_range_list, em_range_list):
        ex_min_idx = dichotomy_search(ex_range_i, ex_min)
        ex_max_idx = dichotomy_search(ex_range_i, ex_max)
        ex_cut = ex_range_i[ex_min_idx: ex_max_idx + 1]

        em_min_idx = dichotomy_search(em_range_i, em_min)
        em_max_idx = dichotomy_search(em_range_i, em_max)
        em_cut = em_range_i[em_min_idx: em_max_idx + 1]

        if ex_cut.shape[0] >= ex_range_opt.shape[0]:
            ex_range_opt = ex_cut
        if em_cut.shape[0] >= em_range_opt.shape[0]:
            em_range_opt = em_cut

    # Interpolate each EEM onto the chosen grid
    _interp = eem_interpolation
    for i in range(len(intensity_list)):
        intensity_list[i] = _interp(
            intensity_list[i],
            ex_range_list[i],
            em_range_list[i],
            ex_range_opt,
            em_range_opt,
            method=interpolation_method
        )

    try:
        eem_stack = np.asarray(intensity_list)
    except ValueError as e:
        raise ValueError(f"Check data dimension after alignment/interpolation for files: {filename_list}") from e

    return eem_stack, ex_range_opt, em_range_opt, indexes


def read_eem_dataset_from_json(path):
    with open(path, 'r') as file:
        eem_dataset_dict = json.load(file)
    eem_dataset = EEMDataset(
        eem_stack=np.array(
            [[[np.nan if x is None else x for x in subsublist] for subsublist in sublist] for sublist
             in eem_dataset_dict['eem_stack']]),
        ex_range=np.array(eem_dataset_dict['ex_range']),
        em_range=np.array(eem_dataset_dict['em_range']),
        index=eem_dataset_dict['index'],
        ref=pd.DataFrame(eem_dataset_dict['ref'][1:], columns=eem_dataset_dict['ref'][0],
                         index=eem_dataset_dict['index'])
        if eem_dataset_dict['ref'] is not None else None,
    )
    return eem_dataset


def read_abs(file_path, index_pos: Union[Tuple, List, None] = None, data_format='default'):
    """
    Import UV absorbance data from UV absorbance file.

    Parameters
    ----------------
    file_path: str
        The filepath to the UV absorbance file
    index_pos: None or tuple with two elements
        The starting and ending positions of index in filenames. For example, if you want to read the index "2024_01_01"
        from the file with the name "EEM_2024_01_01_PEM.dat", a tuple (4, 13) should be passed to this parameter.
    data_format: str
        Format of the UV absorbance file. By passing ``data_format='default'``, the following format is supported::

            Ex_1    A_1
            Ex_2    A_2
            ...     ...
            Ex_n    A_n

        Where A_i correspond to the absorbance at wavelength Ex_i

    Returns
    ----------------
    absorbance:np.ndarray (1d)
        The UV absorbance spectra
    ex_range: np.ndarray (1d)
        The excitation wavelengths
    index: str
        The index of the Absorbance spectrum.
    """
    # ---- index parsing (preserve original slicing convention) ----
    if index_pos:
        index = os.path.basename(file_path)[index_pos[0] : index_pos[1] + 1]
    else:
        index = None

    if data_format != "default":
        warnings.warn(
            'The current version of eempy only supports reading files written in the "default" format.'
        )

    idx: List[float] = []
    data: List[float] = []

    with open(file_path, "r") as of:
        for line in of:
            parts = line.split()
            if not parts:
                continue

            # Try to parse wavelength; skip non-data lines safely
            try:
                wl = float(parts[0])
            except ValueError:
                continue

            idx.append(wl)

            # Absorbance value may be missing
            if len(parts) >= 2:
                try:
                    val = float(parts[1])
                except ValueError:
                    val = np.nan
            else:
                val = np.nan
            data.append(val)

    if len(idx) == 0:
        raise ValueError("No absorbance data rows were found in the file.")

    ex = np.asarray(idx, dtype=float)
    ab = np.asarray(data, dtype=float)

    # ---- robust to arbitrary wavelength order ----
    order = np.argsort(ex)
    ex_range = ex[order]
    absorbance = ab[order]

    if np.unique(ex_range).size != ex_range.size:
        warnings.warn("Duplicate wavelengths detected in absorbance file; data has been sorted but duplicates remain.")

    return absorbance, ex_range, index


def read_abs_dataset(folder_path, mandatory_keywords='ABS', optional_keywords=[], data_format: str = 'default',
                     index_pos: Union[Tuple, List, None] = None, custom_filename_list: Union[Tuple, List, None] = None,
                     wavelength_alignment=False, interpolation_method: str = 'linear'):
    """

    Parameters
    ----------
    folder_path: str
        The path to the folder containing absorbance files.
    mandatory_keywords: list of str
        Keywords for searching absorbance files whose filenames contain all the mandatory keywords.
    optional_keywords: list of str
        Keywords for searching absorbance files whose filenames contain any of the optional keywords.
    data_format: str
        Specify the type of absorbance data format.
    index_pos: list of int
        The starting and ending positions of index in filenames. For example, if you want to read the index "2024_01_01"
        from the file with the name "ABS_2024_01_01_PEM.dat", a tuple (4, 13) should be passed to this parameter.
    custom_filename_list: list or None
        If a list is passed, only the absorbance files whose filenames are specified in the list will be imported.
    wavelength_alignment: bool
        Align the ex range of the absorbance files. This is useful if the absorbance are measured with different ex
        range. Note that ex will be aligned according to the ex ranges with the smallest intervals among all the
        imported absorbance files.
    interpolation_method: str
        The interpolation method used for aligning ex. It is only useful if wavelength_alignment=True.

    Returns
    -------
    abs_stack: np.ndarray (2d)
        A stack of imported absorbance files.
    ex_range: np.ndarray (1d)
        The excitation wavelengths
    indexes: list or None
        The list of absorbance file indexes (if index_pos is specified).
    """
    # ---- resolve file list ----
    filename_list = get_filelist(folder_path, mandatory_keywords, optional_keywords) \
        if not custom_filename_list else list(custom_filename_list)

    if not filename_list:
        raise ValueError("No absorbance files found (filename_list is empty).")

    # Prebind for small speed/readability wins
    join_path = os.path.join
    _read_abs = read_abs

    indexes = []
    abs_list = []
    ex_range_list = []

    # Overlap window across files (only if alignment requested)
    if wavelength_alignment:
        ex_min = -np.inf
        ex_max = np.inf

    # ---- read all spectra (single pass) ----
    for fname in filename_list:
        path = join_path(folder_path, fname)
        absorbance, ex_range, index = _read_abs(path, data_format=data_format, index_pos=index_pos)

        indexes.append(index)
        abs_list.append(absorbance)
        ex_range_list.append(ex_range)

        if wavelength_alignment:
            ex_min = max(ex_min, float(np.min(ex_range)))
            ex_max = min(ex_max, float(np.max(ex_range)))

    # ---- no alignment: stack and return (same behavior as original: ex_range from last file) ----
    if not wavelength_alignment:
        try:
            abs_stack = np.asarray(abs_list)
        except ValueError as e:
            raise ValueError(f"Check absorbance length consistency across files: {filename_list}") from e
        return abs_stack, ex_range, indexes

    # ---- alignment sanity checks ----
    if not (np.isfinite(ex_min) and np.isfinite(ex_max)):
        raise ValueError("Failed to determine finite overlap window for wavelength alignment.")
    if ex_min > ex_max:
        raise ValueError(f"No overlapping wavelength region across absorbance files: ex_min={ex_min}, ex_max={ex_max}")

    # ---- pick optimal target grid (matches your original: longest cut grid within overlap) ----
    ex_range_opt = np.zeros(1)
    for ex_range_i in ex_range_list:
        ex_min_idx = dichotomy_search(ex_range_i, ex_min)
        ex_max_idx = dichotomy_search(ex_range_i, ex_max)
        ex_cut = ex_range_i[ex_min_idx: ex_max_idx + 1]
        if ex_cut.shape[0] >= ex_range_opt.shape[0]:
            ex_range_opt = ex_cut

    # ---- interpolate onto target grid ----
    for i in range(len(abs_list)):
        f = interp1d(
            ex_range_list[i],
            abs_list[i],
            kind=interpolation_method,
            fill_value='extrapolate'
        )
        abs_list[i] = f(ex_range_opt)

    try:
        abs_stack = np.asarray(abs_list)
    except ValueError as e:
        raise ValueError(f"Check absorbance dimension after alignment/interpolation for files: {filename_list}") from e

    return abs_stack, ex_range_opt, indexes


def read_reference_from_text(filepath):
    """
    Read reference data from text file. The reference data can be any 1D data (e.g., dissolved organic carbon
    concentration). This first line of the file should be a header, and then each following line contains one datapoint,
    without any separators other than line breaks.
    For example::

            DOC (mg/L)
            1.0
            2.0
            ...
            5.0

    Parameters
    ----------------
    filepath: str
        The filepath to the reference file.

    Returns
    ----------------
    absorbance:np.ndarray (1d)
        The reference data
    header: str
        The header
    """
    reference_data = []
    with open(filepath, "r") as f:
        line = f.readline()
        header = line.split()[0]
        while line:
            try:
                line = f.readline()
                reference_data.append(float(line.split()[0]))
            except IndexError:
                pass
        f.close()
    return reference_data, header


def get_filelist(folderpath, mandatory_keywords, optional_keywords):
    """
    Get a list containing all filenames with given keywords in a folder. A filename must contain all mandatory keywords
    and any of the optional keywords.

    """
    filelist = os.listdir(folderpath)
    if isinstance(mandatory_keywords, str):
        mandatory_keywords = [mandatory_keywords]
    if isinstance(optional_keywords, str):
        optional_keywords = [optional_keywords]
    filelist_mandatory_filtered = []
    filelist_all_filtered = []

    if mandatory_keywords:
        for f in filelist:
            if all(kw in f for kw in mandatory_keywords):
                filelist_mandatory_filtered.append(f)
    else:
        filelist_mandatory_filtered = filelist

    if optional_keywords:
        for f in filelist_mandatory_filtered:
            if any(kw in f for kw in optional_keywords):
                filelist_all_filtered.append(f)
    else:
        filelist_all_filtered = filelist_mandatory_filtered

    return filelist_all_filtered


def read_parafac_model(filepath):
    """
    Import PARAFAC model from a text file written in the format suggested by OpenFluor (
    https://openfluor.lablicate.com/). Note that the models downloaded from OpenFluor normally don't have scores.

    Parameters
    ----------------
    filepath: str
        The filepath to the model file.

    Returns
    ----------------
    ex_df: pd.DataFrame
        Excitation loadings
    em_df: pd.DataFrame
        Emission loadings
    score_df: pd.DataFrame or None
        Scores (if there's any)
    info_dict: dict
        A dictionary containing the model information
    """
    with open(filepath, 'r') as f:
        line = f.readline().strip()
        line_count = 0
        while '#' in line:
            if "Fluorescence" in line:
                print("Reading fluorescence measurement info...")
            line = f.readline().strip()
            line_count += 1
        info_dict = {}
        while '#' not in line:
            phrase = line.split(sep='\t')
            if len(phrase) > 1:
                info_dict[phrase[0]] = phrase[1]
            else:
                info_dict[phrase[0]] = ''
            line = f.readline().strip()
            line_count += 1
        while '#' in line:
            if "Excitation" in line:
                print("Reading Ex/Em loadings...")
            line = f.readline().strip()
            line_count_spectra_start = line_count
            line_count += 1
        while "Ex" in line:
            line = f.readline().strip()
            line_count += 1
        line_count_ex = line_count
        ex_df = pd.read_csv(filepath, sep="\t", header=None, index_col=[0, 1],
                            skiprows=line_count_spectra_start + 1, nrows=line_count_ex - line_count_spectra_start - 1)
        component_label = ['component {rank}'.format(rank=r + 1) for r in range(ex_df.shape[1])]
        ex_df.columns = component_label
        ex_df.index.names = ['type', 'wavelength']
        while "Em" in line:
            line = f.readline().strip()
            line_count += 1
        line_count_em = line_count
        em_df = pd.read_csv(filepath, sep='\t', header=None, index_col=[0, 1],
                            skiprows=line_count_ex, nrows=line_count_em - line_count_ex)
        em_df.columns = component_label
        em_df.index.names = ['type', 'wavelength']
        score_df = None
        while '#' in line:
            if "Score" in line:
                print("Reading component scores...")
            line = f.readline().strip()
            line_count += 1
        line_count_score = line_count
        while 'Score' in line:
            line = f.readline().strip()
            line_count += 1
        while '#' in line:
            if 'end' in line:
                line_count_end = line_count
                score_df = pd.read_csv(filepath, sep="\t", header=None, index_col=[0, 1],
                                       skiprows=line_count_score, nrows=line_count_end - line_count_score)
                score_df.index = score_df.index.set_levels(
                    [score_df.index.levels[0], pd.to_datetime(score_df.index.levels[1])])
                score_df.columns = component_label
                score_df.index.names = ['type', 'time']
                print('Reading complete')
                line = f.readline().strip()
        f.close()
    return ex_df, em_df, score_df, info_dict


def read_parafac_models(datdir, kw):
    """
    Search all PARAFAC models in a folder by keyword in filenames and import all of them into a dictionary using
    read_parafac_model()
    """
    datlist = get_filelist(datdir, kw)
    parafac_results = []
    for f in datlist:
        filepath = datdir + '/' + f
        ex_df, em_df, score_df, info_dict = read_parafac_model(filepath)
        info_dict['filename'] = f
        d = {'info': info_dict, 'ex': ex_df, 'em': em_df, 'score': score_df}
        parafac_results.append(d)
    return parafac_results

# def str_to_datetime(ts_string, ts_format='%Y-%m-%d-%H-%M-%S'):
#     return datetime.strptime(ts_string, ts_format)
