# eempy

Author: Yongmin Hu (yongminhu@outlook.com)

Last update: 2026-01

This package provides tools for Excitation-Emission Matrix (EEM) fluorescence analysis, including data I/O,
preprocessing, decomposition (PARAFAC/NMF with optional solvers/regularizations), and validation workflows.


## Before jumping into coding

If you wish to do the analysis with an app without coding yourself, you are welcomed to try eempy-vis (https://github.com/YongminHu/eempy-vis). It provides prompt visualization, preprocessing and various EEM interpretation options with a clean UI.

## Get Started

(English documentation: https://yongminhu.github.io/eempy/)

### Installation
```sh
pip install eem-python
```

### Read EEMs and absorbance from text files (.csv, .dat, .txt, etc.)


```python
from pathlib import Path
from eempy.read_data import read_eem_dataset, read_abs_dataset

# ---------Read EEMs and absorbance from text files (.csv, .dat, .txt, etc.)-----------

data_dir = Path("tests") / "sample_data" # path to data folder
eem_stack, ex_range, em_range, indexes = read_eem_dataset(
    str(data_dir),
    # mandatory_keywords: all filenames must include these substrings
    # optional_keywords: filenames must include at least one of these substrings
    # For example, with:
    # - mandatory_keywords=["PEM"]
    # - optional_keywords=["2021-02-02", "2021-02-01"]
    # only files whose filenames meet the following criteria are included:
    # - filename must contain "PEM"
    # - filename must also contain either "2021-02-01" or "2021-02-02"
    mandatory_keywords=["PEM"],
    optional_keywords=["2021-02-01", "2021-02-02"],
    # file_first_row: "ex" means the first row lists excitation wavelengths in the text files.
    # use "em" if the first row means emission wavelengths. If you are using .dat file
    # generated from HORIBA aqualog, use "ex" by default.
    file_first_row="ex",
    # index_pos: (start, end) positions in filenames (1-based, end inclusive) used to extract
    # sample labels from filenames. In the example, a string (something like 
    # "2021-02-01-1400_R1") would be extracted from position 1 to 19 for each sample. It would
    # be used later in the output tables of EEM analysis.  
    index_pos=(1, 19)
)
abs_stack, ex_range_abs, _ = read_abs_dataset(
    str(data_dir),
    mandatory_keywords=["ABS"],
    optional_keywords=["2021-02-01", "2021-02-02"],
)
```
Other data I/O helpers include: `read_eem` for a single EEM file, `read_abs` for a single absorbance spectrum,
`read_eem_dataset_from_json` for loading a saved EEMDataset from JSON, and `read_reference_from_text` for
1D reference variables (e.g., DOC) used in correlation or calibration workflows.

### Build EEM dataset

```python
from eempy.eem_processing import EEMDataset

# Using eem_stack, ex_range, em_range and indexes generated from above
dataset = EEMDataset(
    eem_stack=eem_stack, # 3d array of EEMs
    ex_range=ex_range,
    em_range=em_range,
    index=indexes, # sample labels
)
```
You can also attach reference data (`ref`) such as concentration tables and sample classes (`cluster`).
These labels propagate to outputs (e.g., Fmax tables) and enable correlation analysis and group-wise filtering.

### Preprocessing: inner filter effect (IFE) correction, scattering removal, etc.

```python
dataset = dataset.ife_correction(
    # absorbance/ex_range_abs: absorbance spectra and their wavelength axis used for IFE
    # correction
    absorbance=abs_stack, 
    ex_range_abs=ex_range_abs, 
    inplace=False
    )
dataset = dataset.rayleigh_scattering_removal(
    # width_*: width of scattering bands to be interpolated for first-order and second-order
    # scattering.
    width_o1=10,
    width_o2=10,
    # interpolation_method_*: how to fill the masked Rayleigh bands ("zero", "linear",
    # "cubic", "nan") for first-order and second-order scatterings.
    interpolation_method_o1="zero",
    interpolation_method_o2="zero",
    inplace=False,
)
dataset = dataset.raman_scattering_removal(
    width=5,
    # interpolation_method: how to fill the Raman band ("zero", "linear", "cubic", "nan")
    interpolation_method="zero",
    interpolation_dimension="2d",
    inplace=False,
)
dataset = dataset.cutting(
    # ex_min/ex_max/em_min/em_max: wavelength window retained after cutting (nm)
    ex_min=240,
    ex_max=450,
    em_min=300,
    em_max=550,
    inplace=False,
)
```
Additional preprocessing options include:
- threshold masking to clip extreme intensities,
- median/Gaussian filtering for noise suppression,
- NaN imputation to fill masked pixels,
- interpolation to a new excitation/emission grid,
- total-fluorescence or Raman normalization for intensity scaling.

### Peak picking and regional integration

```python
# peak_picking: target excitation/emission (nm); returns the closest grid point
peak_fi, ex_actual, em_actual = dataset.peak_picking(ex=350, em=450)
print(ex_actual, em_actual)
# regional_integration: sum of fluorescence over a rectangular ex/em region (nm)
ri = dataset.regional_integration(ex_min=250, ex_max=300, em_min=380, em_max=450)
print(peak_fi.head())
print(ri.head())
```
**eempy** also have functions to calculate other fluorescence indicators, including HIX, BIX, FI, AQY, and total fluorescence.

### PARAFAC analysis

```python
from eempy.eem_processing import PARAFAC

parafac = PARAFAC(
    n_components=3, # n_components: number of components (rank)
    solver="hals", # recommended solver
    init="svd", # SVD-based initialization
    # max_iter_als/tol: ALS iteration budget, influencing computation time/accuracy
    max_iter_nnls=300,
    max_iter_als=200,
    random_state=0, # random seed
)
parafac.fit(dataset)
```

**eempy** offers different regularization options to potentially strengthen the accuracy and physical interpretability of the analysis:

- **Non-negativity**
- **Elastic-net regularization** on any factor (L1/L2 mix)
- **Quadratic priors** on sample, excitation, or emission loadings.  
  This is useful when fitted scores or spectral components are desired to be close (but not necessarily identical) to prior knowledge.
- **Ratio constraint on paired rows of sample scores**:  
  `score[idx_top] ≈ beta * score[idx_bot]`

  This is useful when the ratios of component amplitudes between two sets of samples are desired to be constant.  
  For example, if each sample is measured both unquenched and quenched using a fixed quencher dosage, then for a given chemically consistent component the ratio between unquenched and quenched amplitudes may be approximately constant across samples (Hu et al., *ES&T*, 2025).

  In this case, passing the unquenched and quenched sample indices to `idx_top` and `idx_bot` encourages a constant ratio.  
  `lam` controls the strength of this regularization.

### Split-half validation 

```python
from eempy.eem_processing import SplitValidation

validator = SplitValidation(
    base_model=parafac,
    n_splits=4, # 4 splits and 6 combinations
    combination_size="half",
    rule="random",
    random_state=0,
)
validator.fit(dataset)
similarities = validator.compare_parafac_loadings()
print(similarities[0].head())
```
For NMF models, use `compare_components()` to assess component similarity across split-half models; PARAFAC
also supports excitation/emission loading comparisons via `compare_parafac_loadings()`.

Other validation methods include variance explained and core consistency.

### Outputs and visualization

```python
from eempy.plot import plot_eem, plot_fi, plot_loadings, plot_fmax

# Compare a raw EEM and the processed EEM (sample 0)
plot_eem(
    eem_stack[0], 
    ex_range, 
    em_range, 
    title="Raw EEM", 
    display=True
    )
plot_eem(
    dataset.eem_stack[0], 
    dataset.ex_range, 
    dataset.em_range, 
    title="Processed EEM", 
    display=True
    )

# Visualize peak picking
plot_fi(peak_fi)

# Visualize PARAFAC loadings and Fmax
plot_loadings(
    {"PARAFAC": parafac}, 
    )
plot_fmax(parafac)
```
Other plotting helpers include EEM stack grids (`plot_eem_stack`), absorbance curves (`plot_abs`), and score plots
(`plot_score`). Most plotting functions support both matplotlib and plotly backends.
