"""
Central registry of Dash component IDs.

Import IDs from here instead of scattering raw strings.
"""
from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True)
class IDS:
    """All component IDs used in the app."""
    ABSORBANCE_GRAPH: str = "absorbance-graph"
    ABS_DATA_FORMAT: str = "abs-data-format"
    ALIGN_EXEM: str = "align-exem"
    BUILD_EEM_DATASET: str = "build-eem-dataset"
    BUILD_EEM_DATASET_SPINNER: str = "build-eem-dataset-spinner"
    BUILD_KMETHOD_CLUSTERING: str = "build-kmethod-clustering"
    BUILD_KMETHOD_CONSENSUS: str = "build-kmethod-consensus"
    BUILD_NMF_MODEL: str = "build-nmf-model"
    BUILD_PARAFAC_MODEL: str = "build-parafac-model"
    BUILD_PARAFAC_SPINNER: str = "build-parafac-spinner"
    BUILD_PP_MODEL: str = "build-pp-model"
    BUILD_PP_SPINNER: str = "build-pp-spinner"
    BUILD_RI_MODEL: str = "build-ri-model"
    BUILD_RI_SPINNER: str = "build-ri-spinner"
    EEM_DATASET: str = "eem-dataset"
    EEM_DATASET_EXPORT_FORMAT: str = "eem-dataset-export-format"
    EEM_DATA_FORMAT: str = "eem-data-format"
    EEM_GRAPH: str = "eem-graph"
    EEM_GRAPH_OPTIONS: str = "eem-graph-options"
    EEM_PEAK_PICKING: str = "eem-peak-picking"
    EEM_PRE_PROCESSING: str = "eem-pre-processing"
    EEM_REGIONAL_INTEGRATION: str = "eem-regional-integration"
    EMISSION_WAVELENGTH_MAX: str = "emission-wavelength-max"
    EMISSION_WAVELENGTH_MIN: str = "emission-wavelength-min"
    EXCITATION_WAVELENGTH_MAX: str = "excitation-wavelength-max"
    EXCITATION_WAVELENGTH_MIN: str = "excitation-wavelength-min"
    EXPORT_EEM_DATASET: str = "export-eem-dataset"
    EXPORT_EEM_DATASET_SPINNER: str = "export-eem-dataset-spinner"
    FILENAME_EXPORT_EEM_DATASET: str = "filename-export-eem-dataset"
    FILENAME_SAMPLE_DROPDOWN: str = "filename-sample-dropdown"
    FILE_KEYWORD_ABSORBANCE: str = "file-keyword-absorbance"
    FILE_KEYWORD_BLANK: str = "file-keyword-blank"
    FILE_KEYWORD_MANDATORY: str = "file-keyword-mandatory"
    FILE_KEYWORD_OPTIONAL: str = "file-keyword-optional"
    FILE_KEYWORD_SAMPLE: str = "file-keyword-sample"
    FLUORESCENCE_INTENSITY_MAX: str = "fluorescence-intensity-max"
    FLUORESCENCE_INTENSITY_MIN: str = "fluorescence-intensity-min"
    FOLDER_PATH_EXPORT_EEM_DATASET: str = "folder-path-export-eem-dataset"
    FOLDER_PATH_INPUT: str = "folder-path-input"
    GAUSSIAN_BUTTON: str = "gaussian-button"
    GAUSSIAN_SIGMA: str = "gaussian-sigma"
    GAUSSIAN_TRUNCATE: str = "gaussian-truncate"
    HELP_DATA_FORMAT: str = "help-data-format"
    HELP_POP: str = "help-pop"
    HELP_QM: str = "help-qm"
    HOMEPAGE: str = "homepage"
    IFE_BUTTON: str = "ife-button"
    IFE_METHODS: str = "ife-methods"
    INDEX_POS_LEFT: str = "index-pos-left"
    INDEX_POS_RIGHT: str = "index-pos-right"
    INFO_EEM_DATASET: str = "info-eem-dataset"
    KMETHOD: str = "kmethod"
    KMETHOD_BASE_CLUSTERING_PARAMETERS: str = "kmethod-base-clustering-parameters"
    KMETHOD_BASE_MODEL: str = "kmethod-base-model"
    KMETHOD_BASE_MODEL_MESSAGE: str = "kmethod-base-model-message"
    KMETHOD_CLUSTER_EXPORT: str = "kmethod-cluster-export"
    KMETHOD_CLUSTER_EXPORT_MODEL_SELECTION: str = "kmethod-cluster-export-model-selection"
    KMETHOD_CLUSTER_FROM_FILE_CHECKBOX: str = "kmethod-cluster-from-file-checkbox"
    KMETHOD_COMPONENTS: str = "kmethod-components"
    KMETHOD_CONSENSUS_CONVERSION: str = "kmethod-consensus-conversion"
    KMETHOD_CONSENSUS_MATRIX: str = "kmethod-consensus-matrix"
    KMETHOD_CONSENSUS_MATRIX_DATA: str = "kmethod-consensus-matrix-data"
    KMETHOD_CONVERGENCE_TOLERANCE: str = "kmethod-convergence-tolerance"
    KMETHOD_DENDROGRAM: str = "kmethod-dendrogram"
    KMETHOD_EEM_DATASET_ESTABLISH: str = "kmethod-eem-dataset-establish"
    KMETHOD_EEM_DATASET_ESTABLISHMENT_MESSAGE: str = "kmethod-eem-dataset-establishment-message"
    KMETHOD_EEM_DATASET_ESTABLISHMENT_PATH_INPUT: str = "kmethod-eem-dataset-establishment-path-input"
    KMETHOD_EEM_DATASET_PREDICT_MESSAGE: str = "kmethod-eem-dataset-predict-message"
    KMETHOD_EEM_DATASET_PREDICT_PATH_INPUT: str = "kmethod-eem-dataset-predict-path-input"
    KMETHOD_ELIMINATION: str = "kmethod-elimination"
    KMETHOD_ERROR_HISTORY: str = "kmethod-error-history"
    KMETHOD_ESTABLISHMENT_COMPONENTS_CLUSTER_SELECTION: str = "kmethod-establishment-components-cluster-selection"
    KMETHOD_ESTABLISHMENT_COMPONENTS_GRAPH: str = "kmethod-establishment-components-graph"
    KMETHOD_ESTABLISHMENT_COMPONENTS_MODEL_SELECTION: str = "kmethod-establishment-components-model-selection"
    KMETHOD_ESTABLISHMENT_CORR: str = "kmethod-establishment-corr"
    KMETHOD_ESTABLISHMENT_CORR_GRAPH: str = "kmethod-establishment-corr-graph"
    KMETHOD_ESTABLISHMENT_CORR_INDICATOR_SELECTION: str = "kmethod-establishment-corr-indicator-selection"
    KMETHOD_ESTABLISHMENT_CORR_MODEL_SELECTION: str = "kmethod-establishment-corr-model-selection"
    KMETHOD_ESTABLISHMENT_CORR_REF_SELECTION: str = "kmethod-establishment-corr-ref-selection"
    KMETHOD_ESTABLISHMENT_CORR_TABLE: str = "kmethod-establishment-corr-table"
    KMETHOD_ESTABLISHMENT_INDEX_KW_MANDATORY: str = "kmethod-establishment-index-kw-mandatory"
    KMETHOD_ESTABLISHMENT_INDEX_KW_OPTIONAL: str = "kmethod-establishment-index-kw-optional"
    KMETHOD_FMAX: str = "kmethod-fmax"
    KMETHOD_LABEL_HISTORY: str = "kmethod-label-history"
    KMETHOD_MODELS: str = "kmethod-models"
    KMETHOD_NUM_BASE_CLUSTERINGS: str = "kmethod-num-base-clusterings"
    KMETHOD_NUM_FINAL_CLUSTERS: str = "kmethod-num-final-clusters"
    KMETHOD_NUM_INIT_SPLITS: str = "kmethod-num-init-splits"
    KMETHOD_NUM_ITERATIONS: str = "kmethod-num-iterations"
    KMETHOD_PREDICT: str = "kmethod-predict"
    KMETHOD_PREDICT_SPINNER: str = "kmethod-predict-spinner"
    KMETHOD_RANK: str = "kmethod-rank"
    KMETHOD_RECONSTRUCTION_ERROR_REDUCTION: str = "kmethod-reconstruction-error-reduction"
    KMETHOD_RESULTS1: str = "kmethod-results1"
    KMETHOD_RESULTS2: str = "kmethod-results2"
    KMETHOD_SILHOUETTE_SCORE: str = "kmethod-silhouette-score"
    KMETHOD_SORTED_CONSENSUS_MATRIX: str = "kmethod-sorted-consensus-matrix"
    KMETHOD_STEP1_SPINNER: str = "kmethod-step1-spinner"
    KMETHOD_STEP2_SPINNER: str = "kmethod-step2-spinner"
    KMETHOD_SUBSAMPLING_PORTION: str = "kmethod-subsampling-portion"
    KMETHOD_TEST_CORR_GRAPH: str = "kmethod-test-corr-graph"
    KMETHOD_TEST_CORR_INDICATOR_SELECTION: str = "kmethod-test-corr-indicator-selection"
    KMETHOD_TEST_CORR_REF_SELECTION: str = "kmethod-test-corr-ref-selection"
    KMETHOD_TEST_CORR_TABLE: str = "kmethod-test-corr-table"
    KMETHOD_TEST_ERROR: str = "kmethod-test-error"
    KMETHOD_TEST_FMAX: str = "kmethod-test-fmax"
    KMETHOD_TEST_INDEX_KW_MANDATORY: str = "kmethod-test-index-kw-mandatory"
    KMETHOD_TEST_INDEX_KW_OPTIONAL: str = "kmethod-test-index-kw-optional"
    KMETHOD_TEST_MODEL_SELECTION: str = "kmethod-test-model-selection"
    KMETHOD_TEST_PRED_GRAPH: str = "kmethod-test-pred-graph"
    KMETHOD_TEST_PRED_MODEL: str = "kmethod-test-pred-model"
    KMETHOD_TEST_PRED_REF_SELECTION: str = "kmethod-test-pred-ref-selection"
    KMETHOD_TEST_PRED_TABLE: str = "kmethod-test-pred-table"
    KMETHOD_TEST_RESULTS: str = "kmethod-test-results"
    KMETHOD_VALIDATIONS: str = "kmethod-validations"
    MEDIAN_FILTER_BUTTON: str = "median-filter-button"
    MEDIAN_FILTER_MODE: str = "median-filter-mode"
    MEDIAN_FILTER_WINDOW_EM: str = "median-filter-window-em"
    MEDIAN_FILTER_WINDOW_EX: str = "median-filter-window-ex"
    MESSAGE_EEM_DATASET_EXPORT: str = "message-eem-dataset-export"
    NMF: str = "nmf"
    NMF_ALPHA_H: str = "nmf-alpha-h"
    NMF_ALPHA_W: str = "nmf-alpha-w"
    NMF_COMPONENTS: str = "nmf-components"
    NMF_EEM_DATASET_ESTABLISHMENT_MESSAGE: str = "nmf-eem-dataset-establishment-message"
    NMF_EEM_DATASET_ESTABLISHMENT_PATH_INPUT: str = "nmf-eem-dataset-establishment-path-input"
    NMF_EEM_DATASET_PREDICT_MESSAGE: str = "nmf-eem-dataset-predict-message"
    NMF_EEM_DATASET_PREDICT_PATH_INPUT: str = "nmf-eem-dataset-predict-path-input"
    NMF_ESTABLISHMENT_CORR: str = "nmf-establishment-corr"
    NMF_ESTABLISHMENT_CORR_GRAPH: str = "nmf-establishment-corr-graph"
    NMF_ESTABLISHMENT_CORR_INDICATOR_SELECTION: str = "nmf-establishment-corr-indicator-selection"
    NMF_ESTABLISHMENT_CORR_MODEL_SELECTION: str = "nmf-establishment-corr-model-selection"
    NMF_ESTABLISHMENT_CORR_REF_SELECTION: str = "nmf-establishment-corr-ref-selection"
    NMF_ESTABLISHMENT_CORR_TABLE: str = "nmf-establishment-corr-table"
    NMF_ESTABLISHMENT_INDEX_KW_MANDATORY: str = "nmf-establishment-index-kw-mandatory"
    NMF_ESTABLISHMENT_INDEX_KW_OPTIONAL: str = "nmf-establishment-index-kw-optional"
    NMF_ESTABLISHMENT_RECONSTRUCTION_ERROR: str = "nmf-establishment-reconstruction-error"
    NMF_FMAX: str = "nmf-fmax"
    NMF_INIT: str = "nmf-init"
    NMF_L1_RATIO: str = "nmf-l1-ratio"
    NMF_MAX_ITER_ALS: str = "nmf-max-iter-als"
    NMF_MAX_ITER_NNLS: str = "nmf-max-iter-nnls"
    NMF_MODELS: str = "nmf-models"
    NMF_NORMALIZATION_CHECKBOX: str = "nmf-normalization-checkbox"
    NMF_PREDICT: str = "nmf-predict"
    NMF_PREDICT_SPINNER: str = "nmf-predict-spinner"
    NMF_RANK: str = "nmf-rank"
    NMF_RESULTS: str = "nmf-results"
    NMF_SOLVER: str = "nmf-solver"
    NMF_SPINNER: str = "nmf-spinner"
    NMF_SPLIT_HALF: str = "nmf-split-half"
    NMF_TEST_CORR_GRAPH: str = "nmf-test-corr-graph"
    NMF_TEST_CORR_INDICATOR_SELECTION: str = "nmf-test-corr-indicator-selection"
    NMF_TEST_CORR_REF_SELECTION: str = "nmf-test-corr-ref-selection"
    NMF_TEST_CORR_TABLE: str = "nmf-test-corr-table"
    NMF_TEST_ERROR: str = "nmf-test-error"
    NMF_TEST_FMAX: str = "nmf-test-fmax"
    NMF_TEST_INDEX_KW_MANDATORY: str = "nmf-test-index-kw-mandatory"
    NMF_TEST_INDEX_KW_OPTIONAL: str = "nmf-test-index-kw-optional"
    NMF_TEST_MODEL_SELECTION: str = "nmf-test-model-selection"
    NMF_TEST_PRED_GRAPH: str = "nmf-test-pred-graph"
    NMF_TEST_PRED_MODEL: str = "nmf-test-pred-model"
    NMF_TEST_PRED_MODEL_SELECTION: str = "nmf-test-pred-model-selection"
    NMF_TEST_PRED_REF_SELECTION: str = "nmf-test-pred-ref-selection"
    NMF_TEST_PRED_TABLE: str = "nmf-test-pred-table"
    NMF_TEST_RESULTS: str = "nmf-test-results"
    NMF_VALIDATIONS: str = "nmf-validations"
    PARAFAC: str = "parafac"
    PARAFAC_COMPONENTS: str = "parafac-components"
    PARAFAC_CORE_CONSISTENCY: str = "parafac-core-consistency"
    PARAFAC_EEM_DATASET_ESTABLISHMENT_MESSAGE: str = "parafac-eem-dataset-establishment-message"
    PARAFAC_EEM_DATASET_ESTABLISHMENT_PATH_INPUT: str = "parafac-eem-dataset-establishment-path-input"
    PARAFAC_EEM_DATASET_PREDICT_MESSAGE: str = "parafac-eem-dataset-predict-message"
    PARAFAC_EEM_DATASET_PREDICT_PATH_INPUT: str = "parafac-eem-dataset-predict-path-input"
    PARAFAC_ESTABLISHMENT_CORR: str = "parafac-establishment-corr"
    PARAFAC_ESTABLISHMENT_CORR_GRAPH: str = "parafac-establishment-corr-graph"
    PARAFAC_ESTABLISHMENT_CORR_INDICATOR_SELECTION: str = "parafac-establishment-corr-indicator-selection"
    PARAFAC_ESTABLISHMENT_CORR_MODEL_SELECTION: str = "parafac-establishment-corr-model-selection"
    PARAFAC_ESTABLISHMENT_CORR_REF_SELECTION: str = "parafac-establishment-corr-ref-selection"
    PARAFAC_ESTABLISHMENT_CORR_TABLE: str = "parafac-establishment-corr-table"
    PARAFAC_ESTABLISHMENT_INDEX_KW_MANDATORY: str = "parafac-establishment-index-kw-mandatory"
    PARAFAC_ESTABLISHMENT_INDEX_KW_OPTIONAL: str = "parafac-establishment-index-kw-optional"
    PARAFAC_ESTABLISHMENT_RECONSTRUCTION_ERROR: str = "parafac-establishment-reconstruction-error"
    PARAFAC_FMAX: str = "parafac-fmax"
    PARAFAC_INIT_METHOD: str = "parafac-init-method"
    PARAFAC_LEVERAGE: str = "parafac-leverage"
    PARAFAC_LOADINGS: str = "parafac-loadings"
    PARAFAC_MODELS: str = "parafac-models"
    PARAFAC_NN_CHECKBOX: str = "parafac-nn-checkbox"
    PARAFAC_PREDICT: str = "parafac-predict"
    PARAFAC_PREDICT_SPINNER: str = "parafac-predict-spinner"
    PARAFAC_RANK: str = "parafac-rank"
    PARAFAC_RESULTS: str = "parafac-results"
    PARAFAC_SCORES: str = "parafac-scores"
    PARAFAC_SOLVER: str = "parafac-solver"
    PARAFAC_SPLIT_HALF: str = "parafac-split-half"
    PARAFAC_TEST_CORR_GRAPH: str = "parafac-test-corr-graph"
    PARAFAC_TEST_CORR_INDICATOR_SELECTION: str = "parafac-test-corr-indicator-selection"
    PARAFAC_TEST_CORR_REF_SELECTION: str = "parafac-test-corr-ref-selection"
    PARAFAC_TEST_CORR_TABLE: str = "parafac-test-corr-table"
    PARAFAC_TEST_ERROR: str = "parafac-test-error"
    PARAFAC_TEST_FMAX: str = "parafac-test-fmax"
    PARAFAC_TEST_INDEX_KW_MANDATORY: str = "parafac-test-index-kw-mandatory"
    PARAFAC_TEST_INDEX_KW_OPTIONAL: str = "parafac-test-index-kw-optional"
    PARAFAC_TEST_MODEL_SELECTION: str = "parafac-test-model-selection"
    PARAFAC_TEST_PRED_GRAPH: str = "parafac-test-pred-graph"
    PARAFAC_TEST_PRED_MODEL: str = "parafac-test-pred-model"
    PARAFAC_TEST_PRED_MODEL_SELECTION: str = "parafac-test-pred-model-selection"
    PARAFAC_TEST_PRED_REF_SELECTION: str = "parafac-test-pred-ref-selection"
    PARAFAC_TEST_PRED_TABLE: str = "parafac-test-pred-table"
    PARAFAC_TEST_RESULTS: str = "parafac-test-results"
    PARAFAC_TEST_SCORE: str = "parafac-test-score"
    PARAFAC_TF_CHECKBOX: str = "parafac-tf-checkbox"
    PARAFAC_VALIDATIONS: str = "parafac-validations"
    PARAFAC_VARIANCE_EXPLAINED: str = "parafac-variance-explained"
    PATH_REFERENCE: str = "path-reference"
    PP_EEM_DATASET_ESTABLISHMENT_MESSAGE: str = "pp-eem-dataset-establishment-message"
    PP_EEM_DATASET_ESTABLISHMENT_PATH_INPUT: str = "pp-eem-dataset-establishment-path-input"
    PP_EEM_DATASET_PREDICT_MESSAGE: str = "pp-eem-dataset-predict-message"
    PP_EEM_DATASET_PREDICT_PATH_INPUT: str = "pp-eem-dataset-predict-path-input"
    PP_EMISSION: str = "pp-emission"
    PP_ESTABLISHMENT_CORR: str = "pp-establishment-corr"
    PP_ESTABLISHMENT_CORR_GRAPH: str = "pp-establishment-corr-graph"
    PP_ESTABLISHMENT_CORR_REF_SELECTION: str = "pp-establishment-corr-ref-selection"
    PP_ESTABLISHMENT_CORR_TABLE: str = "pp-establishment-corr-table"
    PP_ESTABLISHMENT_INDEX_KW_MANDATORY: str = "pp-establishment-index-kw-mandatory"
    PP_ESTABLISHMENT_INDEX_KW_OPTIONAL: str = "pp-establishment-index-kw-optional"
    PP_EXCITATION: str = "pp-excitation"
    PP_INTENSITIES: str = "pp-intensities"
    PP_MODEL: str = "pp-model"
    PP_PREDICT: str = "pp-predict"
    PP_PREDICT_SPINNER: str = "pp-predict-spinner"
    PP_RESULTS: str = "pp-results"
    PP_TEST_CORR_GRAPH: str = "pp-test-corr-graph"
    PP_TEST_CORR_INDICATOR_SELECTION: str = "pp-test-corr-indicator-selection"
    PP_TEST_CORR_REF_SELECTION: str = "pp-test-corr-ref-selection"
    PP_TEST_CORR_TABLE: str = "pp-test-corr-table"
    PP_TEST_ERROR: str = "pp-test-error"
    PP_TEST_INDEX_KW_MANDATORY: str = "pp-test-index-kw-mandatory"
    PP_TEST_INDEX_KW_OPTIONAL: str = "pp-test-index-kw-optional"
    PP_TEST_INTENSITIES: str = "pp-test-intensities"
    PP_TEST_PRED_GRAPH: str = "pp-test-pred-graph"
    PP_TEST_PRED_MODEL: str = "pp-test-pred-model"
    PP_TEST_PRED_REF_SELECTION: str = "pp-test-pred-ref-selection"
    PP_TEST_PRED_TABLE: str = "pp-test-pred-table"
    PP_TEST_RESULTS: str = "pp-test-results"
    PREDICT_KMETHOD_MODEL: str = "predict-kmethod-model"
    PREDICT_NMF_MODEL: str = "predict-nmf-model"
    PREDICT_PARAFAC_MODEL: str = "predict-parafac-model"
    PREDICT_PP_MODEL: str = "predict-pp-model"
    PREDICT_RI_MODEL: str = "predict-ri-model"
    PRE_PROCESSED_EEM: str = "pre-processed-eem"
    RAMAN_BUTTON: str = "raman-button"
    RAMAN_DIMENSION: str = "raman-dimension"
    RAMAN_METHODS: str = "raman-methods"
    RAMAN_WIDTH: str = "raman-width"
    RAYLEIGH_BUTTON: str = "rayleigh-button"
    RAYLEIGH_O1_DIMENSION: str = "rayleigh-o1-dimension"
    RAYLEIGH_O1_METHODS: str = "rayleigh-o1-methods"
    RAYLEIGH_O1_WIDTH: str = "rayleigh-o1-width"
    RAYLEIGH_O2_DIMENSION: str = "rayleigh-o2-dimension"
    RAYLEIGH_O2_METHODS: str = "rayleigh-o2-methods"
    RAYLEIGH_O2_WIDTH: str = "rayleigh-o2-width"
    RI_EEM_DATASET_ESTABLISHMENT_MESSAGE: str = "ri-eem-dataset-establishment-message"
    RI_EEM_DATASET_ESTABLISHMENT_PATH_INPUT: str = "ri-eem-dataset-establishment-path-input"
    RI_EEM_DATASET_PREDICT_MESSAGE: str = "ri-eem-dataset-predict-message"
    RI_EEM_DATASET_PREDICT_PATH_INPUT: str = "ri-eem-dataset-predict-path-input"
    RI_EM_MAX: str = "ri-em-max"
    RI_EM_MIN: str = "ri-em-min"
    RI_ESTABLISHMENT_CORR: str = "ri-establishment-corr"
    RI_ESTABLISHMENT_CORR_GRAPH: str = "ri-establishment-corr-graph"
    RI_ESTABLISHMENT_CORR_REF_SELECTION: str = "ri-establishment-corr-ref-selection"
    RI_ESTABLISHMENT_CORR_TABLE: str = "ri-establishment-corr-table"
    RI_ESTABLISHMENT_INDEX_KW_MANDATORY: str = "ri-establishment-index-kw-mandatory"
    RI_ESTABLISHMENT_INDEX_KW_OPTIONAL: str = "ri-establishment-index-kw-optional"
    RI_EX_MAX: str = "ri-ex-max"
    RI_EX_MIN: str = "ri-ex-min"
    RI_INTENSITIES: str = "ri-intensities"
    RI_MODEL: str = "ri-model"
    RI_PREDICT: str = "ri-predict"
    RI_PREDICT_SPINNER: str = "ri-predict-spinner"
    RI_RESULTS: str = "ri-results"
    RI_TEST_CORR_GRAPH: str = "ri-test-corr-graph"
    RI_TEST_CORR_INDICATOR_SELECTION: str = "ri-test-corr-indicator-selection"
    RI_TEST_CORR_REF_SELECTION: str = "ri-test-corr-ref-selection"
    RI_TEST_CORR_TABLE: str = "ri-test-corr-table"
    RI_TEST_ERROR: str = "ri-test-error"
    RI_TEST_INDEX_KW_MANDATORY: str = "ri-test-index-kw-mandatory"
    RI_TEST_INDEX_KW_OPTIONAL: str = "ri-test-index-kw-optional"
    RI_TEST_INTENSITIES: str = "ri-test-intensities"
    RI_TEST_PRED_GRAPH: str = "ri-test-pred-graph"
    RI_TEST_PRED_MODEL: str = "ri-test-pred-model"
    RI_TEST_PRED_REF_SELECTION: str = "ri-test-pred-ref-selection"
    RI_TEST_PRED_TABLE: str = "ri-test-pred-table"
    RI_TEST_RESULTS: str = "ri-test-results"
    RSU_DISPLAY: str = "rsu-display"
    SU_BUTTON: str = "su-button"
    SU_EMISSION_WIDTH: str = "su-emission-width"
    SU_EXCITATION: str = "su-excitation"
    SU_NORMALIZATION_FACTOR: str = "su-normalization-factor"
    TABS_CONTENT: str = "tabs-content"
    TIMESTAMP_CHECKBOX: str = "timestamp-checkbox"
    TIMESTAMP_FORMAT: str = "timestamp-format"

# Useful mappings
ID_TO_CONST = {
    "absorbance-graph": "ABSORBANCE_GRAPH",
    "abs-data-format": "ABS_DATA_FORMAT",
    "align-exem": "ALIGN_EXEM",
    "build-eem-dataset": "BUILD_EEM_DATASET",
    "build-eem-dataset-spinner": "BUILD_EEM_DATASET_SPINNER",
    "build-kmethod-clustering": "BUILD_KMETHOD_CLUSTERING",
    "build-kmethod-consensus": "BUILD_KMETHOD_CONSENSUS",
    "build-nmf-model": "BUILD_NMF_MODEL",
    "build-parafac-model": "BUILD_PARAFAC_MODEL",
    "build-parafac-spinner": "BUILD_PARAFAC_SPINNER",
    "build-pp-model": "BUILD_PP_MODEL",
    "build-pp-spinner": "BUILD_PP_SPINNER",
    "build-ri-model": "BUILD_RI_MODEL",
    "build-ri-spinner": "BUILD_RI_SPINNER",
    "eem-dataset": "EEM_DATASET",
    "eem-dataset-export-format": "EEM_DATASET_EXPORT_FORMAT",
    "eem-data-format": "EEM_DATA_FORMAT",
    "eem-graph": "EEM_GRAPH",
    "eem-graph-options": "EEM_GRAPH_OPTIONS",
    "eem-peak-picking": "EEM_PEAK_PICKING",
    "eem-pre-processing": "EEM_PRE_PROCESSING",
    "eem-regional-integration": "EEM_REGIONAL_INTEGRATION",
    "emission-wavelength-max": "EMISSION_WAVELENGTH_MAX",
    "emission-wavelength-min": "EMISSION_WAVELENGTH_MIN",
    "excitation-wavelength-max": "EXCITATION_WAVELENGTH_MAX",
    "excitation-wavelength-min": "EXCITATION_WAVELENGTH_MIN",
    "export-eem-dataset": "EXPORT_EEM_DATASET",
    "export-eem-dataset-spinner": "EXPORT_EEM_DATASET_SPINNER",
    "filename-export-eem-dataset": "FILENAME_EXPORT_EEM_DATASET",
    "filename-sample-dropdown": "FILENAME_SAMPLE_DROPDOWN",
    "file-keyword-absorbance": "FILE_KEYWORD_ABSORBANCE",
    "file-keyword-blank": "FILE_KEYWORD_BLANK",
    "file-keyword-mandatory": "FILE_KEYWORD_MANDATORY",
    "file-keyword-optional": "FILE_KEYWORD_OPTIONAL",
    "file-keyword-sample": "FILE_KEYWORD_SAMPLE",
    "fluorescence-intensity-max": "FLUORESCENCE_INTENSITY_MAX",
    "fluorescence-intensity-min": "FLUORESCENCE_INTENSITY_MIN",
    "folder-path-export-eem-dataset": "FOLDER_PATH_EXPORT_EEM_DATASET",
    "folder-path-input": "FOLDER_PATH_INPUT",
    "gaussian-button": "GAUSSIAN_BUTTON",
    "gaussian-sigma": "GAUSSIAN_SIGMA",
    "gaussian-truncate": "GAUSSIAN_TRUNCATE",
    "help-data-format": "HELP_DATA_FORMAT",
    "help-pop": "HELP_POP",
    "help-qm": "HELP_QM",
    "homepage": "HOMEPAGE",
    "ife-button": "IFE_BUTTON",
    "ife-methods": "IFE_METHODS",
    "index-pos-left": "INDEX_POS_LEFT",
    "index-pos-right": "INDEX_POS_RIGHT",
    "info-eem-dataset": "INFO_EEM_DATASET",
    "kmethod": "KMETHOD",
    "kmethod-base-clustering-parameters": "KMETHOD_BASE_CLUSTERING_PARAMETERS",
    "kmethod-base-model": "KMETHOD_BASE_MODEL",
    "kmethod-base-model-message": "KMETHOD_BASE_MODEL_MESSAGE",
    "kmethod-cluster-export": "KMETHOD_CLUSTER_EXPORT",
    "kmethod-cluster-export-model-selection": "KMETHOD_CLUSTER_EXPORT_MODEL_SELECTION",
    "kmethod-cluster-from-file-checkbox": "KMETHOD_CLUSTER_FROM_FILE_CHECKBOX",
    "kmethod-components": "KMETHOD_COMPONENTS",
    "kmethod-consensus-conversion": "KMETHOD_CONSENSUS_CONVERSION",
    "kmethod-consensus-matrix": "KMETHOD_CONSENSUS_MATRIX",
    "kmethod-consensus-matrix-data": "KMETHOD_CONSENSUS_MATRIX_DATA",
    "kmethod-convergence-tolerance": "KMETHOD_CONVERGENCE_TOLERANCE",
    "kmethod-dendrogram": "KMETHOD_DENDROGRAM",
    "kmethod-eem-dataset-establish": "KMETHOD_EEM_DATASET_ESTABLISH",
    "kmethod-eem-dataset-establishment-message": "KMETHOD_EEM_DATASET_ESTABLISHMENT_MESSAGE",
    "kmethod-eem-dataset-establishment-path-input": "KMETHOD_EEM_DATASET_ESTABLISHMENT_PATH_INPUT",
    "kmethod-eem-dataset-predict-message": "KMETHOD_EEM_DATASET_PREDICT_MESSAGE",
    "kmethod-eem-dataset-predict-path-input": "KMETHOD_EEM_DATASET_PREDICT_PATH_INPUT",
    "kmethod-elimination": "KMETHOD_ELIMINATION",
    "kmethod-error-history": "KMETHOD_ERROR_HISTORY",
    "kmethod-establishment-components-cluster-selection": "KMETHOD_ESTABLISHMENT_COMPONENTS_CLUSTER_SELECTION",
    "kmethod-establishment-components-graph": "KMETHOD_ESTABLISHMENT_COMPONENTS_GRAPH",
    "kmethod-establishment-components-model-selection": "KMETHOD_ESTABLISHMENT_COMPONENTS_MODEL_SELECTION",
    "kmethod-establishment-corr": "KMETHOD_ESTABLISHMENT_CORR",
    "kmethod-establishment-corr-graph": "KMETHOD_ESTABLISHMENT_CORR_GRAPH",
    "kmethod-establishment-corr-indicator-selection": "KMETHOD_ESTABLISHMENT_CORR_INDICATOR_SELECTION",
    "kmethod-establishment-corr-model-selection": "KMETHOD_ESTABLISHMENT_CORR_MODEL_SELECTION",
    "kmethod-establishment-corr-ref-selection": "KMETHOD_ESTABLISHMENT_CORR_REF_SELECTION",
    "kmethod-establishment-corr-table": "KMETHOD_ESTABLISHMENT_CORR_TABLE",
    "kmethod-establishment-index-kw-mandatory": "KMETHOD_ESTABLISHMENT_INDEX_KW_MANDATORY",
    "kmethod-establishment-index-kw-optional": "KMETHOD_ESTABLISHMENT_INDEX_KW_OPTIONAL",
    "kmethod-fmax": "KMETHOD_FMAX",
    "kmethod-label-history": "KMETHOD_LABEL_HISTORY",
    "kmethod-models": "KMETHOD_MODELS",
    "kmethod-num-base-clusterings": "KMETHOD_NUM_BASE_CLUSTERINGS",
    "kmethod-num-final-clusters": "KMETHOD_NUM_FINAL_CLUSTERS",
    "kmethod-num-init-splits": "KMETHOD_NUM_INIT_SPLITS",
    "kmethod-num-iterations": "KMETHOD_NUM_ITERATIONS",
    "kmethod-predict": "KMETHOD_PREDICT",
    "kmethod-predict-spinner": "KMETHOD_PREDICT_SPINNER",
    "kmethod-rank": "KMETHOD_RANK",
    "kmethod-reconstruction-error-reduction": "KMETHOD_RECONSTRUCTION_ERROR_REDUCTION",
    "kmethod-results1": "KMETHOD_RESULTS1",
    "kmethod-results2": "KMETHOD_RESULTS2",
    "kmethod-silhouette-score": "KMETHOD_SILHOUETTE_SCORE",
    "kmethod-sorted-consensus-matrix": "KMETHOD_SORTED_CONSENSUS_MATRIX",
    "kmethod-step1-spinner": "KMETHOD_STEP1_SPINNER",
    "kmethod-step2-spinner": "KMETHOD_STEP2_SPINNER",
    "kmethod-subsampling-portion": "KMETHOD_SUBSAMPLING_PORTION",
    "kmethod-test-corr-graph": "KMETHOD_TEST_CORR_GRAPH",
    "kmethod-test-corr-indicator-selection": "KMETHOD_TEST_CORR_INDICATOR_SELECTION",
    "kmethod-test-corr-ref-selection": "KMETHOD_TEST_CORR_REF_SELECTION",
    "kmethod-test-corr-table": "KMETHOD_TEST_CORR_TABLE",
    "kmethod-test-error": "KMETHOD_TEST_ERROR",
    "kmethod-test-fmax": "KMETHOD_TEST_FMAX",
    "kmethod-test-index-kw-mandatory": "KMETHOD_TEST_INDEX_KW_MANDATORY",
    "kmethod-test-index-kw-optional": "KMETHOD_TEST_INDEX_KW_OPTIONAL",
    "kmethod-test-model-selection": "KMETHOD_TEST_MODEL_SELECTION",
    "kmethod-test-pred-graph": "KMETHOD_TEST_PRED_GRAPH",
    "kmethod-test-pred-model": "KMETHOD_TEST_PRED_MODEL",
    "kmethod-test-pred-ref-selection": "KMETHOD_TEST_PRED_REF_SELECTION",
    "kmethod-test-pred-table": "KMETHOD_TEST_PRED_TABLE",
    "kmethod-test-results": "KMETHOD_TEST_RESULTS",
    "kmethod-validations": "KMETHOD_VALIDATIONS",
    "median-filter-button": "MEDIAN_FILTER_BUTTON",
    "median-filter-mode": "MEDIAN_FILTER_MODE",
    "median-filter-window-em": "MEDIAN_FILTER_WINDOW_EM",
    "median-filter-window-ex": "MEDIAN_FILTER_WINDOW_EX",
    "message-eem-dataset-export": "MESSAGE_EEM_DATASET_EXPORT",
    "nmf": "NMF",
    "nmf-alpha-h": "NMF_ALPHA_H",
    "nmf-alpha-w": "NMF_ALPHA_W",
    "nmf-components": "NMF_COMPONENTS",
    "nmf-eem-dataset-establishment-message": "NMF_EEM_DATASET_ESTABLISHMENT_MESSAGE",
    "nmf-eem-dataset-establishment-path-input": "NMF_EEM_DATASET_ESTABLISHMENT_PATH_INPUT",
    "nmf-eem-dataset-predict-message": "NMF_EEM_DATASET_PREDICT_MESSAGE",
    "nmf-eem-dataset-predict-path-input": "NMF_EEM_DATASET_PREDICT_PATH_INPUT",
    "nmf-establishment-corr": "NMF_ESTABLISHMENT_CORR",
    "nmf-establishment-corr-graph": "NMF_ESTABLISHMENT_CORR_GRAPH",
    "nmf-establishment-corr-indicator-selection": "NMF_ESTABLISHMENT_CORR_INDICATOR_SELECTION",
    "nmf-establishment-corr-model-selection": "NMF_ESTABLISHMENT_CORR_MODEL_SELECTION",
    "nmf-establishment-corr-ref-selection": "NMF_ESTABLISHMENT_CORR_REF_SELECTION",
    "nmf-establishment-corr-table": "NMF_ESTABLISHMENT_CORR_TABLE",
    "nmf-establishment-index-kw-mandatory": "NMF_ESTABLISHMENT_INDEX_KW_MANDATORY",
    "nmf-establishment-index-kw-optional": "NMF_ESTABLISHMENT_INDEX_KW_OPTIONAL",
    "nmf-establishment-reconstruction-error": "NMF_ESTABLISHMENT_RECONSTRUCTION_ERROR",
    "nmf-fmax": "NMF_FMAX",
    "nmf-init": "NMF_INIT",
    "nmf-l1-ratio": "NMF_L1_RATIO",
    "nmf-max-iter-als": "NMF_MAX_ITER_ALS",
    "nmf-max-iter-nnls": "NMF_MAX_ITER_NNLS",
    "nmf-models": "NMF_MODELS",
    "nmf-normalization-checkbox": "NMF_NORMALIZATION_CHECKBOX",
    "nmf-predict": "NMF_PREDICT",
    "nmf-predict-spinner": "NMF_PREDICT_SPINNER",
    "nmf-rank": "NMF_RANK",
    "nmf-results": "NMF_RESULTS",
    "nmf-solver": "NMF_SOLVER",
    "nmf-spinner": "NMF_SPINNER",
    "nmf-split-half": "NMF_SPLIT_HALF",
    "nmf-test-corr-graph": "NMF_TEST_CORR_GRAPH",
    "nmf-test-corr-indicator-selection": "NMF_TEST_CORR_INDICATOR_SELECTION",
    "nmf-test-corr-ref-selection": "NMF_TEST_CORR_REF_SELECTION",
    "nmf-test-corr-table": "NMF_TEST_CORR_TABLE",
    "nmf-test-error": "NMF_TEST_ERROR",
    "nmf-test-fmax": "NMF_TEST_FMAX",
    "nmf-test-index-kw-mandatory": "NMF_TEST_INDEX_KW_MANDATORY",
    "nmf-test-index-kw-optional": "NMF_TEST_INDEX_KW_OPTIONAL",
    "nmf-test-model-selection": "NMF_TEST_MODEL_SELECTION",
    "nmf-test-pred-graph": "NMF_TEST_PRED_GRAPH",
    "nmf-test-pred-model": "NMF_TEST_PRED_MODEL",
    "nmf-test-pred-model-selection": "NMF_TEST_PRED_MODEL_SELECTION",
    "nmf-test-pred-ref-selection": "NMF_TEST_PRED_REF_SELECTION",
    "nmf-test-pred-table": "NMF_TEST_PRED_TABLE",
    "nmf-test-results": "NMF_TEST_RESULTS",
    "nmf-validations": "NMF_VALIDATIONS",
    "parafac": "PARAFAC",
    "parafac-components": "PARAFAC_COMPONENTS",
    "parafac-core-consistency": "PARAFAC_CORE_CONSISTENCY",
    "parafac-eem-dataset-establishment-message": "PARAFAC_EEM_DATASET_ESTABLISHMENT_MESSAGE",
    "parafac-eem-dataset-establishment-path-input": "PARAFAC_EEM_DATASET_ESTABLISHMENT_PATH_INPUT",
    "parafac-eem-dataset-predict-message": "PARAFAC_EEM_DATASET_PREDICT_MESSAGE",
    "parafac-eem-dataset-predict-path-input": "PARAFAC_EEM_DATASET_PREDICT_PATH_INPUT",
    "parafac-establishment-corr": "PARAFAC_ESTABLISHMENT_CORR",
    "parafac-establishment-corr-graph": "PARAFAC_ESTABLISHMENT_CORR_GRAPH",
    "parafac-establishment-corr-indicator-selection": "PARAFAC_ESTABLISHMENT_CORR_INDICATOR_SELECTION",
    "parafac-establishment-corr-model-selection": "PARAFAC_ESTABLISHMENT_CORR_MODEL_SELECTION",
    "parafac-establishment-corr-ref-selection": "PARAFAC_ESTABLISHMENT_CORR_REF_SELECTION",
    "parafac-establishment-corr-table": "PARAFAC_ESTABLISHMENT_CORR_TABLE",
    "parafac-establishment-index-kw-mandatory": "PARAFAC_ESTABLISHMENT_INDEX_KW_MANDATORY",
    "parafac-establishment-index-kw-optional": "PARAFAC_ESTABLISHMENT_INDEX_KW_OPTIONAL",
    "parafac-establishment-reconstruction-error": "PARAFAC_ESTABLISHMENT_RECONSTRUCTION_ERROR",
    "parafac-fmax": "PARAFAC_FMAX",
    "parafac-init-method": "PARAFAC_INIT_METHOD",
    "parafac-leverage": "PARAFAC_LEVERAGE",
    "parafac-loadings": "PARAFAC_LOADINGS",
    "parafac-models": "PARAFAC_MODELS",
    "parafac-nn-checkbox": "PARAFAC_NN_CHECKBOX",
    "parafac-predict": "PARAFAC_PREDICT",
    "parafac-predict-spinner": "PARAFAC_PREDICT_SPINNER",
    "parafac-rank": "PARAFAC_RANK",
    "parafac-results": "PARAFAC_RESULTS",
    "parafac-scores": "PARAFAC_SCORES",
    "parafac-solver": "PARAFAC_SOLVER",
    "parafac-split-half": "PARAFAC_SPLIT_HALF",
    "parafac-test-corr-graph": "PARAFAC_TEST_CORR_GRAPH",
    "parafac-test-corr-indicator-selection": "PARAFAC_TEST_CORR_INDICATOR_SELECTION",
    "parafac-test-corr-ref-selection": "PARAFAC_TEST_CORR_REF_SELECTION",
    "parafac-test-corr-table": "PARAFAC_TEST_CORR_TABLE",
    "parafac-test-error": "PARAFAC_TEST_ERROR",
    "parafac-test-fmax": "PARAFAC_TEST_FMAX",
    "parafac-test-index-kw-mandatory": "PARAFAC_TEST_INDEX_KW_MANDATORY",
    "parafac-test-index-kw-optional": "PARAFAC_TEST_INDEX_KW_OPTIONAL",
    "parafac-test-model-selection": "PARAFAC_TEST_MODEL_SELECTION",
    "parafac-test-pred-graph": "PARAFAC_TEST_PRED_GRAPH",
    "parafac-test-pred-model": "PARAFAC_TEST_PRED_MODEL",
    "parafac-test-pred-model-selection": "PARAFAC_TEST_PRED_MODEL_SELECTION",
    "parafac-test-pred-ref-selection": "PARAFAC_TEST_PRED_REF_SELECTION",
    "parafac-test-pred-table": "PARAFAC_TEST_PRED_TABLE",
    "parafac-test-results": "PARAFAC_TEST_RESULTS",
    "parafac-test-score": "PARAFAC_TEST_SCORE",
    "parafac-tf-checkbox": "PARAFAC_TF_CHECKBOX",
    "parafac-validations": "PARAFAC_VALIDATIONS",
    "parafac-variance-explained": "PARAFAC_VARIANCE_EXPLAINED",
    "path-reference": "PATH_REFERENCE",
    "pp-eem-dataset-establishment-message": "PP_EEM_DATASET_ESTABLISHMENT_MESSAGE",
    "pp-eem-dataset-establishment-path-input": "PP_EEM_DATASET_ESTABLISHMENT_PATH_INPUT",
    "pp-eem-dataset-predict-message": "PP_EEM_DATASET_PREDICT_MESSAGE",
    "pp-eem-dataset-predict-path-input": "PP_EEM_DATASET_PREDICT_PATH_INPUT",
    "pp-emission": "PP_EMISSION",
    "pp-establishment-corr": "PP_ESTABLISHMENT_CORR",
    "pp-establishment-corr-graph": "PP_ESTABLISHMENT_CORR_GRAPH",
    "pp-establishment-corr-ref-selection": "PP_ESTABLISHMENT_CORR_REF_SELECTION",
    "pp-establishment-corr-table": "PP_ESTABLISHMENT_CORR_TABLE",
    "pp-establishment-index-kw-mandatory": "PP_ESTABLISHMENT_INDEX_KW_MANDATORY",
    "pp-establishment-index-kw-optional": "PP_ESTABLISHMENT_INDEX_KW_OPTIONAL",
    "pp-excitation": "PP_EXCITATION",
    "pp-intensities": "PP_INTENSITIES",
    "pp-model": "PP_MODEL",
    "pp-predict": "PP_PREDICT",
    "pp-predict-spinner": "PP_PREDICT_SPINNER",
    "pp-results": "PP_RESULTS",
    "pp-test-corr-graph": "PP_TEST_CORR_GRAPH",
    "pp-test-corr-indicator-selection": "PP_TEST_CORR_INDICATOR_SELECTION",
    "pp-test-corr-ref-selection": "PP_TEST_CORR_REF_SELECTION",
    "pp-test-corr-table": "PP_TEST_CORR_TABLE",
    "pp-test-error": "PP_TEST_ERROR",
    "pp-test-index-kw-mandatory": "PP_TEST_INDEX_KW_MANDATORY",
    "pp-test-index-kw-optional": "PP_TEST_INDEX_KW_OPTIONAL",
    "pp-test-intensities": "PP_TEST_INTENSITIES",
    "pp-test-pred-graph": "PP_TEST_PRED_GRAPH",
    "pp-test-pred-model": "PP_TEST_PRED_MODEL",
    "pp-test-pred-ref-selection": "PP_TEST_PRED_REF_SELECTION",
    "pp-test-pred-table": "PP_TEST_PRED_TABLE",
    "pp-test-results": "PP_TEST_RESULTS",
    "predict-kmethod-model": "PREDICT_KMETHOD_MODEL",
    "predict-nmf-model": "PREDICT_NMF_MODEL",
    "predict-parafac-model": "PREDICT_PARAFAC_MODEL",
    "predict-pp-model": "PREDICT_PP_MODEL",
    "predict-ri-model": "PREDICT_RI_MODEL",
    "pre-processed-eem": "PRE_PROCESSED_EEM",
    "raman-button": "RAMAN_BUTTON",
    "raman-dimension": "RAMAN_DIMENSION",
    "raman-methods": "RAMAN_METHODS",
    "raman-width": "RAMAN_WIDTH",
    "rayleigh-button": "RAYLEIGH_BUTTON",
    "rayleigh-o1-dimension": "RAYLEIGH_O1_DIMENSION",
    "rayleigh-o1-methods": "RAYLEIGH_O1_METHODS",
    "rayleigh-o1-width": "RAYLEIGH_O1_WIDTH",
    "rayleigh-o2-dimension": "RAYLEIGH_O2_DIMENSION",
    "rayleigh-o2-methods": "RAYLEIGH_O2_METHODS",
    "rayleigh-o2-width": "RAYLEIGH_O2_WIDTH",
    "ri-eem-dataset-establishment-message": "RI_EEM_DATASET_ESTABLISHMENT_MESSAGE",
    "ri-eem-dataset-establishment-path-input": "RI_EEM_DATASET_ESTABLISHMENT_PATH_INPUT",
    "ri-eem-dataset-predict-message": "RI_EEM_DATASET_PREDICT_MESSAGE",
    "ri-eem-dataset-predict-path-input": "RI_EEM_DATASET_PREDICT_PATH_INPUT",
    "ri-em-max": "RI_EM_MAX",
    "ri-em-min": "RI_EM_MIN",
    "ri-establishment-corr": "RI_ESTABLISHMENT_CORR",
    "ri-establishment-corr-graph": "RI_ESTABLISHMENT_CORR_GRAPH",
    "ri-establishment-corr-ref-selection": "RI_ESTABLISHMENT_CORR_REF_SELECTION",
    "ri-establishment-corr-table": "RI_ESTABLISHMENT_CORR_TABLE",
    "ri-establishment-index-kw-mandatory": "RI_ESTABLISHMENT_INDEX_KW_MANDATORY",
    "ri-establishment-index-kw-optional": "RI_ESTABLISHMENT_INDEX_KW_OPTIONAL",
    "ri-ex-max": "RI_EX_MAX",
    "ri-ex-min": "RI_EX_MIN",
    "ri-intensities": "RI_INTENSITIES",
    "ri-model": "RI_MODEL",
    "ri-predict": "RI_PREDICT",
    "ri-predict-spinner": "RI_PREDICT_SPINNER",
    "ri-results": "RI_RESULTS",
    "ri-test-corr-graph": "RI_TEST_CORR_GRAPH",
    "ri-test-corr-indicator-selection": "RI_TEST_CORR_INDICATOR_SELECTION",
    "ri-test-corr-ref-selection": "RI_TEST_CORR_REF_SELECTION",
    "ri-test-corr-table": "RI_TEST_CORR_TABLE",
    "ri-test-error": "RI_TEST_ERROR",
    "ri-test-index-kw-mandatory": "RI_TEST_INDEX_KW_MANDATORY",
    "ri-test-index-kw-optional": "RI_TEST_INDEX_KW_OPTIONAL",
    "ri-test-intensities": "RI_TEST_INTENSITIES",
    "ri-test-pred-graph": "RI_TEST_PRED_GRAPH",
    "ri-test-pred-model": "RI_TEST_PRED_MODEL",
    "ri-test-pred-ref-selection": "RI_TEST_PRED_REF_SELECTION",
    "ri-test-pred-table": "RI_TEST_PRED_TABLE",
    "ri-test-results": "RI_TEST_RESULTS",
    "rsu-display": "RSU_DISPLAY",
    "su-button": "SU_BUTTON",
    "su-emission-width": "SU_EMISSION_WIDTH",
    "su-excitation": "SU_EXCITATION",
    "su-normalization-factor": "SU_NORMALIZATION_FACTOR",
    "tabs-content": "TABS_CONTENT",
    "timestamp-checkbox": "TIMESTAMP_CHECKBOX",
    "timestamp-format": "TIMESTAMP_FORMAT",
}

CONST_TO_ID = {
    "ABSORBANCE_GRAPH": "absorbance-graph",
    "ABS_DATA_FORMAT": "abs-data-format",
    "ALIGN_EXEM": "align-exem",
    "BUILD_EEM_DATASET": "build-eem-dataset",
    "BUILD_EEM_DATASET_SPINNER": "build-eem-dataset-spinner",
    "BUILD_KMETHOD_CLUSTERING": "build-kmethod-clustering",
    "BUILD_KMETHOD_CONSENSUS": "build-kmethod-consensus",
    "BUILD_NMF_MODEL": "build-nmf-model",
    "BUILD_PARAFAC_MODEL": "build-parafac-model",
    "BUILD_PARAFAC_SPINNER": "build-parafac-spinner",
    "BUILD_PP_MODEL": "build-pp-model",
    "BUILD_PP_SPINNER": "build-pp-spinner",
    "BUILD_RI_MODEL": "build-ri-model",
    "BUILD_RI_SPINNER": "build-ri-spinner",
    "EEM_DATASET": "eem-dataset",
    "EEM_DATASET_EXPORT_FORMAT": "eem-dataset-export-format",
    "EEM_DATA_FORMAT": "eem-data-format",
    "EEM_GRAPH": "eem-graph",
    "EEM_GRAPH_OPTIONS": "eem-graph-options",
    "EEM_PEAK_PICKING": "eem-peak-picking",
    "EEM_PRE_PROCESSING": "eem-pre-processing",
    "EEM_REGIONAL_INTEGRATION": "eem-regional-integration",
    "EMISSION_WAVELENGTH_MAX": "emission-wavelength-max",
    "EMISSION_WAVELENGTH_MIN": "emission-wavelength-min",
    "EXCITATION_WAVELENGTH_MAX": "excitation-wavelength-max",
    "EXCITATION_WAVELENGTH_MIN": "excitation-wavelength-min",
    "EXPORT_EEM_DATASET": "export-eem-dataset",
    "EXPORT_EEM_DATASET_SPINNER": "export-eem-dataset-spinner",
    "FILENAME_EXPORT_EEM_DATASET": "filename-export-eem-dataset",
    "FILENAME_SAMPLE_DROPDOWN": "filename-sample-dropdown",
    "FILE_KEYWORD_ABSORBANCE": "file-keyword-absorbance",
    "FILE_KEYWORD_BLANK": "file-keyword-blank",
    "FILE_KEYWORD_MANDATORY": "file-keyword-mandatory",
    "FILE_KEYWORD_OPTIONAL": "file-keyword-optional",
    "FILE_KEYWORD_SAMPLE": "file-keyword-sample",
    "FLUORESCENCE_INTENSITY_MAX": "fluorescence-intensity-max",
    "FLUORESCENCE_INTENSITY_MIN": "fluorescence-intensity-min",
    "FOLDER_PATH_EXPORT_EEM_DATASET": "folder-path-export-eem-dataset",
    "FOLDER_PATH_INPUT": "folder-path-input",
    "GAUSSIAN_BUTTON": "gaussian-button",
    "GAUSSIAN_SIGMA": "gaussian-sigma",
    "GAUSSIAN_TRUNCATE": "gaussian-truncate",
    "HELP_DATA_FORMAT": "help-data-format",
    "HELP_POP": "help-pop",
    "HELP_QM": "help-qm",
    "HOMEPAGE": "homepage",
    "IFE_BUTTON": "ife-button",
    "IFE_METHODS": "ife-methods",
    "INDEX_POS_LEFT": "index-pos-left",
    "INDEX_POS_RIGHT": "index-pos-right",
    "INFO_EEM_DATASET": "info-eem-dataset",
    "KMETHOD": "kmethod",
    "KMETHOD_BASE_CLUSTERING_PARAMETERS": "kmethod-base-clustering-parameters",
    "KMETHOD_BASE_MODEL": "kmethod-base-model",
    "KMETHOD_BASE_MODEL_MESSAGE": "kmethod-base-model-message",
    "KMETHOD_CLUSTER_EXPORT": "kmethod-cluster-export",
    "KMETHOD_CLUSTER_EXPORT_MODEL_SELECTION": "kmethod-cluster-export-model-selection",
    "KMETHOD_CLUSTER_FROM_FILE_CHECKBOX": "kmethod-cluster-from-file-checkbox",
    "KMETHOD_COMPONENTS": "kmethod-components",
    "KMETHOD_CONSENSUS_CONVERSION": "kmethod-consensus-conversion",
    "KMETHOD_CONSENSUS_MATRIX": "kmethod-consensus-matrix",
    "KMETHOD_CONSENSUS_MATRIX_DATA": "kmethod-consensus-matrix-data",
    "KMETHOD_CONVERGENCE_TOLERANCE": "kmethod-convergence-tolerance",
    "KMETHOD_DENDROGRAM": "kmethod-dendrogram",
    "KMETHOD_EEM_DATASET_ESTABLISH": "kmethod-eem-dataset-establish",
    "KMETHOD_EEM_DATASET_ESTABLISHMENT_MESSAGE": "kmethod-eem-dataset-establishment-message",
    "KMETHOD_EEM_DATASET_ESTABLISHMENT_PATH_INPUT": "kmethod-eem-dataset-establishment-path-input",
    "KMETHOD_EEM_DATASET_PREDICT_MESSAGE": "kmethod-eem-dataset-predict-message",
    "KMETHOD_EEM_DATASET_PREDICT_PATH_INPUT": "kmethod-eem-dataset-predict-path-input",
    "KMETHOD_ELIMINATION": "kmethod-elimination",
    "KMETHOD_ERROR_HISTORY": "kmethod-error-history",
    "KMETHOD_ESTABLISHMENT_COMPONENTS_CLUSTER_SELECTION": "kmethod-establishment-components-cluster-selection",
    "KMETHOD_ESTABLISHMENT_COMPONENTS_GRAPH": "kmethod-establishment-components-graph",
    "KMETHOD_ESTABLISHMENT_COMPONENTS_MODEL_SELECTION": "kmethod-establishment-components-model-selection",
    "KMETHOD_ESTABLISHMENT_CORR": "kmethod-establishment-corr",
    "KMETHOD_ESTABLISHMENT_CORR_GRAPH": "kmethod-establishment-corr-graph",
    "KMETHOD_ESTABLISHMENT_CORR_INDICATOR_SELECTION": "kmethod-establishment-corr-indicator-selection",
    "KMETHOD_ESTABLISHMENT_CORR_MODEL_SELECTION": "kmethod-establishment-corr-model-selection",
    "KMETHOD_ESTABLISHMENT_CORR_REF_SELECTION": "kmethod-establishment-corr-ref-selection",
    "KMETHOD_ESTABLISHMENT_CORR_TABLE": "kmethod-establishment-corr-table",
    "KMETHOD_ESTABLISHMENT_INDEX_KW_MANDATORY": "kmethod-establishment-index-kw-mandatory",
    "KMETHOD_ESTABLISHMENT_INDEX_KW_OPTIONAL": "kmethod-establishment-index-kw-optional",
    "KMETHOD_FMAX": "kmethod-fmax",
    "KMETHOD_LABEL_HISTORY": "kmethod-label-history",
    "KMETHOD_MODELS": "kmethod-models",
    "KMETHOD_NUM_BASE_CLUSTERINGS": "kmethod-num-base-clusterings",
    "KMETHOD_NUM_FINAL_CLUSTERS": "kmethod-num-final-clusters",
    "KMETHOD_NUM_INIT_SPLITS": "kmethod-num-init-splits",
    "KMETHOD_NUM_ITERATIONS": "kmethod-num-iterations",
    "KMETHOD_PREDICT": "kmethod-predict",
    "KMETHOD_PREDICT_SPINNER": "kmethod-predict-spinner",
    "KMETHOD_RANK": "kmethod-rank",
    "KMETHOD_RECONSTRUCTION_ERROR_REDUCTION": "kmethod-reconstruction-error-reduction",
    "KMETHOD_RESULTS1": "kmethod-results1",
    "KMETHOD_RESULTS2": "kmethod-results2",
    "KMETHOD_SILHOUETTE_SCORE": "kmethod-silhouette-score",
    "KMETHOD_SORTED_CONSENSUS_MATRIX": "kmethod-sorted-consensus-matrix",
    "KMETHOD_STEP1_SPINNER": "kmethod-step1-spinner",
    "KMETHOD_STEP2_SPINNER": "kmethod-step2-spinner",
    "KMETHOD_SUBSAMPLING_PORTION": "kmethod-subsampling-portion",
    "KMETHOD_TEST_CORR_GRAPH": "kmethod-test-corr-graph",
    "KMETHOD_TEST_CORR_INDICATOR_SELECTION": "kmethod-test-corr-indicator-selection",
    "KMETHOD_TEST_CORR_REF_SELECTION": "kmethod-test-corr-ref-selection",
    "KMETHOD_TEST_CORR_TABLE": "kmethod-test-corr-table",
    "KMETHOD_TEST_ERROR": "kmethod-test-error",
    "KMETHOD_TEST_FMAX": "kmethod-test-fmax",
    "KMETHOD_TEST_INDEX_KW_MANDATORY": "kmethod-test-index-kw-mandatory",
    "KMETHOD_TEST_INDEX_KW_OPTIONAL": "kmethod-test-index-kw-optional",
    "KMETHOD_TEST_MODEL_SELECTION": "kmethod-test-model-selection",
    "KMETHOD_TEST_PRED_GRAPH": "kmethod-test-pred-graph",
    "KMETHOD_TEST_PRED_MODEL": "kmethod-test-pred-model",
    "KMETHOD_TEST_PRED_REF_SELECTION": "kmethod-test-pred-ref-selection",
    "KMETHOD_TEST_PRED_TABLE": "kmethod-test-pred-table",
    "KMETHOD_TEST_RESULTS": "kmethod-test-results",
    "KMETHOD_VALIDATIONS": "kmethod-validations",
    "MEDIAN_FILTER_BUTTON": "median-filter-button",
    "MEDIAN_FILTER_MODE": "median-filter-mode",
    "MEDIAN_FILTER_WINDOW_EM": "median-filter-window-em",
    "MEDIAN_FILTER_WINDOW_EX": "median-filter-window-ex",
    "MESSAGE_EEM_DATASET_EXPORT": "message-eem-dataset-export",
    "NMF": "nmf",
    "NMF_ALPHA_H": "nmf-alpha-h",
    "NMF_ALPHA_W": "nmf-alpha-w",
    "NMF_COMPONENTS": "nmf-components",
    "NMF_EEM_DATASET_ESTABLISHMENT_MESSAGE": "nmf-eem-dataset-establishment-message",
    "NMF_EEM_DATASET_ESTABLISHMENT_PATH_INPUT": "nmf-eem-dataset-establishment-path-input",
    "NMF_EEM_DATASET_PREDICT_MESSAGE": "nmf-eem-dataset-predict-message",
    "NMF_EEM_DATASET_PREDICT_PATH_INPUT": "nmf-eem-dataset-predict-path-input",
    "NMF_ESTABLISHMENT_CORR": "nmf-establishment-corr",
    "NMF_ESTABLISHMENT_CORR_GRAPH": "nmf-establishment-corr-graph",
    "NMF_ESTABLISHMENT_CORR_INDICATOR_SELECTION": "nmf-establishment-corr-indicator-selection",
    "NMF_ESTABLISHMENT_CORR_MODEL_SELECTION": "nmf-establishment-corr-model-selection",
    "NMF_ESTABLISHMENT_CORR_REF_SELECTION": "nmf-establishment-corr-ref-selection",
    "NMF_ESTABLISHMENT_CORR_TABLE": "nmf-establishment-corr-table",
    "NMF_ESTABLISHMENT_INDEX_KW_MANDATORY": "nmf-establishment-index-kw-mandatory",
    "NMF_ESTABLISHMENT_INDEX_KW_OPTIONAL": "nmf-establishment-index-kw-optional",
    "NMF_ESTABLISHMENT_RECONSTRUCTION_ERROR": "nmf-establishment-reconstruction-error",
    "NMF_FMAX": "nmf-fmax",
    "NMF_INIT": "nmf-init",
    "NMF_L1_RATIO": "nmf-l1-ratio",
    "NMF_MAX_ITER_ALS": "nmf-max-iter-als",
    "NMF_MAX_ITER_NNLS": "nmf-max-iter-nnls",
    "NMF_MODELS": "nmf-models",
    "NMF_NORMALIZATION_CHECKBOX": "nmf-normalization-checkbox",
    "NMF_PREDICT": "nmf-predict",
    "NMF_PREDICT_SPINNER": "nmf-predict-spinner",
    "NMF_RANK": "nmf-rank",
    "NMF_RESULTS": "nmf-results",
    "NMF_SOLVER": "nmf-solver",
    "NMF_SPINNER": "nmf-spinner",
    "NMF_SPLIT_HALF": "nmf-split-half",
    "NMF_TEST_CORR_GRAPH": "nmf-test-corr-graph",
    "NMF_TEST_CORR_INDICATOR_SELECTION": "nmf-test-corr-indicator-selection",
    "NMF_TEST_CORR_REF_SELECTION": "nmf-test-corr-ref-selection",
    "NMF_TEST_CORR_TABLE": "nmf-test-corr-table",
    "NMF_TEST_ERROR": "nmf-test-error",
    "NMF_TEST_FMAX": "nmf-test-fmax",
    "NMF_TEST_INDEX_KW_MANDATORY": "nmf-test-index-kw-mandatory",
    "NMF_TEST_INDEX_KW_OPTIONAL": "nmf-test-index-kw-optional",
    "NMF_TEST_MODEL_SELECTION": "nmf-test-model-selection",
    "NMF_TEST_PRED_GRAPH": "nmf-test-pred-graph",
    "NMF_TEST_PRED_MODEL": "nmf-test-pred-model",
    "NMF_TEST_PRED_MODEL_SELECTION": "nmf-test-pred-model-selection",
    "NMF_TEST_PRED_REF_SELECTION": "nmf-test-pred-ref-selection",
    "NMF_TEST_PRED_TABLE": "nmf-test-pred-table",
    "NMF_TEST_RESULTS": "nmf-test-results",
    "NMF_VALIDATIONS": "nmf-validations",
    "PARAFAC": "parafac",
    "PARAFAC_COMPONENTS": "parafac-components",
    "PARAFAC_CORE_CONSISTENCY": "parafac-core-consistency",
    "PARAFAC_EEM_DATASET_ESTABLISHMENT_MESSAGE": "parafac-eem-dataset-establishment-message",
    "PARAFAC_EEM_DATASET_ESTABLISHMENT_PATH_INPUT": "parafac-eem-dataset-establishment-path-input",
    "PARAFAC_EEM_DATASET_PREDICT_MESSAGE": "parafac-eem-dataset-predict-message",
    "PARAFAC_EEM_DATASET_PREDICT_PATH_INPUT": "parafac-eem-dataset-predict-path-input",
    "PARAFAC_ESTABLISHMENT_CORR": "parafac-establishment-corr",
    "PARAFAC_ESTABLISHMENT_CORR_GRAPH": "parafac-establishment-corr-graph",
    "PARAFAC_ESTABLISHMENT_CORR_INDICATOR_SELECTION": "parafac-establishment-corr-indicator-selection",
    "PARAFAC_ESTABLISHMENT_CORR_MODEL_SELECTION": "parafac-establishment-corr-model-selection",
    "PARAFAC_ESTABLISHMENT_CORR_REF_SELECTION": "parafac-establishment-corr-ref-selection",
    "PARAFAC_ESTABLISHMENT_CORR_TABLE": "parafac-establishment-corr-table",
    "PARAFAC_ESTABLISHMENT_INDEX_KW_MANDATORY": "parafac-establishment-index-kw-mandatory",
    "PARAFAC_ESTABLISHMENT_INDEX_KW_OPTIONAL": "parafac-establishment-index-kw-optional",
    "PARAFAC_ESTABLISHMENT_RECONSTRUCTION_ERROR": "parafac-establishment-reconstruction-error",
    "PARAFAC_FMAX": "parafac-fmax",
    "PARAFAC_INIT_METHOD": "parafac-init-method",
    "PARAFAC_LEVERAGE": "parafac-leverage",
    "PARAFAC_LOADINGS": "parafac-loadings",
    "PARAFAC_MODELS": "parafac-models",
    "PARAFAC_NN_CHECKBOX": "parafac-nn-checkbox",
    "PARAFAC_PREDICT": "parafac-predict",
    "PARAFAC_PREDICT_SPINNER": "parafac-predict-spinner",
    "PARAFAC_RANK": "parafac-rank",
    "PARAFAC_RESULTS": "parafac-results",
    "PARAFAC_SCORES": "parafac-scores",
    "PARAFAC_SOLVER": "parafac-solver",
    "PARAFAC_SPLIT_HALF": "parafac-split-half",
    "PARAFAC_TEST_CORR_GRAPH": "parafac-test-corr-graph",
    "PARAFAC_TEST_CORR_INDICATOR_SELECTION": "parafac-test-corr-indicator-selection",
    "PARAFAC_TEST_CORR_REF_SELECTION": "parafac-test-corr-ref-selection",
    "PARAFAC_TEST_CORR_TABLE": "parafac-test-corr-table",
    "PARAFAC_TEST_ERROR": "parafac-test-error",
    "PARAFAC_TEST_FMAX": "parafac-test-fmax",
    "PARAFAC_TEST_INDEX_KW_MANDATORY": "parafac-test-index-kw-mandatory",
    "PARAFAC_TEST_INDEX_KW_OPTIONAL": "parafac-test-index-kw-optional",
    "PARAFAC_TEST_MODEL_SELECTION": "parafac-test-model-selection",
    "PARAFAC_TEST_PRED_GRAPH": "parafac-test-pred-graph",
    "PARAFAC_TEST_PRED_MODEL": "parafac-test-pred-model",
    "PARAFAC_TEST_PRED_MODEL_SELECTION": "parafac-test-pred-model-selection",
    "PARAFAC_TEST_PRED_REF_SELECTION": "parafac-test-pred-ref-selection",
    "PARAFAC_TEST_PRED_TABLE": "parafac-test-pred-table",
    "PARAFAC_TEST_RESULTS": "parafac-test-results",
    "PARAFAC_TEST_SCORE": "parafac-test-score",
    "PARAFAC_TF_CHECKBOX": "parafac-tf-checkbox",
    "PARAFAC_VALIDATIONS": "parafac-validations",
    "PARAFAC_VARIANCE_EXPLAINED": "parafac-variance-explained",
    "PATH_REFERENCE": "path-reference",
    "PP_EEM_DATASET_ESTABLISHMENT_MESSAGE": "pp-eem-dataset-establishment-message",
    "PP_EEM_DATASET_ESTABLISHMENT_PATH_INPUT": "pp-eem-dataset-establishment-path-input",
    "PP_EEM_DATASET_PREDICT_MESSAGE": "pp-eem-dataset-predict-message",
    "PP_EEM_DATASET_PREDICT_PATH_INPUT": "pp-eem-dataset-predict-path-input",
    "PP_EMISSION": "pp-emission",
    "PP_ESTABLISHMENT_CORR": "pp-establishment-corr",
    "PP_ESTABLISHMENT_CORR_GRAPH": "pp-establishment-corr-graph",
    "PP_ESTABLISHMENT_CORR_REF_SELECTION": "pp-establishment-corr-ref-selection",
    "PP_ESTABLISHMENT_CORR_TABLE": "pp-establishment-corr-table",
    "PP_ESTABLISHMENT_INDEX_KW_MANDATORY": "pp-establishment-index-kw-mandatory",
    "PP_ESTABLISHMENT_INDEX_KW_OPTIONAL": "pp-establishment-index-kw-optional",
    "PP_EXCITATION": "pp-excitation",
    "PP_INTENSITIES": "pp-intensities",
    "PP_MODEL": "pp-model",
    "PP_PREDICT": "pp-predict",
    "PP_PREDICT_SPINNER": "pp-predict-spinner",
    "PP_RESULTS": "pp-results",
    "PP_TEST_CORR_GRAPH": "pp-test-corr-graph",
    "PP_TEST_CORR_INDICATOR_SELECTION": "pp-test-corr-indicator-selection",
    "PP_TEST_CORR_REF_SELECTION": "pp-test-corr-ref-selection",
    "PP_TEST_CORR_TABLE": "pp-test-corr-table",
    "PP_TEST_ERROR": "pp-test-error",
    "PP_TEST_INDEX_KW_MANDATORY": "pp-test-index-kw-mandatory",
    "PP_TEST_INDEX_KW_OPTIONAL": "pp-test-index-kw-optional",
    "PP_TEST_INTENSITIES": "pp-test-intensities",
    "PP_TEST_PRED_GRAPH": "pp-test-pred-graph",
    "PP_TEST_PRED_MODEL": "pp-test-pred-model",
    "PP_TEST_PRED_REF_SELECTION": "pp-test-pred-ref-selection",
    "PP_TEST_PRED_TABLE": "pp-test-pred-table",
    "PP_TEST_RESULTS": "pp-test-results",
    "PREDICT_KMETHOD_MODEL": "predict-kmethod-model",
    "PREDICT_NMF_MODEL": "predict-nmf-model",
    "PREDICT_PARAFAC_MODEL": "predict-parafac-model",
    "PREDICT_PP_MODEL": "predict-pp-model",
    "PREDICT_RI_MODEL": "predict-ri-model",
    "PRE_PROCESSED_EEM": "pre-processed-eem",
    "RAMAN_BUTTON": "raman-button",
    "RAMAN_DIMENSION": "raman-dimension",
    "RAMAN_METHODS": "raman-methods",
    "RAMAN_WIDTH": "raman-width",
    "RAYLEIGH_BUTTON": "rayleigh-button",
    "RAYLEIGH_O1_DIMENSION": "rayleigh-o1-dimension",
    "RAYLEIGH_O1_METHODS": "rayleigh-o1-methods",
    "RAYLEIGH_O1_WIDTH": "rayleigh-o1-width",
    "RAYLEIGH_O2_DIMENSION": "rayleigh-o2-dimension",
    "RAYLEIGH_O2_METHODS": "rayleigh-o2-methods",
    "RAYLEIGH_O2_WIDTH": "rayleigh-o2-width",
    "RI_EEM_DATASET_ESTABLISHMENT_MESSAGE": "ri-eem-dataset-establishment-message",
    "RI_EEM_DATASET_ESTABLISHMENT_PATH_INPUT": "ri-eem-dataset-establishment-path-input",
    "RI_EEM_DATASET_PREDICT_MESSAGE": "ri-eem-dataset-predict-message",
    "RI_EEM_DATASET_PREDICT_PATH_INPUT": "ri-eem-dataset-predict-path-input",
    "RI_EM_MAX": "ri-em-max",
    "RI_EM_MIN": "ri-em-min",
    "RI_ESTABLISHMENT_CORR": "ri-establishment-corr",
    "RI_ESTABLISHMENT_CORR_GRAPH": "ri-establishment-corr-graph",
    "RI_ESTABLISHMENT_CORR_REF_SELECTION": "ri-establishment-corr-ref-selection",
    "RI_ESTABLISHMENT_CORR_TABLE": "ri-establishment-corr-table",
    "RI_ESTABLISHMENT_INDEX_KW_MANDATORY": "ri-establishment-index-kw-mandatory",
    "RI_ESTABLISHMENT_INDEX_KW_OPTIONAL": "ri-establishment-index-kw-optional",
    "RI_EX_MAX": "ri-ex-max",
    "RI_EX_MIN": "ri-ex-min",
    "RI_INTENSITIES": "ri-intensities",
    "RI_MODEL": "ri-model",
    "RI_PREDICT": "ri-predict",
    "RI_PREDICT_SPINNER": "ri-predict-spinner",
    "RI_RESULTS": "ri-results",
    "RI_TEST_CORR_GRAPH": "ri-test-corr-graph",
    "RI_TEST_CORR_INDICATOR_SELECTION": "ri-test-corr-indicator-selection",
    "RI_TEST_CORR_REF_SELECTION": "ri-test-corr-ref-selection",
    "RI_TEST_CORR_TABLE": "ri-test-corr-table",
    "RI_TEST_ERROR": "ri-test-error",
    "RI_TEST_INDEX_KW_MANDATORY": "ri-test-index-kw-mandatory",
    "RI_TEST_INDEX_KW_OPTIONAL": "ri-test-index-kw-optional",
    "RI_TEST_INTENSITIES": "ri-test-intensities",
    "RI_TEST_PRED_GRAPH": "ri-test-pred-graph",
    "RI_TEST_PRED_MODEL": "ri-test-pred-model",
    "RI_TEST_PRED_REF_SELECTION": "ri-test-pred-ref-selection",
    "RI_TEST_PRED_TABLE": "ri-test-pred-table",
    "RI_TEST_RESULTS": "ri-test-results",
    "RSU_DISPLAY": "rsu-display",
    "SU_BUTTON": "su-button",
    "SU_EMISSION_WIDTH": "su-emission-width",
    "SU_EXCITATION": "su-excitation",
    "SU_NORMALIZATION_FACTOR": "su-normalization-factor",
    "TABS_CONTENT": "tabs-content",
    "TIMESTAMP_CHECKBOX": "timestamp-checkbox",
    "TIMESTAMP_FORMAT": "timestamp-format",
}
