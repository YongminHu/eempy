from .shared import *
from ..ids import IDS


def register_callbacks(app):
    # -------------Callbacks of page #1

    #   ---------------Update file list according to input data folder path

    @app.callback(
        [
            Output(IDS.FILENAME_SAMPLE_DROPDOWN, "options"),
            Output(IDS.FILENAME_SAMPLE_DROPDOWN, "value"),
        ],
        [
            Input(IDS.FOLDER_PATH_INPUT, "value"),
            Input(IDS.FILE_KEYWORD_MANDATORY, "value"),
            Input(IDS.FILE_KEYWORD_OPTIONAL, "value"),
            Input(IDS.FILE_KEYWORD_SAMPLE, "value"),
        ],
    )
    def update_filenames(folder_path, kw_mandatory, kw_optional, kw_sample):
        if folder_path:
            try:
                filenames = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
                if kw_mandatory or kw_optional or kw_sample:
                    kw_mandatory = str_string_to_list(kw_mandatory) if kw_mandatory else []
                    kw_optional = str_string_to_list(kw_optional) if kw_optional else []
                    kw_sample = str_string_to_list(kw_sample) if kw_sample else []
                    filenames = get_filelist(folder_path, kw_mandatory + kw_sample, kw_optional)
                options = [{"label": f, "value": f} for f in filenames]
                return options, None
            except FileNotFoundError as e:
                return [f"{e}"], None
        else:
            return [], None

    #   ---------------Update Data plot with changes of parameters

    @app.callback(
        [
            Output(IDS.EEM_GRAPH, "figure"),
            Output(IDS.ABSORBANCE_GRAPH, "figure"),
            Output(IDS.RSU_DISPLAY, "children"),
        ],
        [
            Input(IDS.FOLDER_PATH_INPUT, "value"),
            Input(IDS.FILENAME_SAMPLE_DROPDOWN, "value"),
            Input(IDS.EEM_GRAPH_OPTIONS, "value"),
            Input(IDS.EEM_DATA_FORMAT, "value"),
            Input(IDS.ABS_DATA_FORMAT, "value"),
            Input(IDS.FILE_KEYWORD_SAMPLE, "value"),
            Input(IDS.FILE_KEYWORD_ABSORBANCE, "value"),
            Input(IDS.FILE_KEYWORD_BLANK, "value"),
            Input(IDS.INDEX_POS_LEFT, "value"),
            Input(IDS.INDEX_POS_RIGHT, "value"),
            Input(IDS.EXCITATION_WAVELENGTH_MIN, "value"),
            Input(IDS.EXCITATION_WAVELENGTH_MAX, "value"),
            Input(IDS.EMISSION_WAVELENGTH_MIN, "value"),
            Input(IDS.EMISSION_WAVELENGTH_MAX, "value"),
            Input(IDS.FLUORESCENCE_INTENSITY_MIN, "value"),
            Input(IDS.FLUORESCENCE_INTENSITY_MAX, "value"),
            Input(IDS.SU_BUTTON, "value"),
            Input(IDS.SU_EXCITATION, "value"),
            Input(IDS.SU_EMISSION_WIDTH, "value"),
            Input(IDS.SU_NORMALIZATION_FACTOR, "value"),
            Input(IDS.IFE_BUTTON, "value"),
            Input(IDS.IFE_METHODS, "value"),
            Input(IDS.RAMAN_BUTTON, "value"),
            Input(IDS.RAMAN_METHODS, "value"),
            Input(IDS.RAMAN_DIMENSION, "value"),
            Input(IDS.RAMAN_WIDTH, "value"),
            Input(IDS.RAYLEIGH_BUTTON, "value"),
            Input(IDS.RAYLEIGH_O1_METHODS, "value"),
            Input(IDS.RAYLEIGH_O1_DIMENSION, "value"),
            Input(IDS.RAYLEIGH_O1_WIDTH, "value"),
            Input(IDS.RAYLEIGH_O2_METHODS, "value"),
            Input(IDS.RAYLEIGH_O2_DIMENSION, "value"),
            Input(IDS.RAYLEIGH_O2_WIDTH, "value"),
            Input(IDS.GAUSSIAN_BUTTON, "value"),
            Input(IDS.GAUSSIAN_SIGMA, "value"),
            Input(IDS.GAUSSIAN_TRUNCATE, "value"),
            Input(IDS.MEDIAN_FILTER_BUTTON, "value"),
            Input(IDS.MEDIAN_FILTER_WINDOW_EX, "value"),
            Input(IDS.MEDIAN_FILTER_WINDOW_EM, "value"),
            Input(IDS.MEDIAN_FILTER_MODE, "value"),
        ],
    )
    def update_eem_plot(
            folder_path,
            file_name_sample,
            graph_options,
            eem_data_format,
            abs_data_format,
            file_kw_sample,
            file_kw_abs,
            file_kw_blank,
            index_pos_left,
            index_pos_right,
            ex_range_min,
            ex_range_max,
            em_range_min,
            em_range_max,
            intensity_range_min,
            intensity_range_max,
            su,
            su_ex,
            su_em_width,
            su_normalization_factor,
            ife,
            ife_method,
            raman,
            raman_method,
            raman_dimension,
            raman_width,
            rayleigh,
            rayleigh_o1_method,
            rayleigh_o1_dimension,
            rayleigh_o1_width,
            rayleigh_o2_method,
            rayleigh_o2_dimension,
            rayleigh_o2_width,
            gaussian,
            gaussian_sigma,
            gaussian_truncate,
            median_filter,
            median_filter_window_ex,
            median_filter_window_em,
            median_filter_mode,
    ):
        try:
            full_path_sample = os.path.join(folder_path, file_name_sample)
            intensity, ex_range, em_range, index = read_eem(
                full_path_sample,
                index_pos=(index_pos_left, index_pos_right) if (index_pos_left and index_pos_right) else None,
                data_format=eem_data_format,
            )
            # Cut EEM
            if (ex_range_min and ex_range_max) or (em_range_min and em_range_max):
                ex_range_min = ex_range_min if ex_range_min else np.min(ex_range)
                ex_range_max = ex_range_max if ex_range_max else np.max(ex_range)
                em_range_min = em_range_min if em_range_min else np.min(em_range)
                em_range_max = em_range_max if em_range_max else np.max(em_range)
                intensity, ex_range, em_range = eem_cutting(
                    intensity, ex_range, em_range, ex_range_min, ex_range_max, em_range_min, em_range_max
                )

            # Scattering unit normalization
            if file_kw_blank and su:
                file_name_blank = file_name_sample.replace(file_kw_sample, file_kw_blank)
                full_path_blank = os.path.join(folder_path, file_name_blank)
                intensity_blank, ex_range_blank, em_range_blank, _ = read_eem(full_path_blank,
                                                                              data_format=eem_data_format)
                intensity, rsu_value = eem_raman_normalization(
                    intensity,
                    blank=intensity_blank,
                    ex_range_blank=ex_range_blank,
                    em_range_blank=em_range_blank,
                    from_blank=True,
                    ex_target=su_ex,
                    bandwidth=su_em_width,
                    rsu_standard=su_normalization_factor,
                )
            else:
                rsu_value = None

            # IFE correction
            if file_kw_abs and ife:
                file_name_abs = file_name_sample.replace(file_kw_sample, file_kw_abs)
                full_path_abs = os.path.join(folder_path, file_name_abs)
                absorbance, ex_range_abs, _ = read_abs(full_path_abs, data_format=abs_data_format)
                intensity = eem_ife_correction(intensity, ex_range, em_range, absorbance, ex_range_abs)

            # Median filter
            if all([median_filter, median_filter_window_ex, median_filter_window_em, median_filter_mode]):
                intensity = eem_median_filter(
                    intensity,
                    window_size=(median_filter_window_ex, median_filter_window_em),
                    mode=median_filter_mode,
                )

            # Raman scattering removal
            if all([raman, raman_method, raman_width, raman_dimension]):
                intensity, _ = eem_raman_scattering_removal(
                    intensity,
                    ex_range,
                    em_range,
                    interpolation_method=raman_method,
                    width=raman_width,
                    interpolation_dimension=raman_dimension,
                )

            # Rayleigh scattering removal
            if all(
                    [
                        rayleigh,
                        rayleigh_o1_method,
                        rayleigh_o1_dimension,
                        rayleigh_o1_width,
                        rayleigh_o2_method,
                        rayleigh_o2_dimension,
                        rayleigh_o2_width,
                    ]
            ):
                intensity, _, _ = eem_rayleigh_scattering_removal(
                    intensity,
                    ex_range,
                    em_range,
                    width_o1=rayleigh_o1_width,
                    width_o2=rayleigh_o2_width,
                    interpolation_method_o1=rayleigh_o1_method,
                    interpolation_method_o2=rayleigh_o2_method,
                    interpolation_dimension_o1=rayleigh_o1_dimension,
                    interpolation_dimension_o2=rayleigh_o2_dimension,
                )

            # Gaussian smoothing
            if all([gaussian, gaussian_sigma, gaussian_truncate]):
                intensity = eem_gaussian_filter(intensity, gaussian_sigma, gaussian_truncate)

            # Plot EEM
            fig_eem = plot_eem(
                intensity,
                ex_range,
                em_range,
                vmin=intensity_range_min,
                vmax=intensity_range_max,
                plot_tool="plotly",
                display=False,
                auto_intensity_range=False,
                cmap="jet",
                fix_aspect_ratio=True if "aspect_one" in graph_options else False,
                rotate=True if "rotate" in graph_options else False,
                title=index if index else None,
            )

        except:
            fig_eem = go.Figure()
            fig_eem.update_layout(
                xaxis=dict(showline=False, linewidth=0, linecolor="black"),
                yaxis=dict(showline=False, linewidth=0, linecolor="black"),
                margin=dict(l=50, r=50, b=50, t=50),
            )
            fig_eem.add_annotation(
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                text="EEM file or parameters unspecified",
                showarrow=False,
                font=dict(size=16),
            )
            rsu_value = None

        try:
            file_name_abs = file_name_sample.replace(file_kw_sample, file_kw_abs)
            full_path_abs = os.path.join(folder_path, file_name_abs)
            absorbance, ex_range_abs, _ = read_abs(full_path_abs, data_format=abs_data_format)
            fig_abs = plot_abs(absorbance, ex_range_abs, figure_size=(7, 2.5), plot_tool="plotly", display=False)
        except:
            fig_abs = go.Figure()
            fig_abs.update_layout(
                xaxis=dict(showline=False, linewidth=0, linecolor="black"),
                yaxis=dict(showline=False, linewidth=0, linecolor="black"),
                width=700,
                height=200,
                margin=dict(l=50, r=50, b=50, t=50),
            )
            fig_abs.add_annotation(
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                text="Absorbance file or parameters unspecified",
                showarrow=False,
                font=dict(size=16),
            )

        return fig_eem, fig_abs, f"RSU: {rsu_value}"

    #   ---------------Build EEM dataset

    @app.callback(
        [
            Output(IDS.EEM_DATASET, "data"),
            Output(IDS.INFO_EEM_DATASET, "children"),
            Output(IDS.BUILD_EEM_DATASET_SPINNER, "children"),
        ],
        [
            Input(IDS.BUILD_EEM_DATASET, "n_clicks"),
            State(IDS.FOLDER_PATH_INPUT, "value"),
            State(IDS.EEM_DATA_FORMAT, "value"),
            State(IDS.ABS_DATA_FORMAT, "value"),
            State(IDS.FILENAME_SAMPLE_DROPDOWN, "options"),
            State(IDS.FILE_KEYWORD_SAMPLE, "value"),
            State(IDS.FILE_KEYWORD_ABSORBANCE, "value"),
            State(IDS.FILE_KEYWORD_BLANK, "value"),
            State(IDS.INDEX_POS_LEFT, "value"),
            State(IDS.INDEX_POS_RIGHT, "value"),
            State(IDS.TIMESTAMP_CHECKBOX, "value"),
            State(IDS.TIMESTAMP_FORMAT, "value"),
            State(IDS.PATH_REFERENCE, "value"),
            State(IDS.EXCITATION_WAVELENGTH_MIN, "value"),
            State(IDS.EXCITATION_WAVELENGTH_MAX, "value"),
            State(IDS.EMISSION_WAVELENGTH_MIN, "value"),
            State(IDS.EMISSION_WAVELENGTH_MAX, "value"),
            State(IDS.SU_BUTTON, "value"),
            State(IDS.SU_EXCITATION, "value"),
            State(IDS.SU_EMISSION_WIDTH, "value"),
            State(IDS.SU_NORMALIZATION_FACTOR, "value"),
            State(IDS.IFE_BUTTON, "value"),
            State(IDS.IFE_METHODS, "value"),
            State(IDS.RAMAN_BUTTON, "value"),
            State(IDS.RAMAN_METHODS, "value"),
            State(IDS.RAMAN_DIMENSION, "value"),
            State(IDS.RAMAN_WIDTH, "value"),
            State(IDS.RAYLEIGH_BUTTON, "value"),
            State(IDS.RAYLEIGH_O1_METHODS, "value"),
            State(IDS.RAYLEIGH_O1_DIMENSION, "value"),
            State(IDS.RAYLEIGH_O1_WIDTH, "value"),
            State(IDS.RAYLEIGH_O2_METHODS, "value"),
            State(IDS.RAYLEIGH_O2_DIMENSION, "value"),
            State(IDS.RAYLEIGH_O2_WIDTH, "value"),
            State(IDS.GAUSSIAN_BUTTON, "value"),
            State(IDS.GAUSSIAN_SIGMA, "value"),
            State(IDS.GAUSSIAN_TRUNCATE, "value"),
            State(IDS.MEDIAN_FILTER_BUTTON, "value"),
            State(IDS.MEDIAN_FILTER_WINDOW_EX, "value"),
            State(IDS.MEDIAN_FILTER_WINDOW_EM, "value"),
            State(IDS.MEDIAN_FILTER_MODE, "value"),
            State(IDS.ALIGN_EXEM, "value"),
        ],
    )
    def on_build_eem_dataset(
            n_clicks,
            folder_path,
            eem_data_format,
            abs_data_format,
            file_name_sample_options,
            file_kw_sample,
            file_kw_abs,
            file_kw_blank,
            index_pos_left,
            index_pos_right,
            timestamp,
            timestamp_format,
            reference_path,
            ex_range_min,
            ex_range_max,
            em_range_min,
            em_range_max,
            su,
            su_ex,
            su_em_width,
            su_normalization_factor,
            ife,
            ife_method,
            raman,
            raman_method,
            raman_dimension,
            raman_width,
            rayleigh,
            rayleigh_o1_method,
            rayleigh_o1_dimension,
            rayleigh_o1_width,
            rayleigh_o2_method,
            rayleigh_o2_dimension,
            rayleigh_o2_width,
            gaussian,
            gaussian_sigma,
            gaussian_truncate,
            median_filter,
            median_filter_ex,
            median_filter_em,
            median_filter_mode,
            align_exem,
    ):
        if n_clicks is None:
            return None, None, "Build"
        try:
            file_name_sample_options = file_name_sample_options or {}
            file_name_sample_list = [f["value"] for f in file_name_sample_options]
            if index_pos_left and index_pos_right:
                index = (index_pos_left, index_pos_right)
            eem_stack, ex_range, em_range, indexes = read_eem_dataset(
                folder_path=folder_path,
                mandatory_keywords=[],
                optional_keywords=[],
                data_format=eem_data_format,
                index_pos=(index_pos_left, index_pos_right) if index_pos_left and index_pos_right else None,
                custom_filename_list=file_name_sample_list,
                wavelength_alignment=True if align_exem else False,
                interpolation_method="linear",
                as_timestamp=True if "timestamp" in timestamp else False,
                timestamp_format=timestamp_format,
            )
        except (UnboundLocalError, IndexError, TypeError) as e:
            error_message = (
                "EEM dataset building failed. Possible causes: (1) There are non-EEM files mixed in the EEM "
                "data file list. Please check data folder path and file filtering keywords settings. "
                "(2) There are necessary parameter boxes that has not been filled in. Please check the "
                "parameter boxes. "
                "(3) The Ex/Em ranges/intervals are different between EEMs, make sure you select the "
                "'Align Ex/Em' checkbox."
            )
            return None, error_message, "Build"

        steps_track = []
        if reference_path is not None and reference_path != []:
            if os.path.exists(reference_path):
                if reference_path.endswith(".csv"):
                    refs_from_file = pd.read_csv(reference_path, index_col=0, header=0)
                elif reference_path.endswith(".xlsx"):
                    refs_from_file = pd.read_excel(reference_path, index_col=0, header=0)
                else:
                    return None, ("Unsupported file format. Please provide a .csv or .xlsx file."), "build"

                if index_pos_left and index_pos_right:
                    extra_indices = [i for i in refs_from_file.index if i not in indexes]
                    missing_indices = [i for i in indexes if i not in refs_from_file.index]
                    if extra_indices or missing_indices:
                        steps_track += [
                            "Warning: indices of EEM dataset and reference file are \nnot exactly the "
                            "same. The reference value of samples \nwith unmatched indices would be "
                            "set as NaN.\n"
                        ]
                    refs = np.array(
                        [
                            refs_from_file.loc[indexes[i]] if indexes[i] in refs_from_file.index
                            else np.full(shape=(refs_from_file.shape[1]), fill_value=np.nan)
                            for i in range(len(indexes))
                        ]
                    )
                    refs = pd.DataFrame(refs, index=indexes, columns=refs_from_file.columns)
                else:
                    if refs_from_file.shape[0] != len(indexes):
                        return None, (
                            "Error: number of samples in reference file is not the same as the EEM dataset. This error "
                            "occurs also when index starting/ending positions are not specified."
                        ), "build"
                    refs = refs_from_file
            else:
                return None, ("Error: No such file or directory: " + reference_path), "Build"
        else:
            refs = None

        steps_track += [
            "EEM dataset building successful!\n",
            f"Number of EEMs: {eem_stack.shape[0]}\n",
            "Pre-processing steps implemented:\n",
        ]

        eem_dataset = EEMDataset(eem_stack, ex_range, em_range, index=indexes, ref=refs)

        if any(
                [
                    np.min(ex_range) != ex_range_min,
                    np.max(ex_range) != ex_range_max,
                    np.min(em_range) != em_range_min,
                    np.max(em_range) != em_range_max,
                ]
        ):
            eem_dataset.cutting(ex_min=ex_range_min, ex_max=ex_range_max, em_min=em_range_min, em_max=em_range_max,
                                inplace=True)
            steps_track += "- EEM cutting \n"

        if file_kw_blank and su:
            file_name_blank_list = [f.replace(file_kw_sample, file_kw_blank) for f in file_name_sample_list]
            blank_stack, ex_range_blank, em_range_blank, _ = read_eem_dataset(
                folder_path=folder_path,
                mandatory_keywords=[],
                optional_keywords=[],
                data_format=eem_data_format,
                index_pos=None,
                custom_filename_list=file_name_blank_list,
                wavelength_alignment=True if align_exem else False,
                interpolation_method="linear",
            )
            eem_dataset.raman_normalization(
                blank=blank_stack,
                ex_range_blank=ex_range_blank,
                em_range_blank=em_range_blank,
                from_blank=True,
                ex_target=su_ex,
                bandwidth=su_em_width,
                rsu_standard=su_normalization_factor,
                inplace=True,
            )
            steps_track += ["- Raman scattering unit normalization\n"]

        if file_kw_abs and ife:
            file_name_abs_list = [f.replace(file_kw_sample, file_kw_abs) for f in file_name_sample_list]
            abs_stack, ex_range_abs, _ = read_abs_dataset(
                folder_path=folder_path,
                mandatory_keywords=[],
                optional_keywords=[],
                data_format=abs_data_format,
                index_pos=None,
                custom_filename_list=file_name_abs_list,
                wavelength_alignment=True if align_exem else False,
                interpolation_method="linear",
            )
            eem_dataset.ife_correction(absorbance=abs_stack, ex_range_abs=ex_range_abs, inplace=True)
            steps_track += ["- Inner filter effect correction\n"]

        if all([median_filter, median_filter_ex, median_filter_em, median_filter_mode]):
            eem_dataset.median_filter(window_size=(median_filter_ex, median_filter_em), mode=median_filter_mode,
                                      inplace=True)
            steps_track += ["- Median filter\n"]

        if all([raman, raman_method, raman_width, raman_dimension]):
            eem_dataset.raman_scattering_removal(
                interpolation_method=raman_method,
                width=raman_width,
                interpolation_dimension=raman_dimension,
                inplace=True,
            )
            steps_track += ["- Raman scattering removal\n"]

        if all(
                [
                    rayleigh,
                    rayleigh_o1_method,
                    rayleigh_o1_dimension,
                    rayleigh_o1_width,
                    rayleigh_o2_method,
                    rayleigh_o2_dimension,
                    rayleigh_o2_width,
                ]
        ):
            eem_dataset.rayleigh_scattering_removal(
                width_o1=rayleigh_o1_width,
                width_o2=rayleigh_o2_width,
                interpolation_method_o1=rayleigh_o1_method,
                interpolation_method_o2=rayleigh_o2_method,
                interpolation_dimension_o1=rayleigh_o1_dimension,
                interpolation_dimension_o2=rayleigh_o2_dimension,
                inplace=True,
            )
            steps_track += ["- Rayleigh scattering removal\n"]

        if all([gaussian, gaussian_sigma, gaussian_truncate]):
            eem_dataset.gaussian_filter(sigma=gaussian_sigma, truncate=gaussian_truncate, inplace=True)
            steps_track += ["- Gaussian smoothing\n"]

        eem_dataset_json_dict = {
            "eem_stack": [
                [[None if np.isnan(x) else x for x in subsublist] for subsublist in sublist]
                for sublist in eem_dataset.eem_stack.tolist()
            ],
            "ex_range": eem_dataset.ex_range.tolist(),
            "em_range": eem_dataset.em_range.tolist(),
            "index": eem_dataset.index,
            "ref": [refs.columns.tolist()] + refs.values.tolist() if eem_dataset.ref is not None else None,
            "cluster": None,
        }

        return eem_dataset_json_dict, dbc.Label(steps_track, style={"whiteSpace": "pre"}), "Build"

    #   ---------------Export EEM

    @app.callback(
        [
            Output(IDS.MESSAGE_EEM_DATASET_EXPORT, "children"),
            Output(IDS.EXPORT_EEM_DATASET_SPINNER, "children"),
        ],
        [
            Input(IDS.EXPORT_EEM_DATASET, "n_clicks"),
            Input(IDS.BUILD_EEM_DATASET, "n_clicks"),
            State(IDS.EEM_DATASET, "data"),
            State(IDS.FOLDER_PATH_EXPORT_EEM_DATASET, "value"),
            State(IDS.FILENAME_EXPORT_EEM_DATASET, "value"),
            State(IDS.EEM_DATASET_EXPORT_FORMAT, "value"),
        ],
    )
    def on_export_eem_dataset(n_clicks_export, n_clicks_build, eem_dataset_json_dict, export_folder_path,
                              export_filename, export_format):
        if ctx.triggered_id == IDS.BUILD_EEM_DATASET:
            return [None], "Export"
        if eem_dataset_json_dict is None:
            message = ["Please first build the eem dataset."]
            return message, "Export"
        if not os.path.isdir(export_folder_path):
            message = ["Error: No such file or directory: " + export_folder_path]
            return message, "Export"
        else:
            path = export_folder_path + "/" + export_filename + "." + export_format
        with open(path, "w") as file:
            json.dump(eem_dataset_json_dict, file)

        return ["EEM dataset exported."], "Export"
