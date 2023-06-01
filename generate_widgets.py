import ipywidgets
import matplotlib.pyplot as plt
import numpy as np
from datetime import date

import pandas as pd
from IPython.display import display
from ipywidgets import Layout, Label, interactive
from read_data import read_reference_from_text, string_to_float_list, plot3DEEM
from EEMprocessing import EEMstack, eem_statistics, plot_eem_interact, decomposition_interact, \
    decomposition_reconstruction_interact, export_parafac, load_eem_stack_interact, eems_regional_integration, \
    eems_isolation_forest, eems_one_class_svm, fast_core_consistency, split_validation, \
    eems_total_fluorescence_normalization, \
    eem_region_masking, eem_grid_imputing, explained_variance, parafac_pixel_error, parafac_sample_error
from tensorly.decomposition import parafac, non_negative_parafac

form_item_layout = Layout(display='flex',
                          flex_flow='row',
                          justify_content='space-between')


# ----------------------Part 1. Specify data directory and filename format-----------------------

class Widgets_read_data:
    def __init__(self, filedir_default):
        self.filedir_default = filedir_default
        self.dir_selection = ipywidgets.Text(
            value=self.filedir_default,
            description='File directory',
            layout=Layout(width='100%')
        )
        self.ts_read_from_filename = ipywidgets.Checkbox(
            value=True,
            description='Do you want to read timestamps from filenames?',
            style={'description_width': 'initial'},
            layout=Layout(width='50%')
        )
        self.ts_format = ipywidgets.Text(
            value='%Y-%m-%d-%H-%M',
            decription='Format of time in the filenames',
            layout=Layout(width='30%')
        )
        self.ts_start_position = ipywidgets.IntText(
            value=1,
            decription='The start position of time in the filename (count from zero)',
            layout=Layout(width='10%')
        )
        self.ts_end_position = ipywidgets.IntText(
            value=16,
            decription='The start position of time in the filename (count from zero)',
            layout=Layout(width='10%')
        )

    def generate_widgets(self):
        ts_widget = ipywidgets.Box([self.ts_format, self.ts_start_position, self.ts_end_position])
        caption0 = ipywidgets.VBox(
            [Label(value='Pleae specify the directory of fluorescence data in the text box below. Example:'),
             Label(value='../../data/introduction/ (relative path)'),
             Label(value='OR'),
             Label(value='C:/Users/Alice/MasterThesis/data/introduction (absolute path)'),
             Label(value='The directory would change automatically after entering new path.')])
        caption1 = ipywidgets.Label(
            value='If you want to read the timestamps from the filename, please specify the time format, start and end '
                  'positions of time in the filename:')
        caption2 = ipywidgets.Label(value='Time format reference: https://strftime.org/')
        caption3 = ipywidgets.VBox([Label(value='Example: "2020-12-02-22-00-00_R2PEM.dat"'),
                                    Label(value='Time format = %Y-%m-%d-%H-%M-%S'),
                                    Label(value='start position = 1'),
                                    Label(value='end position = 19')])

        data_selection_items = [caption0, self.dir_selection, caption1, self.ts_read_from_filename, ts_widget, caption2,
                                caption3]
        data_selection = ipywidgets.Box(data_selection_items, layout=Layout(
            display='flex',
            flex_flow='column',
            border=None,
            align_items='stretch',
            width='100%'
        ))
        return data_selection


# ----------------------Part 2&3. Data preview, parameter selection, and data stacking-----------------------

class Widgets_create_stack:
    def __init__(self, intensity_slider, em_slider, ex_slider, filedir, datlist, ts_format, ts_start_position,
                 ts_end_position):
        # --------Part2 tab 1------------
        self.dilution = ipywidgets.FloatText(value=1, description='Dilution factor')
        self.rotate = ipywidgets.Checkbox(value=False, description='Rotate EEM')
        self.autoscale = ipywidgets.Checkbox(value=False,
                                             description='Autoscale')
        self.raman_normalization = ipywidgets.Checkbox(value=True, description='Raman normalization')
        self.inner_filter_effect = ipywidgets.Checkbox(value=True, description='Inner filter effect correction')
        self.rayleigh_scattering_correction = ipywidgets.Checkbox(value=True,
                                                                  description='Rayleigh scattering removal:')
        self.raman_scattering_correction = ipywidgets.Checkbox(value=True,
                                                               description='Raman scattering removal:')
        self.contour_mask = ipywidgets.Checkbox(value=False,
                                                description='Contour detection')
        self.gaussian_smoothing = ipywidgets.Checkbox(value=True,
                                                      description='Gaussian smoothing')
        self.crange_cw = ipywidgets.IntRangeSlider(
            value=intensity_slider[0:2],
            min=intensity_slider[0],
            max=intensity_slider[1],
            step=intensity_slider[2],
            description='Intensity',
            continuous_update=False,
            style={'description_width': 'initial'})

        self.em_range_display = ipywidgets.IntRangeSlider(
            value=em_slider[0:2],
            min=em_slider[0],
            max=em_slider[1],
            step=em_slider[2],
            description='Emission',
            continuous_update=False,
            style={'description_width': 'initial'})

        self.ex_range_display = ipywidgets.IntRangeSlider(
            value=ex_slider[0:2],
            min=ex_slider[0],
            max=ex_slider[1],
            step=ex_slider[2],
            description='Excitation',
            continuous_update=False,
            style={'description_width': 'initial'})

        self.filedir = ipywidgets.fixed(filedir)

        self.plot_abs = ipywidgets.Checkbox(value=True,
                                            description='Plot absorbance')

        self.filename = ipywidgets.Dropdown(options=datlist,
                                            description='Filename',
                                            style={'description_width': 'initial'},
                                            layout={'width': 'max-content'})

        self.title = ipywidgets.Checkbox(value=False,
                                         description='Figure title (time)')

        self.ABSxmax = ipywidgets.fixed(0.1)

        # --------Part2 tab 2------------
        self.not_from_blank = ipywidgets.Checkbox(value=True, description='Input RSU manually ->',
                                                  style={'description_width': 'initial'})
        self.manual_rsu = ipywidgets.FloatText(value=1,
                                               style={'description_width': 'initial'})
        self.integration_time = ipywidgets.FloatText(value=1, description='Integration time of blank [s]',
                                                     style={'description_width': 'initial'})
        self.ex_lb = ipywidgets.FloatText(value=349, description='Lower bound of ex for RSU',
                                          style={'description_width': 'initial'})
        self.ex_ub = ipywidgets.FloatText(value=351, description='Upper bound of ex for RSU',
                                          style={'description_width': 'initial'})

        self.from_blank = ipywidgets.Checkbox(value=False, description='Calculate RSU from the blank ->',
                                              style={'description_width': 'initial'})
        self.bandwidth = ipywidgets.FloatText(value=1800, description='Raman peak bandwidth',
                                              style={'description_width': 'initial'})
        self.bandwidth_type = ipywidgets.Dropdown(value='wavenumber',
                                                  options=['wavenumber', 'wavelength'],
                                                  style={'description_width': 'initial'})
        self.rsu_standard = ipywidgets.FloatText(value=20000, description='Baseline raman scattering unit [nm^2]',
                                                 style={'description_width': 'initial'})
        self.gaussian_sigma = ipywidgets.FloatText(value=1, description='gaussian smoothing sigma',
                                                   style={'description_width': 'initial'})
        self.gaussian_truncate = ipywidgets.IntText(value=3, description='gaussian smoothing truncate',
                                                    style={'description_width': 'initial'})
        self.contour_otsu = ipywidgets.Checkbox(value=True, description='OTSU automatic thresholding',
                                                style={'description_width': 'initial'})
        self.contour_binary_threshold = ipywidgets.IntText(value=50, description='Mannual thresholding (0-255)',
                                                           style={'description_width': 'initial'})

        self.method_raman = ipywidgets.Dropdown(value='linear',
                                                options=['linear', 'cubic', 'nan'],
                                                description='interpolation method',
                                                style={'description_width': 'initial'},
                                                layout=Layout(width='200px'))
        self.method_o1 = ipywidgets.Dropdown(value='zero',
                                             options=['zero', 'linear', 'cubic', 'nan'],
                                             description='interpolation method',
                                             style={'description_width': 'initial'},
                                             layout=Layout(width='200px'))
        self.method_o2 = ipywidgets.Dropdown(value='linear',
                                             options=['zero', 'linear', 'cubic', 'nan'],
                                             description='interpolation method',
                                             style={'description_width': 'initial'},
                                             layout=Layout(width='200px'))
        self.axis_o1 = ipywidgets.Dropdown(value='grid',
                                           options=['ex', 'em', 'grid'],
                                           description='axis',
                                           style={'description_width': 'initial'},
                                           layout=Layout(width='100px'))
        self.axis_o2 = ipywidgets.Dropdown(value='grid',
                                           options=['ex', 'em', 'grid'],
                                           description='axis',
                                           style={'description_width': 'initial'},
                                           layout=Layout(width='100px'))
        self.axis_raman = ipywidgets.Dropdown(value='grid',
                                              options=['ex', 'em', 'grid'],
                                              description='axis',
                                              style={'description_width': 'initial'},
                                              layout=Layout(width='100px'))
        self.tolerance_raman = ipywidgets.IntText(value=5, description='width [nm]',
                                                  style={'description_width': 'initial'})
        self.tolerance_o1 = ipywidgets.IntText(value=15, description='width [nm]',
                                               style={'description_width': 'initial'})
        self.tolerance_o2 = ipywidgets.IntText(value=15, description='width [nm]',
                                               style={'description_width': 'initial'})
        self.ts_format_plot = ipywidgets.fixed(value=ts_format)
        self.ts_start_position_plot = ipywidgets.fixed(value=ts_start_position - 1)
        self.ts_end_position_plot = ipywidgets.fixed(value=ts_end_position)
        self.preview_parameter_dict = {'filedir': self.filedir, 'filename': self.filename,
                                       'autoscale': self.autoscale,
                                       'crange': self.crange_cw, 'raman_normalization': self.raman_normalization,
                                       'rotate': self.rotate, 'dilution': self.dilution,
                                       'inner_filter_effect': self.inner_filter_effect,
                                       'rayleigh_scattering_correction': self.rayleigh_scattering_correction,
                                       'raman_scattering_correction': self.raman_scattering_correction,
                                       'plot_abs': self.plot_abs, 'abs_xmax': self.ABSxmax, 'title': self.title,
                                       'em_range_display': self.em_range_display,
                                       'ex_range_display': self.ex_range_display,
                                       'contour_mask': self.contour_mask,
                                       'gaussian_smoothing': self.gaussian_smoothing, 'method_raman': self.method_raman,
                                       'method_o1': self.method_o1, 'method_o2': self.method_o2,
                                       'axis_o1': self.axis_o1, 'axis_o2': self.axis_o2, 'axis_raman': self.axis_raman,
                                       'sigma': self.gaussian_sigma, 'truncate': self.gaussian_truncate,
                                       'integration_time': self.integration_time, 'ex_lb': self.ex_lb,
                                       'ex_ub': self.ex_ub, 'rsu_standard': self.rsu_standard,
                                       'bandwidth_type': self.bandwidth_type, 'bandwidth': self.bandwidth,
                                       'manual_rsu': self.manual_rsu, 'from_blank': self.from_blank,
                                       'otsu': self.contour_otsu, 'binary_threshold': self.contour_binary_threshold,
                                       'tolerance_raman': self.tolerance_raman,
                                       'tolerance_o1': self.tolerance_o1, 'tolerance_o2': self.tolerance_o2,
                                       'ts_format': self.ts_format_plot,
                                       'ts_start_position': self.ts_start_position_plot,
                                       'ts_end_position': self.ts_end_position_plot
                                       }
        self.synchronize_resolution = ipywidgets.Checkbox(value=False, description='Synchronize wavelength intervals',
                                                          style={'description_width': 'initial'})

    def on_not_from_blank(self, change):
        self.not_from_blank.value = not change.new

    def on_from_blank(self, change):
        self.from_blank.value = not change.new

    def generate_widgets(self):
        self.from_blank.observe(self.on_not_from_blank, 'value')
        self.not_from_blank.observe(self.on_from_blank, 'value')
        form_item_layout = Layout(
            display='flex',
            flex_flow='row',
            justify_content='space-between'
        )

        plot_eem_items = [
            ipywidgets.Box([self.filename], layout=form_item_layout),
            ipywidgets.Box([self.em_range_display, self.ex_range_display, self.crange_cw], layout=form_item_layout),
            ipywidgets.Box([self.title, self.plot_abs, self.rotate, self.dilution], layout=form_item_layout)
        ]
        plot_eem = ipywidgets.Box(plot_eem_items, layout=Layout(
            display='flex',
            flex_flow='column',
            border='solid 2px',
            align_items='stretch',
            width='100%'
        ))
        normalization_items = [
            ipywidgets.Box([self.inner_filter_effect, self.raman_normalization, ]),
            ipywidgets.Box([ipywidgets.Label(value="Raman normalization settings:")]),
            ipywidgets.Box([self.not_from_blank, self.manual_rsu]),
            ipywidgets.Box([self.from_blank, self.ex_lb, self.ex_ub]),
            ipywidgets.Box([self.bandwidth, self.bandwidth_type]),
            ipywidgets.Box([self.rsu_standard, self.integration_time]),
        ]
        normalization = ipywidgets.Box(normalization_items, layout=Layout(
            display='flex',
            flex_flow='column',
            border='solid 2px',
            align_items='stretch',
            width='100%'
        ))

        scattering_removal_items = [
            ipywidgets.Box([self.rayleigh_scattering_correction], layout=form_item_layout),
            ipywidgets.Box([ipywidgets.Label(value="1st order: ")]),
            ipywidgets.Box([self.method_o1, self.axis_o1, self.tolerance_o1]),
            ipywidgets.Box([ipywidgets.Label(value="2nd order")]),
            ipywidgets.Box([self.method_o2, self.axis_o2, self.tolerance_o2]),
            ipywidgets.Box([self.raman_scattering_correction], layout=form_item_layout),
            ipywidgets.Box([self.method_raman, self.axis_raman, self.tolerance_raman]),
        ]
        scattering_removal = ipywidgets.Box(scattering_removal_items, layout=Layout(
            display='flex',
            flex_flow='column',
            border='solid 2px',
            align_items='stretch',
            width='100%'
        ))

        smoothing_items = [
            ipywidgets.Box([self.gaussian_smoothing], layout=form_item_layout),
            ipywidgets.Box([self.gaussian_sigma, self.gaussian_truncate], layout=form_item_layout)
        ]
        smoothing = ipywidgets.Box(smoothing_items, layout=Layout(
            display='flex',
            flex_flow='column',
            border='solid 2px',
            align_items='stretch',
            width='100%'
        ))

        contour_dection_items = [
            ipywidgets.Box([self.contour_mask], layout=form_item_layout),
            ipywidgets.Box([self.contour_otsu, self.contour_binary_threshold], layout=form_item_layout)
        ]
        contour_dection = ipywidgets.Box(contour_dection_items, layout=Layout(
            display='flex',
            flex_flow='column',
            border='solid 2px',
            align_items='stretch',
            width='100%'
        ))

        tab2 = ipywidgets.Tab()
        tab2.children = [plot_eem, normalization, scattering_removal, smoothing, contour_dection]
        tab2.set_title(0, 'Plot EEM')
        tab2.set_title(1, 'Normalization')
        tab2.set_title(2, 'Scattering removal')
        tab2.set_title(3, 'Smoothing')
        tab2.set_title(4, 'Contour detection')

        out_parameters = ipywidgets.interactive_output(plot_eem_interact, self.preview_parameter_dict)

        note_step2 = ipywidgets.VBox([ipywidgets.Label(value="If you see blank space in the short excitation wavelength"
                                                             " region, it's likely that the inner filter effect is too "
                                                             "strong."),
                                      ipywidgets.Label(
                                          value="Please consider adjust the excitation wavelength range.")])
        return tab2, note_step2, out_parameters

    def generate_widgets2(self):
        stacking_interact = interactive(
            load_eem_stack_interact,
            {'manual': True, 'manual_name': 'Stack data'},
            filedir=ipywidgets.fixed(value=self.filedir.value),
            raman_normalization=ipywidgets.fixed(value=self.raman_normalization.value),
            inner_filter_effect=ipywidgets.fixed(value=self.inner_filter_effect.value),
            rayleigh_scattering_correction=ipywidgets.fixed(value=self.rayleigh_scattering_correction.value),
            raman_scattering_correction=ipywidgets.fixed(value=self.raman_scattering_correction.value),
            em_range_display=ipywidgets.fixed(value=self.em_range_display.value),
            ex_range_display=ipywidgets.fixed(value=self.ex_range_display.value),
            gaussian_smoothing=ipywidgets.fixed(value=self.gaussian_smoothing.value),
            sigma=ipywidgets.fixed(value=self.gaussian_sigma.value),
            truncate=ipywidgets.fixed(value=self.gaussian_truncate.value),
            otsu=ipywidgets.fixed(value=self.contour_otsu.value),
            binary_threshold=ipywidgets.fixed(value=self.contour_binary_threshold.value),
            dilution=ipywidgets.fixed(value=self.dilution.value),
            integration_time=ipywidgets.fixed(value=self.integration_time.value),
            ex_lb=ipywidgets.fixed(value=self.ex_lb.value),
            ex_ub=ipywidgets.fixed(value=self.ex_ub.value),
            bandwidth_type=ipywidgets.fixed(value=self.bandwidth_type.value),
            bandwidth=ipywidgets.fixed(value=self.bandwidth.value),
            from_blank=ipywidgets.fixed(value=self.from_blank.value),
            manual_rsu=ipywidgets.fixed(value=self.manual_rsu.value),
            rsu_standard=ipywidgets.fixed(value=self.rsu_standard.value),
            tolerance_raman=ipywidgets.fixed(value=self.tolerance_raman.value),
            tolerance_o1=ipywidgets.fixed(value=self.tolerance_o1.value),
            tolerance_o2=ipywidgets.fixed(value=self.tolerance_o2.value),
            method_raman=ipywidgets.fixed(value=self.method_raman.value),
            method_o1=ipywidgets.fixed(value=self.method_o1.value),
            method_o2=ipywidgets.fixed(value=self.method_o2.value),
            axis_raman=ipywidgets.fixed(value=self.axis_raman.value),
            axis_o1=ipywidgets.fixed(value=self.axis_o1.value),
            axis_o2=ipywidgets.fixed(value=self.axis_o2.value),
            contour_mask=ipywidgets.fixed(value=self.contour_mask.value),
            keyword_pem=ipywidgets.Text(
                value='PEM.dat',
                style={'description_width': 'initial'},
                description='Filename searching keyword: '
            ),
            existing_datlist=ipywidgets.fixed(value=[]),
            wavelength_synchronization=ipywidgets.Checkbox(
                value=False,
                style={'description_width': 'initial'},
                description='Align wavelengths for all samples'
            ))

        return stacking_interact


# ----------------------Part 4. Remove unwanted data from the data stack-----------------------

class Widgets_data_cleaning:
    def __init__(self, eem_stack, datlist_all, em_range, ex_range, eem_preview_parameters):
        self.eem_stack_imputed = eem_stack.copy()
        self.datlist_all = datlist_all
        self.datlist_filtered = datlist_all.copy()
        self.idx2remove = []
        self.em_range_mask = ipywidgets.IntRangeSlider(
            value=[int(em_range[0] + em_range[-1]) / 2, int(em_range[0] + em_range[-1]) / 2 + 10],
            min=em_range[0],
            max=em_range[-1],
            step=em_range[1] - em_range[0],
            description='Emission range masked',
            continuous_update=False,
            width='30%',
            style={'description_width': 'initial'})
        self.ex_range_mask = ipywidgets.IntRangeSlider(
            value=[int(ex_range[0] + ex_range[-1]) / 2, int(ex_range[0] + ex_range[-1]) / 2 + 10],
            min=ex_range[0],
            max=ex_range[-1],
            step=ex_range[1] - ex_range[0],
            description='Excitation range masked',
            continuous_update=False,
            width='30%',
            style={'description_width': 'initial'})
        self.filelist_preview = ipywidgets.Dropdown(options=self.datlist_all,
                                                    style={'description_width': 'initial'},
                                                    layout={'width': 'max-content'})
        self.preview_parameter_dict = eem_preview_parameters  # from Widgets2and3
        self.eem_stack = eem_stack
        self.em_range = em_range
        self.ex_range = ex_range
        self.auto_detection_method = ipywidgets.Dropdown(options=['Isolation forest', 'One-class-SVM', 'Mixed'],
                                                         description='Artefact detection algorithm')
        self.tf_normalization = ipywidgets.Checkbox(value=True, description='Normalize EEM with total fluorescence')
        self.grid_size = ipywidgets.FloatText(value=10, description='The length and width of each pixel after '
                                                                    'down-sampling')
        self.contamination = ipywidgets.FloatText(value=0.02, description='The proportion of samples with artefacts')
        self.auto_detection_labels = []
        self.mask_impute = ipywidgets.fixed(value=True)

    def update_filelist(self, foo):
        try:
            self.datlist_filtered.remove(self.filelist_preview.value)
            self.idx2remove.append(self.datlist_all.index(self.filelist_preview.value))
        except ValueError:
            pass
        print('"' + self.filelist_preview.value + '"' + " has been removed")

    def generate_widgets_1(self):
        button_update = ipywidgets.Button(description="Remove EEM for stack")
        self.preview_parameter_dict['filename'] = self.filelist_preview

        button_update.on_click(self.update_filelist)

        eem_preview = ipywidgets.interactive_output(plot_eem_interact, self.preview_parameter_dict)

        manual_cleaning_items = [
            ipywidgets.Box([eem_preview], layout=form_item_layout),
            ipywidgets.Box([self.filelist_preview, button_update], layout=form_item_layout)
        ]

        manual_cleaning = ipywidgets.Box(manual_cleaning_items, layout=Layout(
            display='flex',
            flex_flow='column',
            border='solid 2px',
            align_items='stretch',
            width='100%'
        ))
        return manual_cleaning

    def auto_detection(self, foo):
        if self.auto_detection_method.value == 'Isolation forest':
            self.auto_detection_labels = eems_isolation_forest(self.eem_stack, self.em_range, self.ex_range,
                                                               self.tf_normalization.value,
                                                               (self.grid_size.value, self.grid_size.value),
                                                               self.contamination.value)
        if self.auto_detection_method.value == 'One-class-SVM':
            self.auto_detection_labels = eems_one_class_svm(self.eem_stack, self.em_range, self.ex_range,
                                                            self.tf_normalization.value,
                                                            (self.grid_size.value, self.grid_size.value),
                                                            self.contamination.value)
        if self.auto_detection_method.value == 'Mixed':
            y1 = eems_isolation_forest(self.eem_stack, self.em_range, self.ex_range, self.tf_normalization.value,
                                       (self.grid_size.value, self.grid_size.value), self.contamination.value)
            y2 = eems_one_class_svm(self.eem_stack, self.em_range, self.ex_range, self.tf_normalization.value,
                                    (self.grid_size.value, self.grid_size.value), self.contamination.value)
            self.auto_detection_labels = np.array([max(i, j) for i, j in zip(y1, y2)])
        n_outliers = np.count_nonzero(self.auto_detection_labels == -1)
        n_cols = 4
        n_rows = n_outliers // n_cols + 1
        count = 0
        extent = [self.em_range.min(), self.em_range.max(), self.ex_range.min(), self.ex_range.max()]
        print('overview of artefacts detected')
        plt.figure(figsize=(15, n_rows * 3))
        for i in range(len(self.auto_detection_labels)):
            if self.auto_detection_labels[i] == -1:
                axs = plt.subplot2grid((n_rows, n_cols), (count // n_cols, count % n_cols))
                crange = self.preview_parameter_dict['crange'].value
                axs.imshow(self.eem_stack[i], cmap='jet', extent=extent, vmin=min(crange), vmax=max(crange))
                axs.axis('off')
                axs.set_title(self.datlist_all[i], size=8)
                count += 1

    def update_auto_detection(self, foo):
        for i in range(len(self.auto_detection_labels)):
            if self.auto_detection_labels[i] == -1:
                try:
                    self.idx2remove.append(i)
                    self.datlist_filtered.remove(self.datlist_all[i])
                except ValueError:
                    pass
                print('"' + self.datlist_all[i] + '"' + " has been removed")

    def generate_widgets_2(self):
        button_detect = ipywidgets.Button(description="Detect artefacts")
        button_update = ipywidgets.Button(description='Accept artefacts removal')
        button_detect.on_click(self.auto_detection)
        button_update.on_click(self.update_auto_detection)
        auto_cleaning_items = [
            ipywidgets.Box([self.auto_detection_method, self.grid_size, self.contamination], layout=form_item_layout),
            ipywidgets.Box([button_detect], layout=form_item_layout),
            ipywidgets.Box([button_update], layout=form_item_layout)
        ]
        auto_cleaning = ipywidgets.Box(auto_cleaning_items, layout=Layout(
            display='flex',
            flex_flow='column',
            border='solid 2px',
            align_items='stretch',
            width='100%'
        ))
        return auto_cleaning

    def update_mask_impute_for_one(self, foo):
        idx = self.datlist_all.index(self.filelist_preview.value)

        masked, mask = eem_region_masking(self.eem_stack_imputed[idx, :, :], self.em_range, self.ex_range,
                                          self.em_range_mask.value[0], self.em_range_mask.value[-1],
                                          self.ex_range_mask.value[0], self.ex_range_mask.value[-1])
        imputed = eem_grid_imputing(masked, prior_mask=mask)
        self.eem_stack_imputed[idx, :, :] = imputed
        print("Artefact has been imputed for" + '"' + self.filelist_preview.value + '"')

    def update_mask_impute_for_all(self, foo):
        for i in range(self.eem_stack_imputed.shape[0]):
            masked, mask = eem_region_masking(self.eem_stack_imputed[i, :, :], self.em_range, self.ex_range,
                                              self.em_range_mask.value[0], self.em_range_mask.value[-1],
                                              self.ex_range_mask.value[0], self.ex_range_mask.value[-1])
            imputed = eem_grid_imputing(masked, prior_mask=mask)
            self.eem_stack_imputed[i, :, :] = imputed
        print("Artefact has been imputed for all EEMs in the stack")

    def generate_widgets_3(self):
        d = self.preview_parameter_dict.copy()
        d['mask_impute'] = self.mask_impute
        d['mask_ex'] = self.ex_range_mask
        d['mask_em'] = self.em_range_mask
        button_update_one = ipywidgets.Button(description="Remove artefact for one EEM",
                                              layout=Layout(width='25%'))
        d['filename'] = self.filelist_preview
        button_update_one.on_click(self.update_mask_impute_for_one)
        eem_preview = ipywidgets.interactive_output(plot_eem_interact, d)
        button_update_all = ipywidgets.Button(description="Remove artefact for all EEMs",
                                              layout=Layout(width='25%'))
        button_update_all.on_click(self.update_mask_impute_for_all)

        artefact_imputing_items = [
            ipywidgets.Box([eem_preview], layout=form_item_layout),
            ipywidgets.Box([self.filelist_preview], layout=form_item_layout),
            ipywidgets.Box([self.ex_range_mask, self.em_range_mask], layout=form_item_layout),
            ipywidgets.Box([button_update_one], layout=form_item_layout),
            ipywidgets.Box([button_update_all], layout=form_item_layout)
        ]

        manual_cleaning = ipywidgets.Box(artefact_imputing_items, layout=Layout(
            display='flex',
            flex_flow='column',
            border='solid 2px',
            align_items='stretch',
            width='100%'
        ))
        return manual_cleaning

    # ----------------------Part 5. Data stack analysis-----------------------


# -------Tab1: File range selection----------
class Widgets_stack_processing:
    class Widgets_data_range:
        def __init__(self, datlist_cw):
            self.datlist_cw = datlist_cw

        def generate_widgets(self):
            range1 = ipywidgets.Dropdown(value=self.datlist_cw[0],
                                         options=self.datlist_cw,
                                         description='Start',
                                         style={'description_width': 'initial'},
                                         continuous_update=False)
            range2 = ipywidgets.Dropdown(value=self.datlist_cw[-1],
                                         options=self.datlist_cw,
                                         description='End',
                                         style={'description_width': 'initial'},
                                         continuous_update=False)
            data_range_items = [ipywidgets.Box([Label(value='Select the range of data for further analysis')]),
                                ipywidgets.Box([range1, range2], layout=form_item_layout)]
            return data_range_items, range1, range2

    # --------Tab2: Pixel statistics------------------

    class Widgets_pixel_statistics:
        def __init__(self, eem_stack_cw, datlist_cw, range1, range2, em_range_cw, ex_range_cw, timestamps_cw):
            self.eem_stack_cw = eem_stack_cw
            self.datlist_cw = datlist_cw
            self.range1 = range1
            self.range2 = range2
            self.em_range_cw = em_range_cw
            self.ex_range_cw = ex_range_cw
            self.timestamps_cw = timestamps_cw
            self.property_pixel = ipywidgets.Dropdown(options=['Timeseries analysis', 'Correlation analysis'],
                                                      description='Property')
            self.em_pixel = ipywidgets.FloatText(value=400, description='Em [nm]')
            self.ex_pixel = ipywidgets.FloatText(value=300, description='Ex [nm]')
            self.caption_pixel_statistics = ipywidgets.Label(value='For correlation analysis, please specify the '
                                                                   'reference data with either a file path or a manual'
                                                                   ' input')
            self.checkbox_reference_filepath_pixel = ipywidgets.Checkbox(value=False)
            self.reference_filepath_pixel = ipywidgets.Text(value='reference_example.txt',
                                                            description='File path of input reference data',
                                                            style={'description_width': 'initial'},
                                                            layout=Layout(width='400%'))
            self.checkbox_reference_mannual_input_pixel = ipywidgets.Checkbox(value=True)
            self.reference_mannual_input_pixel = ipywidgets.Text(description='Type input reference data manually',
                                                                 style={'description_width': 'initial'},
                                                                 layout=Layout(width='400%'))
            self.button_pixel_statistics = ipywidgets.Button(description='Calculate')

        def pixel_statistics_interact(self, foo):
            EEMstack_class_cw = EEMstack(self.eem_stack_cw[self.datlist_cw.index(self.range1.value):
                                                           self.datlist_cw.index(self.range2.value) + 1],
                                         self.em_range_cw, self.ex_range_cw)
            if self.property_pixel.value == 'Timeseries analysis':
                if self.timestamps_cw is not None:
                    EEMstack_class_cw.pixel_rel_std(em=self.em_pixel.value, ex=self.ex_pixel.value, plot=True,
                                                    timestamp=self.timestamps_cw[
                                                              self.datlist_cw.index(self.range1.value):
                                                              self.datlist_cw.index(
                                                                  self.range2.value) + 1],
                                                    baseline=False, output=True)
                else:
                    labels = self.datlist_cw[self.datlist_cw.index(self.range1.value):
                                             self.datlist_cw.index(self.range2.value) + 1]
                    EEMstack_class_cw.pixel_rel_std(em=self.em_pixel.value, ex=self.ex_pixel.value, plot=True,
                                                    timestamp=False, baseline=False, output=True, labels=labels)
            if self.property_pixel.value == 'Correlation analysis':
                if self.checkbox_reference_filepath_pixel.value:
                    reference = read_reference_from_text(self.reference_filepath_pixel.value)
                if self.checkbox_reference_mannual_input_pixel.value:
                    reference = string_to_float_list(self.reference_mannual_input_pixel.value)
                EEMstack_class_cw.pixel_linreg(em=self.em_pixel.value, ex=self.ex_pixel.value, x=reference)

        def update_mannual(self, change):
            self.checkbox_reference_mannual_input_pixel.value = not change.new

        def update_filepath(self, change):
            self.checkbox_reference_filepath_pixel.value = not change.new

        def generate_widgets(self):
            self.checkbox_reference_filepath_pixel.observe(self.update_mannual, 'value')
            self.checkbox_reference_mannual_input_pixel.observe(self.update_filepath, 'value')
            self.button_pixel_statistics.on_click(self.pixel_statistics_interact)
            pixel_statistics_items = [ipywidgets.Box([self.ex_pixel, self.em_pixel, self.property_pixel]),
                                      ipywidgets.Box([self.caption_pixel_statistics], layout=form_item_layout),
                                      ipywidgets.Box(
                                          [self.checkbox_reference_filepath_pixel, self.reference_filepath_pixel],
                                          layout=form_item_layout),
                                      ipywidgets.Box(
                                          [self.checkbox_reference_mannual_input_pixel,
                                           self.reference_mannual_input_pixel],
                                          layout=form_item_layout),
                                      ipywidgets.Box([self.button_pixel_statistics])]
            return pixel_statistics_items

    # ----------Tab3: EEM statistics---------

    class Widgets_eem_statistics:
        def __init__(self, eem_stack_cw, datlist_cw, range1, range2, em_range_cw, ex_range_cw, timestamps_cw,
                     crange_cw):
            self.eem_stack_cw = eem_stack_cw
            self.datlist_cw = datlist_cw
            self.range1 = range1
            self.range2 = range2
            self.em_range_cw = em_range_cw
            self.ex_range_cw = ex_range_cw
            self.timestamps_cw = timestamps_cw
            self.crange_cw = crange_cw
            self.property_eem = ipywidgets.Dropdown(
                options=['Mean', 'Standard deviation', 'Relative standard deviation',
                         'Correlation: Linearity', 'Correlation: Pearson coef.',
                         'Correlation: Spearman coef.'],
                description='Property', style={'description_width': 'initial'})
            self.caption_eem_statistics = ipywidgets.Label(value='For correlation analysis, please specify the '
                                                                 'reference data with either a file path or a manual '
                                                                 'input')
            self.checkbox_reference_filepath_eem = ipywidgets.Checkbox(value=False)
            self.reference_filepath_eem = ipywidgets.Text(value='reference_example.txt',
                                                          description='File path of input reference data',
                                                          style={'description_width': 'initial'},
                                                          layout=Layout(width='400%'))
            self.checkbox_reference_mannual_input_eem = ipywidgets.Checkbox(value=True)
            self.reference_mannual_input_eem = ipywidgets.Text(description='Type input reference data manually',
                                                               style={'description_width': 'initial'},
                                                               layout=Layout(width='400%'))
            self.title_eem_statistics = ipywidgets.Checkbox(value=True, description='Title',
                                                            style={'description_width': 'initial'})
            self.button_eem_statistics = ipywidgets.Button(description='Calculate')

        def update_manual(self, change):
            self.checkbox_reference_mannual_input_eem.value = not change.new

        def update_filepath(self, change):
            self.checkbox_reference_filepath_eem.value = not change.new

        def eem_statistics_interact(self, foo):
            EEMstack_class_cw = EEMstack(self.eem_stack_cw[self.datlist_cw.index(self.range1.value):
                                                           self.datlist_cw.index(self.range2.value) + 1],
                                         self.em_range_cw, self.ex_range_cw)
            reference = None
            header = None
            if self.property_eem.value == 'Correlation: Linearity' or self.property_eem.value == 'Correlation: ' \
                                                                                                 'Pearson coef.' \
                    or self.property_eem.value == 'Correlation: Spearman coef.':
                if self.checkbox_reference_filepath_eem.value:
                    reference, header = read_reference_from_text(self.reference_filepath_eem.value)
                if self.checkbox_reference_mannual_input_eem.value:
                    reference = string_to_float_list(self.reference_mannual_input_eem.value)
            eem_statistics(EEMstack_class_cw, term=self.property_eem.value, title=self.title_eem_statistics.value,
                           reference=reference, crange=self.crange_cw.value, reference_label=header)

        def generate_widgets(self):
            self.checkbox_reference_filepath_eem.observe(self.update_manual, 'value')
            self.checkbox_reference_mannual_input_eem.observe(self.update_filepath, 'value')
            self.button_eem_statistics.on_click(self.eem_statistics_interact)
            eem_statistics_items = [ipywidgets.Box([self.property_eem]),
                                    ipywidgets.Box([self.caption_eem_statistics], layout=form_item_layout),
                                    ipywidgets.Box([self.checkbox_reference_filepath_eem, self.reference_filepath_eem],
                                                   layout=form_item_layout),
                                    ipywidgets.Box(
                                        [self.checkbox_reference_mannual_input_eem, self.reference_mannual_input_eem],
                                        layout=form_item_layout),
                                    ipywidgets.Box([self.button_eem_statistics])]
            return eem_statistics_items

    # -------Tab4: Regional integration----------

    class Widgets_regional_integration:
        def __init__(self, eem_stack_cw, datlist_cw, range1, range2, em_range_cw, ex_range_cw, timestamps_cw):
            self.eem_stack_cw = eem_stack_cw
            self.datlist_cw = datlist_cw
            self.range1 = range1
            self.range2 = range2
            self.em_range_cw = em_range_cw
            self.ex_range_cw = ex_range_cw
            self.timestamps_cw = timestamps_cw
            self.em_boundary_left = ipywidgets.FloatText(value=300)
            self.em_boundary_right = ipywidgets.FloatText(value=360)
            self.ex_boundary_left = ipywidgets.FloatText(value=280)
            self.ex_boundary_right = ipywidgets.FloatText(value=320)
            self.button_eem_integration = ipywidgets.Button(description='Calculate')
            self.integration_form = ipywidgets.Dropdown(options=['total fluorescence', 'average valid pixel intensity',
                                                                 'number of pixels'],
                                                        description='property', style={'description_width': 'initial'})

        def regional_integration_interact(self, foo):
            eem_stack_cw_selected = \
                self.eem_stack_cw[self.datlist_cw.index(self.range1.value):self.datlist_cw.index(self.range2.value) + 1]
            eem_stack_integration, eem_stack_avg_intensity, eem_stack_num_pixels = \
                eems_regional_integration(eem_stack_cw_selected, self.em_range_cw, self.ex_range_cw,
                                          [self.em_boundary_left.value, self.em_boundary_right.value],
                                          [self.ex_boundary_left.value, self.ex_boundary_right.value])
            if self.timestamps_cw is not None:
                ts_selected = self.timestamps_cw[self.datlist_cw.index(self.range1.value):
                                                 self.datlist_cw.index(self.range2.value) + 1]
                plt.figure(figsize=(10, 6))
                if self.integration_form.value == 'total fluorescence':
                    plt.plot(ts_selected, eem_stack_integration)
                    plt.xlabel('Time')
                    plt.ylabel('Total fluorescence [a.u.]')
                if self.integration_form.value == 'average valid pixel intensity':
                    plt.plot(ts_selected, eem_stack_avg_intensity)
                    plt.xlabel('Time')
                    plt.ylabel('Average intensity [a.u.]')
                if self.integration_form.value == 'number of pixels':
                    plt.plot(ts_selected, eem_stack_num_pixels)
                    plt.xlabel('Time')
                    plt.ylabel('Number of pixels')
            else:
                labels = self.datlist_cw[self.datlist_cw.index(self.range1.value):
                                         self.datlist_cw.index(self.range2.value) + 1]
                ts_selected = [l[:-4] for l in labels]
            tbl = pd.DataFrame(data=eem_stack_integration, index=ts_selected, columns=[self.integration_form.value])
            display(tbl)

        def generate_widgets(self):
            self.button_eem_integration.on_click(self.regional_integration_interact)
            integration_items = [
                ipywidgets.Box([Label(value='Excitation wavelength range: ', style={'description_width': 'initial'}),
                                self.ex_boundary_left, Label(value='to', style={'description_width': 'initial'}),
                                self.ex_boundary_right]),
                ipywidgets.Box([Label(value='Emission wavelength range: ', style={'description_width': 'initial'}),
                                self.em_boundary_left, Label(value='to', style={'description_width': 'initial'}),
                                self.em_boundary_right]),
                self.integration_form, self.button_eem_integration]
            return integration_items

    # -------Tab5: Stack decomposition----------

    # ------N_components optimization------

    # ------PARAFAC---------
    class Widgets_decomposition:
        def __init__(self, data_index, data_index_cw, timestamps_cw, eem_stack_cw, datlist_cw, range1, range2,
                     em_range_cw, ex_range_cw):
            self.data_index = data_index
            self.data_index_cw = data_index_cw
            self.eem_stack_cw = eem_stack_cw
            self.datlist_cw = datlist_cw
            self.range1 = range1
            self.range2 = range2
            self.em_range_cw = em_range_cw
            self.ex_range_cw = ex_range_cw
            self.timestamps_cw = timestamps_cw
            self.rank_display = ipywidgets.IntText(value=4, description='Number of components',
                                                   style={'description_width': 'initial'})
            self.init = ipywidgets.Dropdown(value='svd',
                                            options=['svd', 'random'],
                                            style={'description_width': 'initial'},
                                            description='PARAFAC initialization')
            self.button_decomposition_interact = ipywidgets.Button(description='Decompose',
                                                                   style={'description_width': 'initial'})
            self.show_components = ipywidgets.Checkbox(value=False,
                                                       style={'description_width': 'initial'},
                                                       description='Plot components')
            self.show_loadings = ipywidgets.Checkbox(value=True,
                                                     style={'description_width': 'initial'},
                                                     description='Plot loadings')
            self.decomposition_method_list = ipywidgets.Dropdown(value='non_negative_parafac',
                                                                 options=['parafac', 'non_negative_parafac'],
                                                                 style={'description_width': 'initial'},
                                                                 description='Decomposition method')
            self.dataset_normalization = ipywidgets.Checkbox(value=True,
                                                             style={'description_width': 'initial'},
                                                             description='Normalize the EEMs by the total fluorescence')
            self.show_normalized_score = ipywidgets.Checkbox(value=False,
                                                             style={'description_width': 'initial'},
                                                             description='Normalize the score by mean')
            self.show_normalized_component = ipywidgets.Checkbox(value=False,
                                                                 style={'description_width': 'initial'},
                                                                 description='Normalize the components by their maximum',
                                                                 layout=Layout(width='100%'))
            self.show_normalized_loadings = ipywidgets.Checkbox(value=True,
                                                                style={'description_width': 'initial'},
                                                                description='Normalize the loadings by their STD',
                                                                layout=Layout(width='100%'))
            self.display_score = ipywidgets.Checkbox(value=False,
                                                     style={'description_width': 'initial'},
                                                     description='Plot score')
            self.plot_fmax = ipywidgets.Checkbox(value=True,
                                                 style={'description_width': 'initial'},
                                                 description='Plot Fmax',
                                                 layout=Layout(width='100%'))
            self.score_df = None
            self.exl_df = None
            self.eml_df = None
            self.fmax_df = None

        def decomposition_interact_button(self):
            if not self.data_index:
                self.data_index_cw = self.timestamps_cw
            score_df, exl_df, eml_df, fmax_df, _, _, _ = \
                decomposition_interact(self.eem_stack_cw[self.datlist_cw.index(self.range1.value):
                                                         self.datlist_cw.index(self.range2.value) + 1],
                                       self.em_range_cw, self.ex_range_cw, self.rank_display.value,
                                       index=self.data_index_cw[self.datlist_cw.index(self.range1.value):
                                                                self.datlist_cw.index(self.range2.value) + 1],
                                       decomposition_method=self.decomposition_method_list.value,
                                       init=self.init.value,
                                       plot_loadings=self.show_loadings.value,
                                       plot_components=self.show_components.value,
                                       dataset_normalization=self.dataset_normalization.value,
                                       score_normalization=self.show_normalized_score.value,
                                       loadings_normalization=self.show_normalized_loadings.value,
                                       component_normalization=self.show_normalized_component.value,
                                       component_autoscale=True,
                                       component_cmin=0, component_cmax=1, title=False, cbar=True,
                                       display_score=self.display_score.value, plot_fmax=self.plot_fmax.value)
            self.score_df = score_df
            self.exl_df = exl_df
            self.eml_df = eml_df
            self.fmax_df = fmax_df
            return score_df, exl_df, eml_df, fmax_df

        def generate_widgets(self):
            self.button_decomposition_interact = interactive(self.decomposition_interact_button,
                                                             {'manual': True, 'manual_name': 'Decompose'})
            decomposition_items = [
                ipywidgets.Box([self.rank_display, self.decomposition_method_list, self.init], layout=form_item_layout),
                ipywidgets.Box([
                    Label(value="The above settings applied to all other tabs under '5. PARAFAC'")]),
                ipywidgets.Box([self.dataset_normalization, self.show_components, self.show_normalized_component],
                               layout=form_item_layout),
                ipywidgets.Box([self.show_loadings, self.show_normalized_loadings], layout=form_item_layout),
                ipywidgets.Box([self.display_score, self.show_normalized_score], layout=form_item_layout),
                ipywidgets.Box([self.plot_fmax], layout=form_item_layout),
                self.button_decomposition_interact]
            return decomposition_items

    class Widgets_nc_optimization:
        def __init__(self, eem_stack_cw, decomposition_method, init, dataset_normalization, datlist_cw, range1_cw,
                     range2_cw):
            self.eem_stack = eem_stack_cw
            self.datlist = datlist_cw
            self.range1 = range1_cw
            self.range2 = range2_cw
            self.decomposition_method = decomposition_method
            self.init = init
            self.dataset_normalization = dataset_normalization
            self.rank_range = ipywidgets.IntRangeSlider(value=[1, 10],
                                                        min=1,
                                                        max=10,
                                                        step=1,
                                                        description='Range of ranks',
                                                        continuous_update=False,
                                                        style={'description_width': 'initial'})
            self.diagnose_method = ipywidgets.Dropdown(value='fast core consistency',
                                                       options=['fast core consistency', 'explained variance'],
                                                       style={'description_width': 'initial'},
                                                       description='Diagnose method')
            self.button_nc_optimization_interact = ipywidgets.Button(description='Calculate',
                                                                     style={'description_width': 'initial'})

        def nc_optimization_interact_button(self):
            rank_list = [i for i in range(self.rank_range.value[0], self.rank_range.value[1] + 1)]
            if self.diagnose_method.value == 'fast core consistency':
                cc_list = fast_core_consistency(self.eem_stack[self.datlist.index(self.range1.value):
                                                               self.datlist.index(self.range2.value) + 1],
                                                rank=rank_list,
                                                decomposition_method=self.decomposition_method.value,
                                                init=self.init.value,
                                                dataset_normalization=self.dataset_normalization.value,
                                                plot_cc=True)
                return cc_list
            if self.diagnose_method.value == 'explained variance':
                ev_list = explained_variance(self.eem_stack[self.datlist.index(self.range1.value):
                                                            self.datlist.index(self.range2.value) + 1], rank=rank_list,
                                             decomposition_method=self.decomposition_method.value,
                                             init=self.init.value,
                                             dataset_normalization=self.dataset_normalization.value,
                                             plot_ve=True)
                return ev_list

        def generate_widgets(self):
            self.button_nc_optimization_interact = interactive(self.nc_optimization_interact_button,
                                                               {'manual': True, 'manual_name': 'Calculate'})
            nc_optimization_items = [
                ipywidgets.Box([self.rank_range, self.diagnose_method], layout=form_item_layout),
                self.button_nc_optimization_interact]
            return nc_optimization_items

    # ------Reconstruction error----------

    class Widgets_reconstruction:
        def __init__(self, decomposition_method_list, rank_display, init, crange_cw, data_index, data_index_cw,
                     timestamps_cw, eem_stack_cw, datlist_cw, range1, range2, em_range_cw, ex_range_cw,
                     dataset_normalization):
            self.data_index = data_index
            self.data_index_cw = data_index_cw
            self.eem_stack_cw = eem_stack_cw
            self.datlist_cw = datlist_cw
            self.range1 = range1
            self.range2 = range2
            self.em_range_cw = em_range_cw
            self.ex_range_cw = ex_range_cw
            self.timestamps_cw = timestamps_cw
            self.decomposition_method_list = decomposition_method_list
            self.rank_display = rank_display
            self.init = init
            self.crange_cw = crange_cw
            self.data_to_view = ipywidgets.Dropdown(options=datlist_cw[datlist_cw.index(range1.value):
                                                                       datlist_cw.index(range2.value) + 1],
                                                    description='Select the EEM for inspection',
                                                    style={'description_width': 'initial'},
                                                    layout={'width': 'max-content'})
            self.button_decomposition_re_interact = ipywidgets.Button(description='Reconstruct',
                                                                      style={'description_width': 'initial'})
            self.button_pixel_error_interact = ipywidgets.Button(description='Calculate pixel-wise error',
                                                                 style={'description_width': 'initial'},
                                                                 layout={'width': 'max-content'})
            self.button_sample_error_interact = ipywidgets.Button(description='Calculate sample-wise error',
                                                                  style={'description_width': 'initial'},
                                                                  layout={'width': 'max-content'})
            self.dataset_normalization = dataset_normalization
            self.res_abs = ipywidgets.fixed(value=np.full((self.datlist_cw.index(self.range2.value) + 1 -
                                     self.datlist_cw.index(self.range1.value),
                                     eem_stack_cw.shape[1],
                                     eem_stack_cw.shape[2]), np.nan))
            self.res_ratio = ipywidgets.fixed(value=np.full((self.datlist_cw.index(self.range2.value) + 1 -
                                     self.datlist_cw.index(self.range1.value),
                                     eem_stack_cw.shape[1],
                                     eem_stack_cw.shape[2]), np.nan))
            self.sample_error_type = ipywidgets.Dropdown(options=['MSE', 'PSNR', 'SSIM'],
                                                         description='Sample-wise error type',
                                                         style={'description_width': 'initial'},
                                                         layout={'width': 'max-content'}
                                                         )
            self.plot_pixel_error_switch = ipywidgets.Checkbox(value=False)

        def decomposition_interact_re_button(self, foo):
            dataset = self.eem_stack_cw[self.datlist_cw.index(self.range1.value):
                                        self.datlist_cw.index(self.range2.value) + 1]
            if self.dataset_normalization.value:
                dataset, tf = eems_total_fluorescence_normalization(dataset)
            if self.decomposition_method_list.value == 'parafac':
                factors = parafac(dataset, rank=self.rank_display.value, init=self.init.value)
            elif self.decomposition_method_list.value == 'non_negative_parafac':
                factors = non_negative_parafac(dataset, rank=self.rank_display.value, init=self.init.value)
            elif self.decomposition_method_list.value == 'test_function':
                factors = non_negative_parafac(dataset, rank=self.rank_display.value, fixed_modes=[0, 1],
                                               init=self.init.value)
            I = factors[1][0]
            J = factors[1][1]
            K = factors[1][2]
            if self.dataset_normalization.value:
                I = np.multiply(I, tf[:, np.newaxis])
            decomposition_reconstruction_interact(I, J, K,
                                                  self.eem_stack_cw[self.datlist_cw.index(self.data_to_view.value)],
                                                  self.em_range_cw, self.ex_range_cw,
                                                  self.datlist_cw[self.datlist_cw.index(self.range1.value):
                                                                  self.datlist_cw.index(self.range2.value) + 1],
                                                  self.data_to_view.value, crange=self.crange_cw.value)

        def get_pixel_error(self, foo):
            dataset = self.eem_stack_cw[self.datlist_cw.index(self.range1.value):
                                        self.datlist_cw.index(self.range2.value) + 1]
            res_abs, res_ratio = parafac_pixel_error(dataset, self.em_range_cw, self.ex_range_cw,
                                                     self.rank_display.value,
                                                     decomposition_method=self.decomposition_method_list.value,
                                                     init=self.init.value,
                                                     dataset_normalization=self.dataset_normalization.value)
            self.res_abs.value = res_abs
            self.res_ratio.value = res_ratio
            self.plot_pixel_error_switch.value = True

        def plot_pixel_error(self, res_abs, res_ratio, switch, datlist, data_to_view):
            plt.close()
            if switch:
                plot3DEEM(res_abs[datlist.index(data_to_view)],
                          self.em_range_cw, self.ex_range_cw, autoscale=True,
                          title='Absolute error [a.u.]', cbar_label='Diff. of intensity [a.u.]')
                plot3DEEM(res_ratio[datlist.index(data_to_view)],
                          self.em_range_cw, self.ex_range_cw, cmin=-30, cmax=30,
                          title='Relative error [%]', cbar_label='Diff. / original intensity [%]')

        def plot_sample_error(self, foo):
            dataset = self.eem_stack_cw[self.datlist_cw.index(self.range1.value):
                                        self.datlist_cw.index(self.range2.value) + 1]
            if self.timestamps_cw is not None:
                id = self.timestamps_cw
            else:
                id = self.datlist_cw
            err_list = parafac_sample_error(dataset,
                                            id[self.datlist_cw.index(self.range1.value):
                                               self.datlist_cw.index(self.range2.value) + 1],
                                            self.rank_display.value, error_type=self.sample_error_type.value,
                                            decomposition_method=self.decomposition_method_list.value,
                                            init=self.init.value,
                                            dataset_normalization=self.dataset_normalization.value,
                                            plot_error=True)

        def generate_widgets(self):
            # self.button_decomposition_re_interact.on_click(self.decomposition_interact_re_button)
            self.button_sample_error_interact.on_click(self.plot_sample_error)
            self.button_pixel_error_interact.on_click(self.get_pixel_error)
            pixel_error_interact = ipywidgets.interactive_output(
                self.plot_pixel_error,
                {'res_abs': self.res_abs,
                 'res_ratio': self.res_ratio,
                 'switch': self.plot_pixel_error_switch,
                 'datlist': ipywidgets.fixed(value=self.datlist_cw),
                 'data_to_view': self.data_to_view})

            reconstruction_error_items = [
                ipywidgets.Box([self.sample_error_type, self.button_sample_error_interact], layout=form_item_layout),
                ipywidgets.Box([self.data_to_view, self.button_pixel_error_interact],
                               layout=form_item_layout)]
            return reconstruction_error_items, pixel_error_interact

    # -------Split validation-----------

    class Widgets_split_validation:
        def __init__(self, eem_stack_cw, em_range_cw, ex_range_cw, rank, init, decomposition_method_list, datlist_cw,
                     range1, range2, dataset_normalization):
            self.eem_stack_cw = eem_stack_cw
            self.em_range_cw = em_range_cw
            self.ex_range_cw = ex_range_cw
            self.rank = rank
            self.init = init
            self.n_split = ipywidgets.IntText(value=4, description='Number of splits',
                                              style={'description_width': 'initial'})
            self.combination_size = ipywidgets.Text(value='half',
                                                    description='Size of the combinations',
                                                    style={'description_width': 'initial'})
            self.n_test = ipywidgets.Text(value='max',
                                          description='Number of tests',
                                          style={'description_width': 'initial'})
            self.rule = ipywidgets.Dropdown(value='random',
                                            options=['random', 'chronological'],
                                            style={'description_width': 'initial'},
                                            description='Spliting method')
            self.criteria = ipywidgets.Dropdown(value='Tucker congruence',
                                                options=['Tucker congruence'],
                                                style={'description_width': 'initial'},
                                                description='Similarity index')
            self.plot_tests = ipywidgets.Checkbox(value=True,
                                                  style={'description_width': 'initial'},
                                                  description='plot the tests')
            self.decomposition_method_list = decomposition_method_list
            self.datlist_cw = datlist_cw
            self.range1 = range1
            self.range2 = range2
            self.dataset_normalization = dataset_normalization
            self.button_split_validation_interact = ipywidgets.Button(description='Run',
                                                                      style={'description_width': 'initial'})

        def split_test_button(self, foo):
            dataset = self.eem_stack_cw[self.datlist_cw.index(self.range1.value):
                                        self.datlist_cw.index(self.range2.value) + 1]
            models, sims_df = split_validation(dataset, self.em_range_cw, self.ex_range_cw, rank=self.rank.value,
                                               datlist=self.datlist_cw,
                                               decomposition_method=self.decomposition_method_list.value,
                                               n_split=self.n_split.value, combination_size=self.combination_size.value,
                                               n_test=self.n_test.value, rule='random', index=[], criteria='TCC',
                                               plot_all_combos=self.plot_tests.value,
                                               dataset_normalization=self.dataset_normalization.value,
                                               init=self.init.value)
            return models, sims_df

        def generate_widgets(self):
            self.button_split_validation_interact.on_click(self.split_test_button)
            split_validation_items = [ipywidgets.Box([
                Label(value='Please first specify the number of components and decomposition method tab '
                            '"Decomposition"')]),
                self.n_split, self.combination_size, self.n_test, self.rule, self.criteria, self.plot_tests,
                self.button_split_validation_interact]
            return split_validation_items


# ----------------------Part 6. Save PARAFAC result-----------------------

class Widgets_export_parafac:
    def __init__(self, score_df, exl_df, eml_df, filedir_default, inner_filter_effect, raman_scattering_correction,
                 rayleigh_scattering_correction, gaussian_smoothing, decomposition_method_list, tf_normalization):
        self.score_df = score_df
        self.exl_df = exl_df
        self.eml_df = eml_df
        self.filedir_default = filedir_default + '/parafac_output.txt'
        self.inner_filter_effect = inner_filter_effect
        self.raman_scattering_correction = raman_scattering_correction
        self.rayleigh_scattering_correction = rayleigh_scattering_correction
        self.gaussian_smoothing = gaussian_smoothing
        self.decomposition_method_list = decomposition_method_list
        self.tf_normalization = tf_normalization
        self.filepath_i = ipywidgets.Text(
            value=self.filedir_default,
            description='file save path*',
            style={'description_width': 'initial'},
            layout=Layout(width='100%'))
        self.name_i = ipywidgets.Text(
            value='',
            description='project name*',
            style={'description_width': 'initial'},
            layout=Layout(width='33%'))
        self.creator_i = ipywidgets.Text(
            value='Yongmin Hu',
            description='file creator*',
            style={'description_width': 'initial'},
            layout=Layout(width='33%'))
        self.date_i = ipywidgets.Text(
            value=date.today().strftime("%Y-%m-%d"),
            description='date*',
            style={'description_width': 'initial'},
            layout=Layout(width='33%'))
        self.email_i = ipywidgets.Text(
            value='',
            description='email',
            style={'description_width': 'initial'},
            layout=Layout(width='100%'))
        self.sources_i = ipywidgets.Text(
            value='',
            description='water sample source',
            style={'description_width': 'initial'},
            layout=Layout(width='50%'))
        self.fluorometer_i = ipywidgets.Text(
            value='Horiba Aqualog',
            description='fluorometer',
            style={'description_width': 'initial'},
            layout=Layout(width='25%'))
        self.nSample_i = ipywidgets.Text(
            value=str(self.score_df.shape[0]),
            description='number of samples',
            style={'description_width': 'initial'},
            layout=Layout(width='25%'))
        self.dataset_calibration_i = ipywidgets.Text(
            value='Internal calibration: Raman Peak area, ' + 'Normalization by total fluorescence: '
                  + str(self.tf_normalization.value),
            description='dataset calibration',
            style={'description_width': 'initial'},
            layout=Layout(width='100%'))
        self.preprocess_i = ipywidgets.Text(
            value=self.generate_preprocess_info(),
            description='preprocessing method',
            style={'description_width': 'initial'},
            layout=Layout(width='100%'))
        self.decomposition_method_i = ipywidgets.Text(
            value=self.decomposition_method_list.value,
            description='decomposition method',
            style={'description_width': 'initial'},
            layout=Layout(width='100%'))
        self.validation_i = ipywidgets.Text(
            value='',
            description='validation method',
            style={'description_width': 'initial'},
            layout=Layout(width='100%'))
        self.description_i = ipywidgets.Textarea(
            value='',
            description='description',
            style={'description_width': 'initial'},
            layout=Layout(width='100%', height='100%'))
        self.button_output_interact = ipywidgets.Button(description='Save parafac model',
                                                        style={'description_width': 'initial'})

    def generate_preprocess_info(self):
        info = ''
        if self.inner_filter_effect.value:
            info += 'Inner_filter_effect, '
        if self.rayleigh_scattering_correction.value:
            info += 'Rayleigh scattering removal, '
        if self.raman_scattering_correction.value:
            info += 'Raman scattering removal, '
        if self.gaussian_smoothing.value:
            info += 'Gaussian smoothing.'
        return info

    def export_parafac_interact(self):
        export_parafac(self.filepath_i.value, self.score_df, self.exl_df, self.eml_df, self.name_i.value,
                       self.creator_i.value,
                       toolbox='EEM_python_toolkit',
                       date=self.date_i.value, fluorometer=self.fluorometer_i.value, nSample=self.nSample_i.value,
                       sources=self.sources_i.value,
                       dataset_calibration=self.dataset_calibration_i.value,
                       decomposition_method=self.decomposition_method_i.value,
                       preprocess=self.preprocess_i.value,
                       validation=self.validation_i.value, description=self.description_i.value)

    def generate_widgets(self):
        self.button_output_interact = interactive(self.export_parafac_interact,
                                                  {'manual': True, 'manual_name': 'Save parafac model'})
        output_items = [ipywidgets.Box([Label(value='Mandatory fields are marked with *')]),
                        ipywidgets.Box([self.filepath_i], layout=form_item_layout),
                        ipywidgets.Box([self.name_i, self.creator_i, self.date_i], layout=form_item_layout),
                        ipywidgets.Box([self.email_i], layout=form_item_layout),
                        ipywidgets.Box([self.sources_i, self.fluorometer_i, self.nSample_i], layout=form_item_layout),
                        ipywidgets.Box([self.dataset_calibration_i], layout=form_item_layout),
                        ipywidgets.Box([self.preprocess_i], layout=form_item_layout),
                        ipywidgets.Box([self.decomposition_method_i], layout=form_item_layout),
                        ipywidgets.Box([self.validation_i], layout=form_item_layout),
                        ipywidgets.Box([self.description_i], layout=form_item_layout),
                        ipywidgets.Box([self.button_output_interact], layout=form_item_layout),
                        ]
        return output_items
