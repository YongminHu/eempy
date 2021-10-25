import ipywidgets
from datetime import date
from ipywidgets import Layout, Label, interactive
from read_data import read_reference_from_text, string_to_float_list
from EEMprocessing import EEMstack, EEM_statistics, decomposition_interact, decomposition_reconstruction_interact,\
    export_parafac
from tensorly.decomposition import parafac, non_negative_parafac


form_item_layout = Layout(display='flex',
                          flex_flow='row',
                          justify_content='space-between')


# ----------------------Part 5. Data stack analysis-----------------------

# -------Tab1: File range selection----------

class Widgets51:
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

class Widgets52:
    def __init__(self, EEMstack_cw, datlist_cw, range1, range2, Em_range_cw, Ex_range_cw, timestamps_cw):
        self.EEMstack_cw = EEMstack_cw
        self.datlist_cw = datlist_cw
        self.range1 = range1
        self.range2 = range2
        self.Em_range_cw = Em_range_cw
        self.Ex_range_cw = Ex_range_cw
        self.timestamps_cw = timestamps_cw
        self.property_pixel = ipywidgets.Dropdown(options=['Timeseries analysis', 'Correlation analysis'],
                                                  description='Property')
        self.em_pixel = ipywidgets.FloatText(value=400, description='Em [nm]')
        self.ex_pixel = ipywidgets.FloatText(value=300, description='Ex [nm]')
        self.caption_pixel_statistics = ipywidgets.Label(value='For correlation analysis, please specify the reference '
                                                               'data with either a file path or a manual input')
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
        EEMstack_class_cw = EEMstack(self.EEMstack_cw[self.datlist_cw.index(self.range1.value):
                                                      self.datlist_cw.index(self.range2.value) + 1],
                                     self.Em_range_cw, self.Ex_range_cw)
        if self.property_pixel.value == 'Timeseries analysis':
            if self.timestamps_cw:
                EEMstack_class_cw.pixel_rel_std(Em=self.em_pixel.value, Ex=self.ex_pixel.value, plot=True,
                                                timestamp=self.timestamps_cw[self.datlist_cw.index(self.range1.value):
                                                                             self.datlist_cw.index(
                                                                                 self.range2.value) + 1],
                                                baseline=False, output=True)
            else:
                EEMstack_class_cw.pixel_rel_std(Em=self.em_pixel.value, Ex=self.ex_pixel.value, plot=True,
                                                timestamp=False, baseline=False, output=True)
        if self.property_pixel.value == 'Correlation analysis':
            if self.checkbox_reference_filepath_pixel.value:
                reference = read_reference_from_text(self.reference_filepath_pixel.value)
            if self.checkbox_reference_mannual_input_pixel.value:
                reference = string_to_float_list(self.reference_mannual_input_pixel.value)
            EEMstack_class_cw.pixel_linreg(Em=self.em_pixel.value, Ex=self.ex_pixel.value, x=reference)

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
                                      [self.checkbox_reference_mannual_input_pixel, self.reference_mannual_input_pixel],
                                      layout=form_item_layout),
                                  ipywidgets.Box([self.button_pixel_statistics])]
        return pixel_statistics_items


# ----------Tab3: EEM statistics---------

class Widgets53:
    def __init__(self, EEMstack_cw, datlist_cw, range1, range2, Em_range_cw, Ex_range_cw, timestamps_cw, crange_cw):
        self.EEMstack_cw = EEMstack_cw
        self.datlist_cw = datlist_cw
        self.range1 = range1
        self.range2 = range2
        self.Em_range_cw = Em_range_cw
        self.Ex_range_cw = Ex_range_cw
        self.timestamps_cw = timestamps_cw
        self.crange_cw = crange_cw
        self.property_eem = ipywidgets.Dropdown(options=['Mean', 'Standard deviation', 'Relative standard deviation',
                                                         'Correlation: Linearity', 'Correlation: Pearson coef.',
                                                         'Correlation: Spearman coef.'],
                                                description='Property', style={'description_width': 'initial'})
        self.caption_eem_statistics = ipywidgets.Label(value='For correlation analysis, please specify the reference '
                                                             'data with either a file path or a manual input')
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

    def EEM_statistics_interact(self, foo):
        EEMstack_class_cw = EEMstack(self.EEMstack_cw[self.datlist_cw.index(self.range1.value):
                                                      self.datlist_cw.index(self.range2.value) + 1],
                                     self.Em_range_cw, self.Ex_range_cw)
        reference = None
        header = None
        if self.property_eem.value == 'Correlation: Linearity' or self.property_eem.value == 'Correlation: Pearson coef.' \
                or self.property_eem.value == 'Correlation: Spearman coef.':
            if self.checkbox_reference_filepath_eem.value:
                reference, header = read_reference_from_text(self.reference_filepath_eem.value)
            if self.checkbox_reference_mannual_input_eem.value:
                reference = string_to_float_list(self.reference_mannual_input_eem.value)
        EEM_statistics(EEMstack_class_cw, term=self.property_eem.value, title=self.title_eem_statistics.value,
                       reference=reference, crange=self.crange_cw.value, reference_label=header)

    def generate_widgets(self):
        self.checkbox_reference_filepath_eem.observe(self.update_manual, 'value')
        self.checkbox_reference_mannual_input_eem.observe(self.update_filepath, 'value')
        self.button_eem_statistics.on_click(self.EEM_statistics_interact)
        eem_statistics_items = [ipywidgets.Box([self.property_eem]),
                                ipywidgets.Box([self.caption_eem_statistics], layout=form_item_layout),
                                ipywidgets.Box([self.checkbox_reference_filepath_eem, self.reference_filepath_eem],
                                               layout=form_item_layout),
                                ipywidgets.Box(
                                    [self.checkbox_reference_mannual_input_eem, self.reference_mannual_input_eem],
                                    layout=form_item_layout),
                                ipywidgets.Box([self.button_eem_statistics])]
        return eem_statistics_items


# -------Tab5: Stack decomposition----------

class Widgets55:
    def __init__(self, data_index, data_index_cw, timestamps_cw, EEMstack_cw, datlist_cw, range1, range2, Em_range_cw, Ex_range_cw):
        self.data_index = data_index
        self.data_index_cw = data_index_cw
        self.EEMstack_cw = EEMstack_cw
        self.datlist_cw = datlist_cw
        self.range1 = range1
        self.range2 = range2
        self.Em_range_cw = Em_range_cw
        self.Ex_range_cw = Ex_range_cw
        self.timestamps_cw = timestamps_cw
        self.rank_display = ipywidgets.IntText(value=4, description='Number of components',
                                               style={'description_width': 'initial'})
        self.button_decomposition_interact = ipywidgets.Button(description='Decompose',
                                                               style={'description_width': 'initial'})
        self.show_components = ipywidgets.Checkbox(value=False,
                                                   style={'description_width': 'initial'},
                                                   description='Plot components')
        self.show_loadings = ipywidgets.Checkbox(value=True,
                                                 style={'description_width': 'initial'},
                                                 description='Plot loadings')
        self.decomposition_method_list = ipywidgets.Dropdown(value='parafac',
                                                             options=['parafac', 'non_negative_parafac',
                                                                      'test_function'],
                                                             style={'description_width': 'initial'},
                                                             description='Decomposition method')
        self.show_normalized_score = ipywidgets.Checkbox(value=False,
                                                         style={'description_width': 'initial'},
                                                         description='Normalize the score by mean')
        self.show_normalized_component = ipywidgets.Checkbox(value=False,
                                                             style={'description_width': 'initial'},
                                                             description='Normalize the component so that the maxima '
                                                                         'intensity is equal to one',
                                                             layout=Layout(width='100%'))
        self.show_normalized_loadings = ipywidgets.Checkbox(value=True,
                                                            style={'description_width': 'initial'},
                                                            description='Normalize the loadings by their STD',
                                                            layout=Layout(width='100%'))

    def decomposition_interact_button(self):
        if not self.data_index:
            self.data_index_cw = self.timestamps_cw
        parafac_table, _, _, _, J_df, K_df = \
            decomposition_interact(self.EEMstack_cw[self.datlist_cw.index(self.range1.value):
                                                    self.datlist_cw.index(self.range2.value) + 1],
                                   self.Em_range_cw, self.Ex_range_cw, self.rank_display.value,
                                   index=self.data_index_cw[self.datlist_cw.index(self.range1.value):
                                                            self.datlist_cw.index(self.range2.value) + 1],
                                   decomposition_method=self.decomposition_method_list.value,
                                   plot_loadings=self.show_loadings.value,
                                   plot_components=self.show_components.value,
                                   score_normalization=self.show_normalized_score.value,
                                   loadings_normalization=self.show_normalized_loadings.value,
                                   component_normalization=self.show_normalized_component.value,
                                   component_autoscale=True,
                                   component_cmin=0, component_cmax=1, title=False, cbar=True)
        return parafac_table, J_df, K_df

    def generate_widgets(self):
        self.button_decomposition_interact = interactive(self.decomposition_interact_button,
                                                    {'manual': True, 'manual_name': 'Decompose'})
        decomposition_items = [
            ipywidgets.Box([self.rank_display, self.decomposition_method_list], layout=form_item_layout),
            ipywidgets.Box([Label(value='The number of components should be no more than the number of samples')]),
            self.show_components, self.show_loadings, self.show_normalized_loadings, self.show_normalized_component,
            self.show_normalized_score, self.button_decomposition_interact]
        return decomposition_items


# --------Tab6: Data reconstruction----------

class Widgets56:
    def __init__(self, decomposition_method_list, rank_display, crange_cw, data_index, data_index_cw, timestamps_cw, EEMstack_cw, datlist_cw,
                 range1, range2, Em_range_cw, Ex_range_cw):
        self.data_index = data_index
        self.data_index_cw = data_index_cw
        self.EEMstack_cw = EEMstack_cw
        self.datlist_cw = datlist_cw
        self.range1 = range1
        self.range2 = range2
        self.Em_range_cw = Em_range_cw
        self.Ex_range_cw = Ex_range_cw
        self.timestamps_cw = timestamps_cw
        self.decomposition_method_list = decomposition_method_list
        self.rank_display = rank_display
        self.crange_cw = crange_cw
        self.data_to_view = ipywidgets.Dropdown(options=datlist_cw[datlist_cw.index(range1.value):datlist_cw.index(range2.value)+1],
                                                description='Select the data for reconstruction',
                                                style={'description_width': 'initial'},
                                                layout={'width':'max-content'})
        self.button_decomposition_re_interact= ipywidgets.Button(description='Reconstruct',
                                                                 style={'description_width': 'initial'})

    def decomposition_interact_re_button(self, foo):
        dataset = self.EEMstack_cw[self.datlist_cw.index(self.range1.value):
                                   self.datlist_cw.index(self.range2.value)+1]
        if self.decomposition_method_list.value=='parafac':
            factors = parafac(dataset, rank=self.rank_display.value)
        elif self.decomposition_method_list.value=='non_negative_parafac':
            factors = non_negative_parafac(dataset, rank=self.rank_display.value)
        elif self.decomposition_method_list.value=='test_function':
            factors = non_negative_parafac(dataset, rank=self.rank_display.value, fixed_modes=[0,1], init="random")
        I_0 = factors[1][0]
        J_0 = factors[1][1]
        K_0 = factors[1][2]
        decomposition_reconstruction_interact(I_0, J_0, K_0, self.EEMstack_cw[self.datlist_cw.index(self.data_to_view.value)],
                                              self.Em_range_cw, self.Ex_range_cw,
                                              self.datlist_cw[self.datlist_cw.index(self.range1.value):
                                                              self.datlist_cw.index(self.range2.value)+1],
                                              self.data_to_view.value, crange=self.crange_cw.value)

    def generate_widgets(self):
        self.button_decomposition_re_interact.on_click(self.decomposition_interact_re_button)
        decomposition_reconstruction_items = [ipywidgets.Box([
            Label(value='Please first specify the number of components and decomposition method in tab-5 '
                        '"PARAFAC decomposition"')]),
            ipywidgets.Box([self.data_to_view, self.button_decomposition_re_interact],
                           layout=form_item_layout)]
        return decomposition_reconstruction_items


# ----------------------Part 6. Save PARAFAC result-----------------------

class Widgets6:
    def __init__(self, I_df, J_df, K_df, inner_filter_effect, scattering_correction, gaussian_smoothing, decomposition_method_list):
        self.I_df = I_df
        self.J_df = J_df
        self.K_df = K_df
        self.inner_filter_effect = inner_filter_effect
        self.scattering_correction = scattering_correction
        self.gaussian_smoothing = gaussian_smoothing
        self.decomposition_method_list = decomposition_method_list
        self.filepath_i = ipywidgets.Text(
            value='C:/Users/YongminHu/Documents/PhD/Fluo-detect/_data/GW_RT_MT/GW_RT_P3.txt',
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
            value=str(self.I_df.shape[0]),
            description='number of samples',
            style={'description_width': 'initial'},
            layout=Layout(width='25%'))
        self.dataset_calibration_i = ipywidgets.Text(
            value='Internal calibration: Raman Peak area',
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
        if self.scattering_correction.value:
            info += 'Rayleigh scattering masking, '
        if self.gaussian_smoothing.value:
            info += 'Gaussian smoothing.'
        return info

    def export_parafac_interact(self):
        export_parafac(self.filepath_i.value, self.I_df, self.J_df, self.K_df, self.name_i.value, self.creator_i.value,
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

    # return filepath_i, name_i, creator_i, email_i, date_i, sources_i, fluorometer_i, nSample_i, dataset_calibration_i, \
    #        preprocess_i, decomposition_method_i, validation_i, description_i
