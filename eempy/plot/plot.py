"""
Functions for plotting EEM-related data
Author: Yongmin Hu (yongminhu@outlook.com)
Last update: 2024-02-13
"""

from eempy.utils import *
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import pandas as pd
import math
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from matplotlib.colors import LogNorm, TABLEAU_COLORS



def plot_eem(intensity, ex_range, em_range, auto_intensity_range=True, scale_type='linear', vmin=0, vmax=10000,
             n_cbar_ticks=5, cbar=True, cmap='jet', figure_size=(10, 7), label_font_size=20,
             cbar_label="Intensity (a.u.)", cbar_font_size=16, fix_aspect_ratio=True, rotate=False,
             plot_tool='matplotlib'):
    """
    plot EEM or EEM-like data.

    Parameters
    ----------------
    intensity: np.ndarray (2d)
        The EEM.
    ex_range: np.ndarray (1d)
        The excitation wavelengths.
    em_range: np.ndarray (1d)
        The emission wavelengths.
    auto_intensity_range: bool
        Whether to use the colorbar range generated automatically by matplotlib for plotting fluorescence intensity.
    scale_type: str, {'linear', 'log'}
        The type of colorbar scale used for plotting fluorescence intensity.
    vmin: int or float
        The minimum intensity displayed, if you wish to adjust the intensity scale yourself (autoscale=False).
    vmax: int or float
        The maximum intensity displayed, if you wish to adjust the intensity scale yourself (autoscale=False).
    n_cbar_ticks: int
        The number of ticks in colorbar.
    cmap: str
        The colormap, see https://matplotlib.org/stable/users/explain/colors/colormaps.html.
    figure_size: tuple or list with two elements
        The figure size.
    label_font_size: int
        The fontsize of the x and y axes labels.
    cbar: bool
        Whether to plot the colorscale bar.
    cbar_label: str
        The label of the colorbar scale.
    cbar_font_size: int
        The label size of the colorbar scale.
    fix_aspect_ratio: bool
        Whether to fix the aspect ratio to be one.
    rotate: bool
        Whether to rotate the EEM, so that the x-axis is excitation and y-axis is emission.
    plot_tool: str, {'matplotlib', 'plotly'}
        Which python package to use for plotting.

    Returns
    ----------------
    fig：matplotlib figure
    ax: array of matplotlib axes
    """
    if plot_tool == 'matplotlib':
        fig, ax = plt.subplots(figsize=figure_size)
        font = {'size': label_font_size}
        plt.rc('font', **font)
        # reset the axis direction
        if scale_type == 'log':
            c_norm = LogNorm(vmin=vmin, vmax=vmax)
            t_cbar = np.logspace(math.log(vmin), math.log(vmax), n_cbar_ticks)
        else:
            c_norm = None
            t_cbar = np.linspace(vmin, vmax, n_cbar_ticks)
        if not rotate:
            extent = (em_range.min(), em_range.max(), ex_range.min(), ex_range.max())
            ax.set_ylim([ex_range[0], ex_range[-1]])
        else:
            extent = (ex_range.min(), ex_range.max(), em_range.min(), em_range.max())
            ax.set_xlim([ex_range[0], ex_range[-1]])
        ax.set_xlabel('Emission wavelength [nm]' if not rotate else 'Excitation wavelength [nm]')
        ax.set_ylabel('Excitation wavelength [nm]' if not rotate else 'Emission wavelength [nm]')
        if not auto_intensity_range:
            if scale_type == 'log':
                im = ax.imshow(intensity if not rotate else np.flipud(np.fliplr(intensity.T)), cmap=cmap,
                               interpolation='none', extent=extent, origin='upper',
                               aspect=1 if fix_aspect_ratio else None, norm=c_norm)
            else:
                im = ax.imshow(intensity if not rotate else np.flipud(np.fliplr(intensity.T)), cmap=cmap,
                               interpolation='none', extent=extent, vmin=vmin, vmax=vmax,
                               origin='upper', aspect=1 if fix_aspect_ratio else None, norm=c_norm)
        else:
            im = ax.imshow(intensity if not rotate else np.flipud(np.fliplr(intensity.T)), cmap=cmap,
                           interpolation='none', extent=extent, origin='upper',
                           aspect=1 if fix_aspect_ratio else None)
        if cbar:
            cbar = fig.colorbar(im, ax=ax, ticks=t_cbar, fraction=0.03, pad=0.04)
            cbar.set_label(cbar_label, labelpad=1.5)
            cbar.ax.tick_params(labelsize=cbar_font_size)
        return fig, ax

    elif plot_tool == 'plotly':

        if scale_type == 'log':
            t_cbar = np.logspace(math.log(vmin), math.log(vmax), n_cbar_ticks)
            trace = go.Heatmap(z=np.log10(intensity) if not rotate else np.flipud(np.fliplr(np.log10(intensity).T)),
                               x=em_range if not rotate else ex_range,
                               y=ex_range[::-1] if not rotate else em_range[::-1],
                               colorscale=cmap,
                               zmin=vmin if not auto_intensity_range else None,
                               zmax=vmax if not auto_intensity_range else None,
                               colorbar=dict(title=cbar_label, tickvals=np.log10(t_cbar), ticktext=t_cbar,
                                             tickfont=cbar_font_size) if not
                               auto_intensity_range else dict(title=cbar_label, tickfont=cbar_font_size))

        elif scale_type == 'linear':
            trace = go.Heatmap(
                z=intensity if not rotate else np.flipud(np.fliplr(intensity.T)),
                x=em_range if not rotate else ex_range,
                y=ex_range[::-1] if not rotate else em_range[::-1],
                colorscale=cmap,
                zmin=vmin if not auto_intensity_range else None,
                zmax=vmax if not auto_intensity_range else None,
                colorbar=dict(title=cbar_label,
                              tickvals=np.linspace(vmin, vmax, n_cbar_ticks),
                              tickfont=dict(size=cbar_font_size)) if not auto_intensity_range
                else dict(title=cbar_label,
                          tickfont=dict(size=cbar_font_size)))

        xaxis_title = 'Emission wavelength [nm]' if not rotate else 'Excitation wavelength [nm]'
        yaxis_title = 'Excitation wavelength [nm]' if not rotate else 'Emission wavelength [nm]'
        layout = go.Layout(
            xaxis=dict(title=xaxis_title),
            yaxis=dict(title=yaxis_title),
            font=dict(size=label_font_size),
            width=figure_size[0] * 100,
            height=figure_size[1] * 100,
            yaxis_scaleanchor="x" if fix_aspect_ratio else None,
            autosize=True
        )

        fig = go.Figure(data=[trace], layout=layout)

        return fig


def plot_abs(absorbance, ex_range, xmax=0.05, ex_range_display=(200, 800), plot_tool='matplotlib'):
    """
    Plot the UV absorbance data

    Parameters
    ----------------
    absorbance: np.ndarray (1d)
        The absorbance.
    ex_range: np.ndarray (1d)
        The excitation wavelengths.
    xmax: float (0~1)
        The maximum absorbance displayed.
    ex_range_display: tuple with two elements
        The range of excitation wavelengths displayed.
    plot_tool: str, {'matplotlib', 'plotly'}
        Which python package to use for plotting.

    Returns
    ----------------
    fig：matplotlib figure
    ax: array of matplotlib axes
    """
    if plot_tool == 'matplotlib':
        fig, ax = plt.subplots(figsize=(6.5, 2))
        ax.plot(ex_range, absorbance)
        ax.set_xlim(ex_range_display)
        ax.set_ylim([0, xmax])
        ax.set_xlabel('Wavelength [nm]', fontsize=14)
        ax.set_ylabel('Absorbance [a.u.]', fontsize=14)
        ax.tick_params(labelsize=14)
        return fig, ax
    elif plot_tool == 'plotly':
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ex_range, y=absorbance, mode='lines', name='Absorbance'))
        fig.update_xaxes(title_text='Wavelength [nm]', range=ex_range_display)
        fig.update_yaxes(title_text='Absorbance [a.u.]', range=[0, xmax])
    return fig


def plot_fi(fi: pd.DataFrame, q: float = 0.05):
    """
    Plot the fluorescence intensities of samples at a specific pair of ex/em.

    Parameters
    ----------
    fi: pandas DataFrame
        A DataFrame of shape (n,1), where n is the number of samples. The output of EEMdataset.peak_picking() can be
        passed to this parameter.
    q: float
        Where to plot the reference lines. By default, this is set to be 0.05 - two horizontal lines will be plotted at
        95% and 105% of the average fluorescence intensity.

    Returns
    ----------
    fig：matplotlib figure
    ax: array of matplotlib axes
    """
    index = fi.index
    fi = fi.to_numpy()
    rel_std = stats.variation(fi)[0]
    std = np.std(fi)
    fi_mean = fi.mean()
    ql = abs(q * fi_mean)
    fig, ax = plt.subplots(figsize=(15, 5))
    p = ax.plot(index, fi, markersize=13)
    m0 = ax.axhline(fi_mean, linestyle='--', label='mean', c='black')
    mq_u = ax.axhline(fi_mean + ql, linestyle='--', label='+3%', c='red')
    mq_l = ax.axhline(fi_mean - ql, linestyle='--', label='-3%', c='blue')
    if max(fi) > fi_mean + ql or min(fi) < fi_mean - ql:
        ax.set_ylim(min(fi) - ql, max(fi) + ql)
    ax.legend([m0, mq_u, mq_l], ['mean', '+3%', '-3%'], prop={'size': 10})
    ax.tick_params(axis='x', rotation=90)
    ax.text(0.5, 1.2, f"Mean: {fi_mean:.4f}", ha='center', transform=ax.transAxes)
    ax.text(0.5, 1.125, f"Standard deviation: {std:.4f}", ha='center', transform=ax.transAxes)
    ax.text(0.5, 1.05, f"Relative Standard deviation: {rel_std:.4f}", ha='center', transform=ax.transAxes)
    return fig, ax


def plot_fi_correlation(fi: pd.DataFrame, ref):
    """
    Plot fluorescence intensity versus the reference value.

    Parameters
    ----------
    fi: pandas DataFrame
        A DataFrame of shape (n,1), where n is the number of samples. The output of EEMdataset.peak_picking() can be
        passed to this parameter.
    ref: np.ndarray (1d)
        The reference value. It should be an 1d numpy array of shape (n,), where n is the number of samples.

    Returns
    ----------
    fig：matplotlib figure
    ax: array of matplotlib axes
    """
    fi_2d = fi.to_numpy()
    fi = fi_2d.reshape(-1)
    x = ref
    x_2d = x.reshape(ref.shape[0], 1)
    reg = LinearRegression()
    reg.fit(x_2d, fi_2d)
    w = reg.coef_[0][0]
    b = reg.intercept_[0]
    r2 = reg.score(x_2d, fi)
    coef_p, p_value_p = stats.pearsonr(x, fi)
    coef_s, p_value_s = stats.spearmanr(x, fi)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.scatter(x, fi)
    p = np.array([x.min(), x.max()])
    q = w * p + b
    ax.plot(p, q)
    ax.set_xlabel('Reference', fontdict={"size": 14})
    ax.set_ylabel('Fluorescence Intensity', fontdict={"size": 14})
    ax.tick_params(labelsize=14)
    ax.text(0.5, 1.3, f"Linear regression model: y={w:.4f}" + f"x+{b:.4f}", ha='center', transform=ax.transAxes,
            fontsize=12)
    ax.text(0.5, 1.225, f"Linear regression R2:', '{r2:.4f}", ha='center', transform=ax.transAxes, fontsize=12)
    ax.text(0.5, 1.15, f"Pearson coefficient: {coef_p:.4f} (p-value = {p_value_p:.4e})", ha='center',
            transform=ax.transAxes, fontsize=12)
    ax.text(0.5, 1.075, f"Spearman coefficient: {coef_s:.4f} (p-value = {p_value_s:.4e})", ha='center',
            transform=ax.transAxes, fontsize=12)
    return fig, ax


def plot_loadings(model_dict: dict, colors=TABLEAU_COLORS.values(), plot_tool='matplotlib'):
    """
    Plot the excitation and emission loadings for PARAFAC models.

    Parameters
    ----------
    model_dict: dict
        A dictionary of PARAFAC objects. Each PARAFAC model is labelled with the dictionary key.
    colors: format for matplotlib color
        A list of colors used for plotting.
    plot_tool: str, {'matplotlib', 'plotly'}
        Which python package to use for plotting.

    Returns
    -------
    fig：matplotlib figure
    ax: array of matplotlib axes
    """
    component_labels = []
    for model in model_dict.values():
        component_labels.append(model.score.columns)
    component_labels = sorted(list(set(component_labels)))
    num_rows = len(component_labels) // 4
    num_cols = 4

    if plot_tool == 'matplotlib':
        fig, ax = plt.subplots(num_rows, len(component_labels)%4,
                               figsize=(10, 2.7*num_rows), sharey='row')
        fig.subplots_adjust(wspace=0, hspace=0.7)

        for i, (model_label, model) in enumerate(model_dict.items()):
            for r, component in enumerate(model.ex_loadings.columns):
                order = component_labels.index(component)
                ax[order//4, order%4].plot(model.ex_loadings.index, model.ex_loadings.iloc[:, r], '--',
                                           label=model_label+'-ex', c=colors[i])
                ax[order//4, order%4].plot(model.em_loadings.index, model.em_loadings.iloc[:, r], label=model_label+'-em',
                                           c=colors[i])

        for j in range(len(component_labels)):
            ax[j//4, j%4].text(0.8, 0.85, component_labels[j], transform=ax[j//4, j%4].transAxes, fontsize=18)

        leg_ax = fig.add_subplot(111)
        leg_ax.axis('off')
        handles, labels = ax[0].get_legend_handles_labels()
        leg_ax.legend(flip_legend_order(handles, 3), flip_legend_order(labels, 3), loc='upper center',
                      bbox_to_anchor=(1.23, 0.88), fontsize=11, ncol=3)
        fig.text(0.4, -0.15, 'Wavelength (nm)', fontsize=18)
        fig.text(0.07, 0.35, 'Loadings', fontsize=18, rotation='vertical')
        return fig, ax

    elif plot_tool == 'plotly':
        num_rows = len(component_labels) // 4
        num_cols = 4
        fig = make_subplots(rows=num_rows, cols=num_cols, subplot_titles=component_labels)

        for i, (model_label, model) in enumerate(model_dict.items()):
            for r, component in enumerate(model.ex_loadings.columns):
                order = component_labels.index(component)
                fig.add_trace(go.Scatter(x=model.ex_loadings.index, y=model.ex_loadings.iloc[:, r],
                                         mode='lines', name=f"{model_label}-ex",
                                         line=dict(color=colors[i], dash='dash')),
                              row=order // num_cols + 1, col=order % num_cols + 1)
                fig.add_trace(go.Scatter(x=model.em_loadings.index, y=model.em_loadings.iloc[:, r],
                                         mode='lines', name=f"{model_label}-em", line=dict(color=colors[i])),
                              row=order // num_cols + 1, col=order % num_cols + 1)

        for j in range(len(component_labels)):
            fig.update_xaxes(title_text='Wavelength (nm)', row=j // num_cols + 1, col=j % num_cols + 1)
            fig.update_yaxes(title_text='Loadings', row=j // num_cols + 1, col=j % num_cols + 1)

        fig.update_layout(title='Excitation and Emission Loadings for PARAFAC Models',
                          title_x=0.5, height=600 * num_rows)


def plot_loadings(parafac_models_dict: dict, colors=TABLEAU_COLORS.values(), component_labels_dict=None,
                  plot_tool='matplotlib'):
    """
    Plot the excitation and emission loadings for PARAFAC models.

    Parameters
    ----------
    parafac_models_dict: dict
        A dictionary of PARAFAC objects. Each PARAFAC model is labelled with a dictionary key.
    colors: format for matplotlib color
        A list of colors used for plotting.
    component_labels_dict: dict or None
        A dictionary of component names for each PARAFAC model. The keys of the dict should be identical to those of the
        parafac_models_dict. The value of each key is a list of component names.
    plot_tool: str, {'matplotlib', 'plotly'}
        Which python package to use for plotting.

    Returns
    -------
    fig：matplotlib figure
    ax: array of matplotlib axes
    """

    # Determine layout
    if component_labels_dict:
        n_tot_components = len(set(component_labels_dict.values()))
    else:
        n_tot_components = max([m.rank for m in parafac_models_dict.values()])
    n_rows = n_tot_components//4+1
    n_cols = 4

    if plot_tool == 'matplotlib':
        fig, ax = plt.subplots(n_rows, 4, figsize=(10, 2.7*n_rows), sharey='row', sharex='col')
        fig.subplots_adjust(wspace=0, hspace=0)

        for k, (model_name, model) in enumerate(parafac_models_dict.items()):
            for i in range(model.rank):
                component_label = component_labels_dict[model_name][i]
                pos = list(set(component_labels_dict.values())).index(component_label)
                ax[pos//n_cols, pos%n_cols].plot(model.ex_range, model.ex_loadings.iloc[:, i], '--', c=colors[k])
                ax[pos//n_cols, pos%n_cols].plot(model.em_range, model.em_loadings.iloc[:, i], label=model_name,
                                                 c=colors[k])
                ax[pos//n_cols, pos%n_cols].tick_params(labelsize=14)

        for j in range(n_tot_components):
            if component_labels_dict:
                ax[j].text(0.8, 0.85, list(set(component_labels_dict.values()))[j], transform=ax[i].transAxes,
                           fontsize=18)
            else:
                ax[j].text(0.8, 0.85, 'C{i}'.format(i=i+1), transform=ax[i].transAxes, fontsize=18)

        leg_ax = fig.add_subplot(111)
        leg_ax.axis('off')
        handles, labels = ax[0].get_legend_handles_labels()
        leg_ax.legend(flip_legend_order(handles,5), flip_legend_order(labels,5),
                      loc='upper center', bbox_to_anchor=(0.5, -0.35), fontsize=14, ncol=5)
        fig.text(0.4, -0.15, 'Wavelength (nm)', fontsize=18)
        fig.text(0.07, 0.35, 'Loadings', fontsize=18, rotation='vertical')
        return fig, ax

    elif plot_tool == 'plotly':
        fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=list(set(component_labels_dict.values())))
        for k, (model_name, model) in enumerate(parafac_models_dict.items()):
            for i in range(model.rank):
                component_label = component_labels_dict[model_name][i]
                pos = list(set(component_labels_dict.values())).index(component_label)
                fig.add_trace(go.Scatter(x=model.ex_range, y=model.ex_loadings.iloc[:, i],
                                         mode='lines', name=model_name,
                                         line=dict(color=colors[k], dash='dash')),
                              row=pos // n_cols + 1, col=pos % n_cols + 1)
                fig.add_trace(go.Scatter(x=model.em_loadings.index, y=model.em_loadings.iloc[:, i],
                                         mode='lines', name=model_name, line=dict(color=colors[i])),
                              row=pos // n_cols + 1, col=pos % n_cols + 1)

        # for j in range(n_tot_components):
        #     fig.update_xaxes(title_text='Wavelength (nm)', row=j // num_cols + 1, col=j % num_cols + 1)
        #     fig.update_yaxes(title_text='Loadings', row=j // num_cols + 1, col=j % num_cols + 1)
        return fig


