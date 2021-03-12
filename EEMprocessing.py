"""
Functions for fluorescence python toolkit
Author: Yongmin Hu (yongmin.hu@eawag.ch, yongminhu@outlook.com)
Last update: 2021-03-07
"""

from read_data import *
import scipy.stats as stats
import random
import cv2
import numpy as np
from sklearn.linear_model import LinearRegression
from matplotlib.colors import LogNorm, Normalize
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max
from scipy import interpolate
from datetime import datetime, timedelta
from tensorly.decomposition import parafac, non_negative_parafac
import pandas as pd
from IPython.display import display
import itertools
from pandas.plotting import register_matplotlib_converters
from scipy.sparse.linalg import ArpackError

register_matplotlib_converters()

def stackDat(datdir, kw, timestamp_reference_list=[], **par):
    datlist = get_filelist(datdir, kw)
    if timestamp_reference_list:
        datlist = [x for _, x in sorted(zip(timestamp_reference_list, datlist))]
    n = 0
    for f in datlist:
        path = datdir + '/' + f
        intensity, Em_range, Ex_range = readEEM(path)
        if 'filter_type' in par.keys():
            if par['filter_type'] == 'gaussian':
                intensity = gaussian_filter(intensity, sigma=par['sigma'])
        if n == 0:
            num_datfile = len(datlist)
            EEMstack = np.zeros([num_datfile, intensity.shape[0], intensity.shape[1]])
        try:
            EEMstack[n, :, :] = intensity
        except ValueError:
            print('Check data dimension: ', f)
        n += 1
    return EEMstack, Em_range, Ex_range, datlist


def stackABS(datdir, datlist):
#    datlist = get_filelist(datdir, kw)
    n = 0
    for f in datlist:
        path = datdir + '/' + f
        absorbance, Ex_range = readABS(path)
        if n==0:
            num_datfile = len(datlist)
            ABSstack = np.zeros([num_datfile, absorbance.shape[0]])
        ABSstack[n,:] = absorbance
        n += 1
    return ABSstack, Ex_range, datlist


def mask_low_values(EEMstack, threshold, interpolation):
    EEMstack[EEMstack < threshold] = interpolation
    return EEMstack


def dichotomy_search(nums, target):
    start = 0
    end = len(nums) - 1
    if target < min(nums):
        return np.where(nums==min(nums))[0][0]
    if target > max(nums):
        return np.where(nums==max(nums))[0][0]
    while start <= end:
        mid = (start + end) // 2
        fdiff = nums[mid] - target
        bdiff = nums[mid - 1] - target
        if fdiff * bdiff <= 0:
            if abs(fdiff) < abs(bdiff):
                return mid
            if abs(bdiff) <= abs(fdiff):
                return mid - 1
        elif nums[mid] < target:
            start = mid + 1
        else:
            end = mid - 1


def EEM_mask(intensity, Em_range, Ex_range, ref_matrix, threshold, residual, plot=False, cmin=0, cmax=4000):
    mask = np.ones(ref_matrix.shape)
    extent = [Em_range.min(), Em_range.max(), Ex_range.min(), Ex_range.max()]
    if residual == 'big':
        mask[np.where(ref_matrix < threshold)] = np.nan
    if residual == 'small':
        mask[np.where(ref_matrix > threshold)] = np.nan
    if plot:
        plt.figure(figsize=(8,8))
        plot3DEEM(intensity, Em_range=Em_range, Ex_range=Ex_range, autoscale=False, cmin=cmin, cmax=cmax)
        plt.imshow(mask, extent=extent, alpha=0.9, cmap="binary")
        # plt.title('Relative STD<{threshold}'.format(threshold=threshold))
    return mask


def EEM_set_region_to_zero(intensity, Em_range, Ex_range, Em_min, Em_max, Ex_min, Ex_max):
    zero_mask = np.zeros(intensity.shape)
    Em_min_idx = dichotomy_search(Em_range, Em_min)
    Em_max_idx = dichotomy_search(Em_range, Em_max)
    Ex_min_idx = dichotomy_search(Ex_range, Ex_min)
    Ex_max_idx = dichotomy_search(Ex_range, Ex_max)
    zero_mask[Ex_range.shape[0]-Ex_max_idx-1:Ex_range.shape[0]-Ex_min_idx, Em_min_idx:Em_max_idx]=1
    intensity[np.where(zero_mask==0)]=0
    return intensity


def EEMs_gaussianfilter(EEMstack, sigma=1, truncate=3):
    EEMstack_filtered = np.zeros(EEMstack.shape)
    for i in range(EEMstack.shape[0]):
        EEMstack_filtered[i] = gaussian_filter(EEMstack[i], sigma=sigma, truncate=truncate)
    return EEMstack_filtered


def EEM_cutting(intensity, Em_range, Ex_range, Em_min, Em_max, Ex_min, Ex_max):
    Em_min_idx = dichotomy_search(Em_range, Em_min)
    Em_max_idx = dichotomy_search(Em_range, Em_max)
    Ex_min_idx = dichotomy_search(Ex_range, Ex_min)
    Ex_max_idx = dichotomy_search(Ex_range, Ex_max)
    intensity_cut = intensity[Ex_range.shape[0]-Ex_max_idx-1:Ex_range.shape[0]-Ex_min_idx, Em_min_idx:Em_max_idx]
    Em_range_cut = Em_range[Em_min_idx:Em_max_idx+1]
    Ex_range_cut = Ex_range[Ex_min_idx:Ex_max_idx+1]
    return intensity_cut, Em_range_cut, Ex_range_cut


def EEMs_cutting(EEMstack, Em_range, Ex_range, Em_min=250, Em_max=810, Ex_min=230, Ex_max=500):
    for i in range(EEMstack.shape[0]):
        intensity = np.array(EEMstack[i,:,:])
        intensity_cut, Em_range_cut, Ex_range_cut = EEM_cutting(intensity, Em_range, Ex_range,
                                                                Em_min=Em_min, Em_max=Em_max, Ex_min=Ex_min,
                                                                Ex_max=Ex_max)
        intensity_cut = np.array([intensity_cut])
        if i==0:
            EEMstack_cut = intensity_cut
        if i>0:
            EEMstack_cut = np.concatenate([EEMstack_cut, intensity_cut], axis=0)
    return EEMstack_cut, Em_range_cut, Ex_range_cut


def EEMs_scattering_correction(EEMstack, Em_range, Ex_range, scattering_interpolation='zero',
                               tolerance=10):
    EEMstack_r = np.zeros(EEMstack.shape)
    for i in range(EEMstack.shape[0]):
        intensity = np.array(EEMstack[i, :, :])
        intensity_masked, rayleigh_mask = rayleigh_masking(intensity, Em_range, Ex_range, tolerance=tolerance,
                                                           interpolation=scattering_interpolation)
        intensity_masked = np.array([intensity_masked])
        EEMstack_r[i] = intensity_masked
    return EEMstack_r


def EEMs_contour_masking(EEMstack, Em_range, Ex_range, otsu=True, binary_threshold=50):
    EEMstack_m = np.zeros(EEMstack.shape)
    for i in range(EEMstack.shape[0]):
        intensity = np.array(EEMstack[i, :, :])
        intensity_binary = contour_detection(intensity, Em_range=Em_range, Ex_range=Ex_range, otsu=otsu,
                                                 plot=False, binary_threshold=binary_threshold)
        intensity[np.where(intensity_binary==0)]=0
        intensity = np.array([intensity])
        EEMstack_m[i] = intensity
    return EEMstack_m


def EEM_inner_filter_effect(intensity, Em_range, Ex_range, absorbance, Ex_range2, cuvette_length=1, Ex_lower_limit=200,
                             Ex_upper_limit=825):
    Ex_range2 = np.concatenate([[Ex_upper_limit], Ex_range2, [Ex_lower_limit]])
    absorbance = np.concatenate([[0], absorbance, [max(absorbance)]])
    f1 = interpolate.interp1d(Ex_range2, absorbance, kind='linear')
    absorbance_ex = np.fliplr(np.array([f1(Ex_range)]))
    absorbance_em = np.array([f1(Em_range)])
    ife_factors = 10**(cuvette_length*(absorbance_ex.T.dot(np.ones(absorbance_em.shape)) +
                                       np.ones(absorbance_ex.shape).T.dot(absorbance_em)))
#    plot3DEEM(ife_factors,Em_range, Ex_range, autoscale=False, cmax=2, cmin=1)
    intensity_filtered = intensity * ife_factors
    return intensity_filtered


def EEMs_inner_filter_effect(EEMstack, ABSstack, Em_range, Ex_range, Ex_range2):
    EEMstack_filtered = np.array(EEMstack)
    for i in range(EEMstack.shape[0]):
        intensity = EEMstack[i]
        absorbance = ABSstack[i]
        EEMstack_filtered[i] = EEM_inner_filter_effect(intensity, Em_range, Ex_range, absorbance, Ex_range2)
    return EEMstack_filtered


def EEMs_set_region_to_zero(EEMstack, Em_range, Ex_range, Em_min=250, Em_max=550, Ex_min=230, Ex_max=450):
    for i in range(EEMstack.shape[0]):
        intensity = np.array(EEMstack[i,:,:])
        intensity_masked = EEM_set_region_to_zero(intensity, Em_range, Ex_range, Em_min=Em_min, Em_max=Em_max,
                                                  Ex_min=Ex_min, Ex_max=Ex_max)
        intensity_masked = np.array([intensity_masked])
        if i==0:
            EEMstack_masked = intensity_masked
        if i>0:
            EEMstack_masked = np.concatenate([EEMstack_masked, intensity_masked], axis=0)
    return EEMstack_masked


def detect_local_peak(intensity, min_distance=20, plot=False):
    coordinates = peak_local_max(intensity, min_distance=min_distance)
    if plot:
        plt.plot(coordinates[:, 1], coordinates[:, 0], 'r.')
        plt.axis('off')
    return coordinates


def EEMs_random_selection(EEMstack):
    i = random.randrange(EEMstack.shape[0])
    EEM_selected = EEMstack[i]
    return EEM_selected


def get_TS_from_filename(filename, ts_format='%Y-%m-%d-%H-%M-%S', ts_start_position=0, ts_end_position=19):
    ts_string = filename[ts_start_position:ts_end_position]
    ts = datetime.strptime(ts_string, ts_format)
    return ts


def datetime_to_str(datetime_list, output=False, filename='timestamp.txt'):
    tsstr = [datetime_list[i].strftime("%Y-%m-%d-%H-%M") for i in range(len(datetime_list))]
    if output:
        file= open(filename, 'w')
        for fp in tsstr:
            file.write(str(fp))
            file.write('\n')
        file.close()
    return tsstr


def ABS_plot_at_wavelength(ABSstack, Ex_range, Ex, plot=True, timestamp=False):
    Ex_ref = dichotomy_search(Ex_range, Ex)
    y = ABSstack[:, Ex_ref]
    x = np.array(range(1, ABSstack.shape[0] + 1))
    if plot:
        if timestamp:
            x = timestamp
        plt.figure(figsize=(15, 5))
        plt.scatter(x, y)
        plt.xlim([x[0]-timedelta(hours=1), x[-1]+timedelta(hours=1)])

class EEMstack:
    def __init__(self, intensities, Em_range, Ex_range):
        # The Em/Ex ranges should be sorted in ascending order
        self.intensities = intensities
        self.Em_range = Em_range
        self.Ex_range = Ex_range

    def zscore(self):
        transformed_data = stats.zscore(self.intensities, axis=0)
        return transformed_data

    def mean(self):
        mean = np.mean(self.intensities, axis=0)
        return mean

    def variance(self):
        variance = np.var(self.intensities, axis=0)
        return variance

    def rel_std(self, threshold=0.05):
        coef_variation = stats.variation(self.intensities, axis=0)
        rel_std = abs(coef_variation)
        if threshold:
            mean = np.mean(self.intensities, axis=0)
            # mask = EEM_mask(mean, Em_range=self.Em_range, Ex_range=self.Ex_range,
            #          ref_matrix=coef_variation, threshold=threshold, residual='big')
            qualified_pixel_proportion = np.count_nonzero(rel_std<threshold) / np.count_nonzero(~np.isnan(rel_std))
            print("The proportion of pixels with relative STD < {t}: ".format(t=threshold),
                  qualified_pixel_proportion)
        return rel_std

    def std(self):
        return np.std(self.intensities, axis=0)

    def pixel_rel_std(self, Em, Ex, plot=True, timestamp=False, baseline=False, output=True):
        M = self.intensities
        Em_ref = dichotomy_search(self.Em_range, Em)
        Ex_ref = dichotomy_search(self.Ex_range, Ex)
        y = M[:, self.Ex_range.shape[0]-Ex_ref-1, Em_ref]
        cod = (self.Em_range[Em_ref], self.Ex_range[Ex_ref])
        rel_std = stats.variation(y)
        std = np.std(y)
        x = np.array(range(1, M.shape[0]+1))
        y_mean = y.mean()
        q5 = abs(0.05*y_mean)
        q3 = abs(0.03*y_mean)
        print("Mean: {mean}".format(mean=y_mean))
        print("Standard deviation: {std}".format(std=std))
        print("Relative Standard deviation: {rel_std}".format(rel_std=rel_std))
        if timestamp:
            x = timestamp
        if output:
            table = pd.DataFrame(y)
            table.index = x
            table.columns = ['Intensity (Ex/Em = {ex}/{em})'.format(ex=cod[1], em=cod[0])]
            display(table)
        if plot:
            plt.figure(figsize=(15, 5))
            marker = itertools.cycle(('o', 'v', '^', 's', 'D'))
            plt.plot(x, y, marker=next(marker), markersize=13)
            if timestamp:
                plt.xlim([x[0]-timedelta(hours=1), x[-1]+timedelta(hours=1)])
            if baseline:
                m0 = plt.axhline(y_mean, linestyle='--', label='mean', c='black')
                mp3 = plt.axhline(y_mean+q3, linestyle='--', label='+3%', c='red')
                mn3 = plt.axhline(y_mean-q3, linestyle='--', label='-3%', c='blue')
                if max(y)>y_mean+q5 or min(y)<y_mean-q5:
                    plt.ylim(min(y)-q3, max(y)+q3)
                    # plt.text(1, max(y)+abs(0.15*y_mean), "rel_STD={rel_std}".format(rel_std=round(rel_std,3)))
                else:
                    plt.ylim(y_mean-q5, y_mean+q5)
                    #plt.text(1, y_mean+abs(0.03*y_mean), "rel_STD={rel_std}".format(rel_std=round(rel_std,3)))
                plt.legend([m0, mp3, mn3], ['mean','+3%','-3%'],prop={'size': 10})
            plt.title('(Em, Ex) = {cod}'.format(cod=cod))
            plt.show()
        return rel_std


    def pixel_linreg(self, Em, Ex, x, plot=True, output=True):
        # Find the closest point to the given coordinate (Em, Ex)
        Em_ref = dichotomy_search(self.Em_range, Em)
        Ex_ref = dichotomy_search(self.Ex_range, Ex)
        M = self.intensities
        x_reshaped = x.reshape(M.shape[0], 1)
        cod = (self.Em_range[Em_ref], self.Ex_range[Ex_ref])
        print('The closest data point (Em, Ex) is: {cod}'.format(cod=cod))
        y = (M[:, self.Ex_range.shape[0] - Ex_ref - 1, Em_ref])
        reg = LinearRegression().fit(x_reshaped, y)
        w = reg.coef_
        b = reg.intercept_
        r2 = reg.score(x_reshaped, y)
        pearson_coef, p_value_p = stats.pearsonr(x, y)
        spearman_coef, p_value_s = stats.spearmanr(x, y)
        print('Linear regression model: y={w}x+{b}'.format(w=w[0], b=b))
        print('Linear regression R2:', '{r2}'.format(r2=r2))
        print('Pearson coefficient: {coef_p} (p-value = {p_p})'.format(coef_p=pearson_coef, p_p=p_value_p))
        print('Spearman coefficient: {coef_s} (p-value = {p_s})'.format(coef_s=spearman_coef, p_s=p_value_s))
        if output:
            table = pd.DataFrame(np.concatenate([[x], [y]]).T)
            intensity_label = 'Intensity (Em, Ex)={cod}'.format(cod=cod)
            table.columns = ['Reference value', intensity_label]
            display(table)
        if plot:
            plt.scatter(x, y)
            p = np.array([x.min(), x.max()])
            q = w[0] * p + b
            plt.plot(p, q)
            plt.title('(Em, Ex) = {cod}'.format(cod=cod))
            # plt.text(p.min() + 0.1 * (p.max() - p.min()), q.max() - 0.1 * (p.max() - p.min()),
            #          "$R^2$={r2}".format(r2=round(r2, 5)))
            plt.show()
#        return w[0], b, r2, pearson_coef, p_value_p, spearman_coef, p_value_s


    def EEM_linreg(self, x, plot=True):
        M = self.intensities
        x = x.reshape(M.shape[0], 1)
        W = np.empty([M.shape[1], M.shape[2]])
        B = np.empty([M.shape[1], M.shape[2]])
        R2 = np.empty([M.shape[1], M.shape[2]])
        E = np.empty(M.shape)
        for i in range(M.shape[1]):
            for j in range(M.shape[2]):
                try:
                    y = (M[:, i, j])
                    reg = LinearRegression().fit(x, y)
                    W[i, j] = reg.coef_
                    B[i, j] = reg.intercept_
                    R2[i, j] = reg.score(x, y)
                    E[:, i, j] = reg.predict(x) - y
                except:
                    pass
        if plot:
            plt.figure(figsize=(6, 6))
            extent = [self.Em_range.min(), self.Em_range.max(), self.Ex_range.min(), self.Ex_range.max()]
            plt.imshow(1 - R2, cmap='jet', interpolation='none', extent=extent, aspect='1.2',
                       norm=LogNorm(vmin=0.0001, vmax=0.1))
            t = [0.0001, 0.001, 0.01, 0.1]
            plt.colorbar(ticks=t, fraction=0.03, pad=0.04)
            plt.title('1-$R^2$')
            plt.xlabel('Emission wavelength [nm]')
            plt.ylabel('Excitation wavelength [nm]')
        return W, B, R2, E


    def EEM_pearson_coef(self, x, plot=True, crange=(0, 1)):
        M = self.intensities
        C_p = np.empty([M.shape[1], M.shape[2]])
        P_p = np.empty([M.shape[1], M.shape[2]])
        for i in range(M.shape[1]):
            for j in range(M.shape[2]):
                try:
                    y = (M[:, i, j])
                    pc, p_value_p = stats.pearsonr(x, y)
                    C_p[i, j] = pc
                    P_p[i, j] = p_value_p
                except:
                    pass
        if plot:
            plt.figure(figsize=(8, 8))
            extent = [self.Em_range.min(), self.Em_range.max(), self.Ex_range.min(), self.Ex_range.max()]
            plt.imshow(abs(C_p), cmap='jet', interpolation='none', extent=extent, aspect='1.2',
                       norm=Normalize(crange[0], crange[1]))
            plt.colorbar(fraction=0.03, pad=0.04)
            plt.title('abs(Pearson correlation coefficient)')
            plt.xlabel('Emission wavelength [nm]')
            plt.ylabel('Excitation wavelength [nm]')

            plt.figure(figsize=(8, 8))
            extent = [self.Em_range.min(), self.Em_range.max(), self.Ex_range.min(), self.Ex_range.max()]
            plt.imshow(P_p, cmap='jet', interpolation='none', extent=extent, aspect='1.2',
                       norm=Normalize(vmin=0, vmax=0.1))
            plt.colorbar(fraction=0.03, pad=0.04)
            plt.title('p-value (Pearson)')
            plt.xlabel('Emission wavelength [nm]')
            plt.ylabel('Excitation wavelength [nm]')


    def EEM_spearman_coef(self, x, plot=True, crange=(0, 1)):
        M = self.intensities
        C_s = np.empty([M.shape[1], M.shape[2]])
        P_s = np.empty([M.shape[1], M.shape[2]])
        for i in range(M.shape[1]):
            for j in range(M.shape[2]):
                try:
                    y = (M[:, i, j])
                    sc, p_value_s = stats.spearmanr(x, y)
                    C_s[i, j] = sc
                    P_s[i, j] = p_value_s
                except:
                    pass
        if plot:
            plt.figure(figsize=(8, 8))
            extent = [self.Em_range.min(), self.Em_range.max(), self.Ex_range.min(), self.Ex_range.max()]
            plt.imshow(abs(C_s), cmap='jet', interpolation='none', extent=extent, aspect='1.2',
                       norm=Normalize(crange[0], crange[1]))
            plt.colorbar(fraction=0.03, pad=0.04)
            plt.title('abs(Spearman correlation coefficient)')
            plt.xlabel('Emission wavelength [nm]')
            plt.ylabel('Excitation wavelength [nm]')

            plt.figure(figsize=(8, 8))
            extent = [self.Em_range.min(), self.Em_range.max(), self.Ex_range.min(), self.Ex_range.max()]
            plt.imshow(P_s, cmap='jet', interpolation='none', extent=extent, aspect='1.2',
                       norm=Normalize(vmin=0, vmax=0.1))
            plt.colorbar(fraction=0.03, pad=0.04)
            plt.title('p-value (Spearman)')
            plt.xlabel('Emission wavelength [nm]')
            plt.ylabel('Excitation wavelength [nm]')



    def plot_EEM_linreg_error(self, error, x, x_index):
        extent = [self.Em_range.min(), self.Em_range.max(), self.Ex_range.min(), self.Ex_range.max()]
        plt.imshow(error[x_index, :, :], cmap='jet', interpolation='none', extent=extent, aspect='1.2', vmin=-100,
                   vmax=100)
        plt.title('x={x}'.format(x=x[x_index]))
        plt.xlabel('Emission wavelength [nm]')
        plt.ylabel('Excitation wavelength [nm]')


def EEM_statistics(EEMstack_class, term, crange, reference_label=None, reference=None, title=False):
    if term=='Mean':
        plot3DEEM(EEMstack_class.mean(), EEMstack_class.Em_range, EEMstack_class.Ex_range,
                  cmin=crange[0], cmax=crange[1])
    if term=='Standard deviation':
        plot3DEEM(EEMstack_class.std(), EEMstack_class.Em_range, EEMstack_class.Ex_range,
                  cmin=0, cmax=0.1*crange[1])
    if term=='Relative standard deviation':
        plot3DEEM(EEMstack_class.rel_std(), EEMstack_class.Em_range, EEMstack_class.Ex_range,
                  cmin=0, cmax=0.1)
    if term=='Correlation: Linearity':
        EEMstack_class.EEM_linreg(x=reference)
    if term=='Correlation: Pearson coef.':
        EEMstack_class.EEM_pearson_coef(x=reference)
    if term=='Correlation: Spearman coef.':
        EEMstack_class.EEM_spearman_coef(x=reference)


def EEM_diag_diff(intensity, threshold=0):
    fil = np.array([[-1, 0, 0],
                    [0, 0, 0],
                    [0, 0, 1]])
    intensity_diag_diff = cv2.filter2D(intensity, -1, fil)
    intensity_diag_diff[intensity_diag_diff<threshold] = 0
    return intensity_diag_diff


def EEM_to_uint8(intensity):
    intensity[intensity<0]=0
    intensity_scaled = np.interp(intensity, (0, intensity.max()), (0, 255))
    intensity_scaled = intensity_scaled.astype(np.uint8)
    return intensity_scaled


def rayleigh_masking(intensity, Em_range, Ex_range, tolerance=15, interpolation='zero'):
    intensity_masked = np.array(intensity)
    rayleigh_mask = np.zeros(intensity.shape)
    tol_pixel = tolerance # [nm]
    if Ex_range[0] < Em_range[0]:
        exidx_1st = dichotomy_search(Ex_range, Em_range[0])
        emidx_2nd = dichotomy_search(Em_range, Ex_range[0]*2)
        tol_emidx = int(np.round(tol_pixel/(Em_range[1]-Em_range[0])))
        tol_exidx = int(np.round(tol_pixel/(Ex_range[1]-Ex_range[0])))
        exidx_old = Ex_range.shape[0]-exidx_1st-1
        for emidx in range(0, len(Em_range)):
            intensity_masked[:exidx_old, emidx] = 0
            if Em_range[emidx] <= Ex_range[-1]:
                exidx_new = Ex_range.shape[0]-dichotomy_search(Ex_range, Em_range[emidx])-1
                if emidx < tol_emidx:
                    rayleigh_mask[exidx_new:exidx_old+1, 0:emidx+tol_emidx+1] = 1
                    rayleigh_mask[Ex_range.shape[0]-exidx_1st-1:exidx_new+tol_exidx, 0:emidx+1] = 1
                else:
                    rayleigh_mask[exidx_new:exidx_old+1, emidx:emidx+tol_emidx+1] = 1
                exidx_old = Ex_range.shape[0]-dichotomy_search(Ex_range, Em_range[emidx])-1
            if emidx > emidx_2nd:
                exidx2_new = Ex_range.shape[0]-dichotomy_search(Ex_range, Em_range[emidx]*0.5)-1
                try:
                    if len(Em_range)-emidx < tol_emidx:
                        rayleigh_mask[exidx2_new:exidx2_old+1, emidx-tol_emidx:] = 1
                        rayleigh_mask[exidx2_new-tol_exidx:exidx2_new, emidx:] = 1
                    elif len(Em_range)-emidx >= tol_emidx:
                        rayleigh_mask[exidx2_new:exidx2_old+1, emidx-tol_emidx:emidx+tol_emidx+1] = 1
                except NameError:
                    rayleigh_mask[exidx2_new:exidx2_new+1, emidx-tol_emidx:] = 1
                exidx2_old = Ex_range.shape[0]-dichotomy_search(Ex_range, Em_range[emidx]*0.5)-1

    if interpolation=='zero':
        intensity_masked[np.where(rayleigh_mask==1)]=0
        return intensity_masked, rayleigh_mask

    if interpolation=='nan':
        intensity_masked[np.where(rayleigh_mask==1)]=np.nan
        return intensity_masked, rayleigh_mask

    if interpolation=='linear':
        for i in range(0, intensity.shape[0]):
            spec = np.concatenate([[0],intensity_masked[i, :],[0]])
            #Em_range_extend = np.concatenate([[0],Em_range,[0]])
            ref = np.concatenate([[0],rayleigh_mask[i,:],[0]])
            y = spec[np.where(ref==0)]
            x_predict = np.arange(0,len(Em_range)+2,1)
            x = x_predict[np.where(ref==0)]
            f1 = interpolate.interp1d(x, y, kind='linear')
            y_predict = f1(x_predict)
            intensity_masked[i,:] = y_predict[1:-1]
        return intensity_masked, rayleigh_mask

    if interpolation=='linear2':
        for j in range(0, intensity.shape[1]):
            spec = np.concatenate([[0],intensity_masked[:, j],[0]])
            #Em_range_extend = np.concatenate([[0],Em_range,[0]])
            ref = np.concatenate([[0],rayleigh_mask[:,j],[0]])
            y = spec[np.where(ref==0)]
            x_predict = np.arange(0,len(Ex_range)+2,1)
            x = x_predict[np.where(ref==0)]
            f1 = interpolate.interp1d(x, y, kind='linear')
            y_predict = f1(x_predict)
            intensity_masked[:,j] = y_predict[1:-1]
        return intensity_masked, rayleigh_mask

def uv_noise_masking():
    pass


def contour_detection(intensity, Em_range=False, Ex_range=False, binary_threshold=50, maxval=255,
                      bluring=False, otsu=False, plot=False):
    if bluring:
        intensity = gaussian_filter(intensity, sigma=1, truncate=2)
    intensity_gray = EEM_to_uint8(intensity)
    if otsu:
        ret, intensity_binary = cv2.threshold(intensity_gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    if otsu==False:
        ret, intensity_binary = cv2.threshold(intensity_gray,binary_threshold,maxval,cv2.THRESH_BINARY)
    if plot:
        extent = [Em_range.min(), Em_range.max(), Ex_range.min(), Ex_range.max()]
        plot3DEEM(intensity, Em_range, Ex_range, autoscale=False, cmax=600, cmin=0, cmap='jet')
        plt.imshow(intensity_binary, extent=extent, alpha=0.3, cmap="binary")
    return intensity_binary


def plot3DEEM_interact(filedir, filename, autoscale=False, crange=[0, 3000], scattering_correction = True,
                       plot_abs=False, ABSxmax=0.1, title=False, Em_range_display=[250,820],
                       Ex_range_display=[200,500], contour_mask=False, gaussian_smoothing=True, sigma=1,
                       truncate=3, otsu=True, binary_threshold=50, tolerance=15, scattering_interpolation='linear2',
                       inner_filter_effect=True, ts_format='%Y-%m-%d-%H-%M-%S', ts_start_position=0, ts_end_position=19):
    filepath = filedir + '/' + filename
    intensity, Em_range, Ex_range = readEEM(filepath)
    if inner_filter_effect:
        absorbance, Ex_range2 = readABS(filepath[0:-7] + 'ABS.dat')
        intensity = EEM_inner_filter_effect(intensity, Em_range=Em_range, Ex_range=Ex_range, absorbance=absorbance,
                                            Ex_range2=Ex_range2)
    if gaussian_smoothing:
        intensity = gaussian_filter(intensity, sigma=sigma, truncate=truncate)
    if scattering_correction:
        intensity, rayleigh_mask = rayleigh_masking(intensity, Em_range, Ex_range, tolerance=tolerance,
                                                    interpolation=scattering_interpolation)
    intensity, Em_range, Ex_range = EEM_cutting(intensity, Em_range, Ex_range, Em_range_display[0],
                                                Em_range_display[1], Ex_range_display[0],
                                                Ex_range_display[1])
    plot3DEEM(intensity, Em_range, Ex_range, autoscale, crange[1], crange[0])
    if contour_mask:
        intensity_binary = contour_detection(intensity, otsu=otsu, binary_threshold=binary_threshold)
        binary_mask = EEM_mask(intensity=intensity, Em_range=Em_range, Ex_range=Ex_range, ref_matrix=intensity_binary, threshold=1,
                 residual='small', cmin=crange[0], cmax=crange[1])
        extent = [Em_range.min(), Em_range.max(), Ex_range.min(), Ex_range.max()]
        plt.imshow(binary_mask, extent=extent, alpha=0.5, cmap="binary", aspect=1.2)
    if title:
        tstitle = get_TS_from_filename(filename, ts_format=ts_format, ts_start_position=ts_start_position,
                                       ts_end_position=ts_end_position)
        plt.title(tstitle)
    if plot_abs:
        absorbance, Ex_range2 = readABS(filepath[0:-7] + 'ABS.dat')
        plotABS(absorbance, Ex_range2, ABSxmax)


def load_EEMstack_interact(filedir, scattering_correction=False, Em_range_display=[250,820],
                           Ex_range_display=[200,500], gaussian_smoothing=True, inner_filter_effect=True,
                           sigma=1, truncate=3, otsu=True, binary_threshold=50, tolerance=15,
                           scattering_interpolation='linear2', contour_mask=True, keyword_pem='PEM.dat'):
    EEMstack, Em_range, Ex_range_pem, datlist_pem = stackDat(filedir, kw=keyword_pem)
    if inner_filter_effect:
        datlist_abs = [dat[0:-7]+'ABS.dat' for dat in datlist_pem]
        ABSstack, Ex_range_abs, datlist_abs = stackABS(filedir, datlist=datlist_abs)
        EEMstack = EEMs_inner_filter_effect(EEMstack, ABSstack, Em_range, Ex_range_pem, Ex_range_abs)
    if gaussian_smoothing:
        EEMstack = EEMs_gaussianfilter(EEMstack, sigma=sigma, truncate=truncate)
    if scattering_correction:
        EEMstack = EEMs_scattering_correction(EEMstack, Em_range, Ex_range_pem,
                                              scattering_interpolation=scattering_interpolation, tolerance=tolerance)
    EEMstack, Em_range_cut, Ex_range_cut = EEMs_cutting(EEMstack, Em_range, Ex_range_pem, Em_min=Em_range_display[0],
                                                        Em_max=Em_range_display[1],
                                                        Ex_min=Ex_range_display[0], Ex_max=Ex_range_display[1])
    if contour_mask:
        EEMstack = EEMs_contour_masking(EEMstack, Em_range_cut, Ex_range_cut, otsu=otsu,
                                        binary_threshold=binary_threshold)
    print('Number of samples in the stack:', EEMstack.shape[0])
    return EEMstack, Em_range_cut, Ex_range_cut, datlist_pem


def decomposition_interact(EEMstack, Em_range, Ex_range, rank, timestamp, decomposition_method='parafac'):
    plt.close()
    try:
        if decomposition_method=='parafac':
            factors = parafac(EEMstack, rank=rank)
        if decomposition_method=='non_negative_parafac':
            factors = non_negative_parafac(EEMstack, rank=rank)
    except ArpackError:
        print("Please check if there's blank space in the fluorescence footprint in 'section 2. Fluorescence preview "
              "and parameter selection'. If yes, please adjust the excitation wavelength range to avoid excessive "
              "inner filter effect")
    I = factors[1][0]
    J = factors[1][1]
    K = factors[1][2]
    column_labels = []
    for r in range(rank):
        component = np.array([J[:, r]]).T.dot(np.array([K[:, r]]))
        component_label = 'component {rank}'.format(rank=r+1)
        column_labels.append(component_label)
        if I[:,r].sum() < 0:
            component = -component
            I[:,r] = -I[:,r]
        plot3DEEM(component, Em_range, Ex_range, autoscale=True, title=component_label)
        # component[np.where(abs(component)<abs(component).max()*quantile)] = 0
    parafac_table = pd.DataFrame(I)
    parafac_table.columns = column_labels
    parafac_table.index = timestamp[:]
    display(parafac_table)
    I_standardized = I/np.mean(I, axis=0)
    plt.figure(figsize=(15, 5))
    legend = []
    marker = itertools.cycle(('o', 'v', '^', 's', 'D'))
    for r in range(rank):
        plt.plot(timestamp, I_standardized[:, r], marker=next(marker), markersize=13)
        legend.append('component {rank}'.format(rank=r+1))
        plt.xlabel('Time')
        plt.ylabel('Standardized loading')
    plt.legend(legend)


def decomposition_reconstruction_interact(I, J, K, intensity, Em_range, Ex_range, datlist, data_to_view, crange=[0,1000]):
    idx = datlist.index(data_to_view)
    rank = I.shape[1]
    for r in range(rank):
        component = np.array([J[:, r]]).T.dot(np.array([K[:, r]]))
        if r == 0:
            sample_r = I[idx, r] * component
        else:
            sample_r += I[idx, r] * component
        # reconstruction_error = np.linalg.norm(sample_r - EEMstack[idx])
        plot3DEEM(sample_r, Em_range, Ex_range, autoscale=False, cmin=crange[0], cmax=crange[1], figure_size=(8, 8),
                  title='Accumulate to component {rank}'.format(rank=r+1))
    plot3DEEM(intensity, Em_range, Ex_range, autoscale=False, cmin=crange[0], cmax=crange[1],
              title='Original footprint')


