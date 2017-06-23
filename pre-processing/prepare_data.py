import sys
sys.path.append("../pre-processing/methods/")
from timeDomain import *
from freqDomain import *
from parse import *
import numpy as np
import matplotlib.pyplot as plt

'''
A master preprocessing method that includes high pass filtering, low pass
filtering, band stop filtering, peak rejection, and FFT transformations

Parameters
____________
too many. If you are wondering what they are, please explore the timeDomain
file in ./methods

Returns
____________
if return_freq == True, then a 2d array, where data[i] is the frequency
makeup of size 'window'

if return_freq_std_window == True, then a 2d array, where we turn data[i] into
the standard deviation of the frequency makeup of the window behind it, and
then take a window of of those standard deviations

if return_multi_feat == true, then a 2d array, where data[i][0] is the
standard deviation of its frequency makeup window, data[i][1] is its
approximate slope, and data[i][2] is data[i][0] in relation to the points
around it
'''
def prepare_data(filePath, num_channels = 4, filter_order = 2,
                 do_high_pass = True, do_low_pass = True,
                 high_pass_critical_freq = .1, low_pass_critical_freq = .1,
                 band_stop_min_freq = 50, band_stop_max_freq = 60,
                 reject_z_cutoff = 2.5, reject_divide_factor = 4,
                 window = 36, return_freq = False,
                 return_freq_std_window = False, return_multi_feat= False):
    data = parse(filePath, num_channels)
    if do_high_pass:
        data = highPass(data, filter_order, high_pass_critical_freq)
    if do_low_pass:
        data = lowPass(data, filter_order, low_pass_critical_freq)
    data = bandStop(data, band_stop_min_freq, band_stop_max_freq)
    data = peakReject(data, reject_z_cutoff, reject_divide_factor, window)
    if return_freq or return_multi_feat or return_freq_std_window:
        data = FFT(data, window)
        if return_freq:
            return data
        if (return_freq_std_window or return_multi_feat):
            if return_multi_feat:
                two_dimension_data = np.zeros((len(data), len(data[0]), 3))
                for j in range(len(data)):
                    for i in range(len(data[j])):
                        two_dimension_data[j][i][0] = np.max(data[j][i])
                    for k in range(2*window, len(data[j])):
                        two_dimension_data[j][k][1] = ((two_dimension_data[j][k][0] - two_dimension_data[j][k - 2*window][0])/(2*window))
                        local = two_dimension_data[j][k - window : k, 0]
                        two_dimension_data[j][k][2] = (two_dimension_data[j][k][0] - np.mean(local))
                return two_dimension_data
            else:
                two_dimension_data = np.zeros((len(data), len(data[0]), window))
                for j in range(len(data)):
                    for k in range(len(data[j])):
                        if k < window:
                            two_dimension_data[j][k] = [0 for _ in range(window)]
                        else:
                            std = [np.std(data[j][l]) for l in range(k - window, k)]
                            two_dimension_data[j][k] = std
                data = two_dimension_data
    return np.array(data)
