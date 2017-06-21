import sys
sys.path.append("./methods/")
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
if return_freq == False, then a 1d array, where data[i] is the amplitude of the
EMG signal at that point
'''
def prepare_data(filePath, num_channels = 4, filter_order = 2,
                 do_high_pass = True, do_low_pass = True,
                 high_pass_critical_freq = .1, low_pass_critical_freq = .1,
                 band_stop_min_freq = 50, band_stop_max_freq = 60,
                 reject_z_cutoff = 2.5, reject_divide_factor = 4,
                 window = 36, return_freq = True, normalize = True):
    data = parse(filePath, num_channels)
    if do_high_pass:
        data = highPass(data, filter_order, high_pass_critical_freq)
    if do_low_pass:
        data = lowPass(data, filter_order, low_pass_critical_freq)
    data = bandStop(data, band_stop_min_freq, band_stop_max_freq)
    data = peakReject(data, reject_z_cutoff, reject_divide_factor, window)
    if return_freq:
        data = FFT(data, window)
    return np.array(data)
