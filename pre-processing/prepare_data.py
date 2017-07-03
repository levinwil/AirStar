import sys
sys.path.append("../pre-processing/methods/")
from timeDomain import *
from freqDomain import *
from parse import *
import numpy as np
import matplotlib.pyplot as plt

'''
A master preprocessing method that includes high pass filtering, low pass
filtering, band stop filtering, peak rejection, normalization, a savgol
filter, and FFT transformations

Parameters
____________
too many. If you are wondering what they are, please explore the timeDomain
file in ./methods

Returns
____________
a 2d array, where data[i][0] is the maximum of its frequency makeup window,
data[i][1] is its approximate slope, and data[i][2] is data[i][0] in relation
to the points around it
'''
def prepare_data(filePath, num_channels = 1, filter_order = 2,
                 do_high_pass = True, do_low_pass = True, do_peak_reject = True,
                 high_pass_critical_freq = .1, low_pass_critical_freq = .1,
                 band_stop_min_freq = 50, band_stop_max_freq = 60,
                 reject_z_cutoff = 3.5, reject_divide_factor = 2,
                 window = 225):
    data, _ = parse(filePath, num_channels)
    #high pass filter
    if do_high_pass:
        data = highPass(data, filter_order, high_pass_critical_freq)

    #low pass filter
    if do_low_pass:
        data = lowPass(data, filter_order, low_pass_critical_freq)

    #band stop
    data = bandStop(data, band_stop_min_freq, band_stop_max_freq)

    #peak rejection
    if do_peak_reject:
        data = peakReject(data, reject_z_cutoff, reject_divide_factor, window)

    #apply the Fourier transform
    data = FFT(data, window)

    #this is the data we'll be returning
    two_dimension_data = np.zeros((len(data), len(data[0]), 3))
    for j in range(len(data)):
        for i in range(len(data[j])):

            #the first feature is simply the max of the FFT
            two_dimension_data[j][i][0] = np.max(data[j][i])
        for k in range(2*window, len(data[j])):

            #the second feature is the tangent slope
            two_dimension_data[j][k][1] = ((two_dimension_data[j][k][0] - two_dimension_data[j][k - window / 2][0])/(window / 2))

            #the third feature is the timepoint in comparison to its local mean
            local = two_dimension_data[j][k - window / 2: k, 0]
            two_dimension_data[j][k][2] = (two_dimension_data[j][k][0] - np.mean(local))

    #get rid of nan values
    two_dimension_data = get_rid_nan_values(two_dimension_data)

    #smooth
    two_dimension_data = savgol(two_dimension_data)

    #normalize about x axis
    two_dimension_data = normalize(two_dimension_data, percentile = 10)

    return two_dimension_data
