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
def prepare_data(filePath, num_channels = 1, filter_order = 2,
                 do_high_pass = True, do_low_pass = True,
                 high_pass_critical_freq = .1, low_pass_critical_freq = .1,
                 band_stop_min_freq = 50, band_stop_max_freq = 60,
                 reject_z_cutoff = 2.5, reject_divide_factor = 4,
                 window = 100, return_freq = False,
                 return_freq_std_window = False, return_multi_feat = True):
    data = parse(filePath, num_channels)
    #high pass filter
    if do_high_pass:
        data = highPass(data, filter_order, high_pass_critical_freq)

    #low pass filter
    if do_low_pass:
        data = lowPass(data, filter_order, low_pass_critical_freq)

    #band stop
    data = bandStop(data, band_stop_min_freq, band_stop_max_freq)

    #peak rejection
    data = peakReject(data, reject_z_cutoff, reject_divide_factor, window)

    if return_freq or return_multi_feat or return_freq_std_window:
        #apply the Fourier transform
        data = FFT(data, window)

        #if return_freq, we're done
        if return_freq:
            return data
        if (return_freq_std_window or return_multi_feat):

            #multiple features
            if return_multi_feat:

                #this is the data we'll be returning
                two_dimension_data = np.zeros((len(data), len(data[0]), 3))
                for j in range(len(data)):
                    for i in range(len(data[j])):

                        #the first feature is simply the max of the FFT
                        two_dimension_data[j][i][0] = np.max(data[j][i])
                    for k in range(2*window, len(data[j])):

                        #the second feature is the tangent slope
                        two_dimension_data[j][k][1] = ((two_dimension_data[j][k][0] - two_dimension_data[j][k - window][0])/(window))

                        #the third feature is the timepoint in comparison to its local mean
                        local = two_dimension_data[j][k - window : k, 0]
                        two_dimension_data[j][k][2] = (two_dimension_data[j][k][0] - np.mean(local))

                #get rid of nan values
                two_dimension_data = get_rid_nan_values(two_dimension_data)

                #smooth
                two_dimension_data = savgol(two_dimension_data)

                #normalize all the channels
                two_dimension_data = normalize(two_dimension_data)
                return two_dimension_data
            else:
                #the thing we'll be returning
                two_dimension_data = np.zeros((len(data), len(data[0]), window))
                for j in range(len(data)):
                    for k in range(len(data[j])):
                        if k < window:
                            two_dimension_data[j][k] = [0 for _ in range(window)]
                        else:
                            mx = [np.max(data[j][l]) for l in range(k - window, k)]
                            #make it the max value of each index in a window behind
                            #it
                            two_dimension_data[j][k] = mx
                data = two_dimension_data

    #check for Nan values
    for chan in range(len(data)):
        for tp in range(len(data[chan])):
            for feat in range(len(data[chan][tp])):
                if math.isnan(data[chan][tp][feat]) or math.isinf(data[chan][tp][feat]):
                    data[chan][tp][feat] = np.mean(data[chan][tp][feat - 10:feat])
    return np.array(data)
