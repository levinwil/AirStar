import numpy as np
from scipy.signal import savgol_filter
import math

'''
peakReject

runs peak rejection across every channel in the input data. When a point is
considered a peak (in the global sense), we try to bring it closer to the mean
of the 'window' points to its left

Parameters
____________

data : 3d array
    the data you want to peak rejection
z_cutoff : Double
    the z-score required for a point to be considered a peak
divide_factor : Int or Double
    when a data point is considered a peak, we find the distance that point is
    away from the local mean (will be explained in the next parameter), then
    multiply that distance by divide_factor, then add that back to the local
    mean
window : int
    the number of points to consider when bringing the peak closer to the local
    mean

Returns
____________
data : 2d array
    the peak rejected data
'''
def peakReject(data, z_cutoff = 3.5, divide_factor = 3, window = 50):
    for j in range(len(data)):
        originalChannel = data[j]
        channel = data[j]
        mean = np.mean(channel)
        std = np.std(channel)
        for i in range(len(channel)):
            zscore = (channel[i] - mean)/(std)
            if np.abs(zscore) >= z_cutoff:
                lowerBound = window
                if lowerBound < 2 * window:
                    lowerBound = i - 2 * window
                localMean = np.mean([originalChannel[i - k] for k in range(lowerBound)])
                channel[i] = (channel[i] - localMean)/divide_factor + localMean
        data[j] = channel
    return data

'''
normalize

centers the 0th channel about the x-axis and filters out background noise

Parameters
____________

data : 3d array
    the data you want to be normalized

percentile : Int
    the percentile you'd like to subtract from the 0th channel

window: Int
    the window we consider when normalizing

background_value : Int
    the static background value

Returns
____________
data : 3d array
    the normalized data
'''
def normalize(data,  background_value, percentile = 20, window = 500):
    data = np.array(data)
    for chan in range(data.shape[0]):
        for tp in range(window, data.shape[1]):
            data[chan][tp, 0] = data[chan][tp, 0] - \
            np.percentile(data[chan][tp - window : tp, 0], percentile) -\
            background_value
    return data


'''
savgol

applies a savgol smoothing filter to every channel in the data

Parameters
____________

data : 3d array
    the data you want to be smoothened
window : Int
    the window size you consider when estimating polynomials
max_degree_poly : Int
    the maximum degree polynomial you will estimate

Returns
____________
data : 3d array
    the smooth data
'''
def savgol(data, window = 2001, max_degree_poly = 5):
    for chan in range(len(data)):
        for feat in range(len(data[chan][0])):
            data[chan][:, feat] = savgol_filter(data[chan][:, feat],
                                                window,
                                                max_degree_poly)
    return data

'''
get_rid_nan_values

gets rid of nan values in a 2D array and replaces them with 0's

Parameters
__________
two_dimension_data : array

Returns
__________
two_dimension_data : array
'''
def get_rid_nan_values(two_dimension_data):
    for chan in range(len(two_dimension_data)):
        for tp in range(len(two_dimension_data[chan])):
            for feat in range(len(two_dimension_data[chan][tp])):
                if math.isnan(two_dimension_data[chan][tp][feat]) or math.isinf(two_dimension_data[chan][tp][feat]):
                    two_dimension_data[chan][tp][feat] = 0.0
    return two_dimension_data
