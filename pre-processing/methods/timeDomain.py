import numpy as np
from scipy.signal import savgol_filter

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
                lowerBound = i - window
                if lowerBound < 2 * window:
                    lowerBound = 0
                localMean = np.mean([originalChannel[i - k] for k in range(lowerBound)])
                channel[i] = (channel[i] - localMean)/divide_factor + localMean
        data[j] = channel
    return data

'''
normalize

divides each channel by the max of that channel, such that each channel
is roughyl on the same scale

Parameters
____________

data : 3d array
    the data you want to be normalized

Returns
____________
data : 3d array
    the normalized data
'''
def normalize(data):
    for i in range(len(data)):
        data[i] = [data[i][k] / np.max(data[i]) for k in range(len(data[i]))]
    return data


'''
savgol

applies a savgol smoothing filter to every channel in the data

Parameters
____________

data : 3d array
    the data you want to be smoothened

Returns
____________
data : 3d array
    the smooth data
'''
def savgol(data):
    for chan in range(len(data)):
        for feat in range(len(data[chan][0])):
            data[chan][:, feat] = savgol_filter(data[chan][:, feat], 2001, 5)
    return data
