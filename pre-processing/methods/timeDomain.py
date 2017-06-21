import numpy as np
from scipy import signal
from obspy.signal.filter import bandstop

'''
peakReject

runs peak rejection across every channel in the input data. When a point is
considered a peak (in the global sense), we try to bring it closer to the mean
of the 'window' points to its left

Parameters
____________

data : 2d array
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

centers the input data about the x axis so the average is 0. NOTE: we do NOT
divide by the standard deviation

Parameters
____________

data : 2d array
    the data you want to be normalized

Returns
____________
data : 2d array
    the normalized data
'''
def normalize(data):
    for i in range(len(data)):
        data[i] = (data[i] - np.mean(data[i]))
    return data

'''
highPass

applies a high pass filter to the input data

Parameters
____________

data : 2d array
    the data you want to be filtered
order : Int
    the order of the high pass filter
critical_freq : Double (0 to 1)
    the critical frequency. Is normalized between 0 and 1, where 1 is the
    Nyquist frequency (and thus will pass the original signal) and 0 will pass
    no signal

Returns
____________
data : 2d array
    the normalized data
'''
def highPass(data, order = 1, critical_freq = .1):
    ret = []
    def helper(d):
        B, A = signal.butter(order, critical_freq, btype = 'highpass', output='ba')
        return signal.filtfilt(B,A, d)
    for i in range(len(data)):
        ret.append(helper(data[i]))
    return ret


'''
lowPass

applies a low pass filter to the input data

Parameters
____________

data : 2d array
    the data you want to be filtered
order : Int
    the order of the high pass filter
critical_freq : Double (0 to 1)
    the critical frequency. Is normalized between 0 and 1, where 1 is the
    Nyquist frequency (and thus will pass the original signal) and 0 will pass
    no signal

Returns
____________
data : 2d array
    the normalized data
'''
def lowPass(data, order = 1, critical_freq = .1):
    ret = []
    def helper(d):
        B, A = signal.butter(order, critical_freq, btype = 'lowpass', output='ba')
        return signal.filtfilt(B,A, d)
    for i in range(len(data)):
        ret.append(helper(data[i]))
    return ret

'''
bandStop

applies a band stop filter between freqmin and freqmax Hz

Parameters
____________

data : 2d array
    the data you want to be filtered
freqmin : Int (in Hz)
    the min frequency in the band stop filter
freqmax : Int (in Hz)
    the max frequency in the band stop filter

Returns
____________
data : 2d array
    the filtered data
'''
def bandStop(data, freq_min = 50, freq_max = 60):
    ret = []
    def helper(d):
        return bandstop(d, freqmin = freq_min, freqmax = freq_max, df = 200)
    for i in range(len(data)):
        ret.append(helper(data[i]))
    return ret
