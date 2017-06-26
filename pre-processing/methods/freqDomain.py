from numpy.fft import fft
import numpy as np
from scipy import signal
from obspy.signal.filter import bandstop

"""
FFT

applies the fourier transform to every index of the input array, viewing 'window' number
of indices behind it as the input to the Fourier Transform

Parameters
---------
array-like data    --- Either an np.ndarray containing the data (MUST BE 2D, where first dimension is channel, second is time)
Integer window     --- the window size viewed behind each index as input to the Fourier Transform

Return
------
a 3D numpy array, where the first dimension is channel, the second is time, and the third is frequency

"""
def FFT(data, window = 100):
    ret = np.zeros((len(data), len(data[0]), window))
    for j in range(len(data)):
        for i in range(0, len(data[j])):
            if i < window:
                ret[j][i] = [0] * window
            else:
                ret[j][i] = fft(data[j][(i - window): i])
    return np.array(ret)

'''
highPass

applies a high pass filter to the input data

Parameters
____________

data : 3d array
    the data you want to be filtered
order : Int
    the order of the high pass filter
critical_freq : Double (0 to 1)
    the critical frequency. Is normalized between 0 and 1, where 1 is the
    Nyquist frequency (and thus will pass the original signal) and 0 will pass
    no signal

Returns
____________
data : 3d array
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

data : 3d array
    the data you want to be filtered
order : Int
    the order of the high pass filter
critical_freq : Double (0 to 1)
    the critical frequency. Is normalized between 0 and 1, where 1 is the
    Nyquist frequency (and thus will pass the original signal) and 0 will pass
    no signal

Returns
____________
data : 3d array
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

data : 3d array
    the data you want to be filtered
freqmin : Int (in Hz)
    the min frequency in the band stop filter
freqmax : Int (in Hz)
    the max frequency in the band stop filter

Returns
____________
data : 3d array
    the filtered data
'''
def bandStop(data, freq_min = 50, freq_max = 60):
    ret = []
    def helper(d):
        return bandstop(d, freqmin = freq_min, freqmax = freq_max, df = 200)
    for i in range(len(data)):
        ret.append(helper(data[i]))
    return ret
