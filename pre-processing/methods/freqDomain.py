from numpy.fft import fft
import numpy as np

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
