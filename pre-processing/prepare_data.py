import sys
sys.path.append("/Users/williamlevine/Documents/BCI/motion-tracking/pre-processing/scripts")
from timeDomain import *
from freqDomain import *
import numpy as np
import matplotlib.pyplot as plt

def parse(filename, numChannels):
    data = open(filename).read().split("\n")
    data = np.array([line.split(", ") for line in data if "%" not in line])
    #what we are returning
    ret = []
    for i in range(1, numChannels + 1):
        channelData = []
        for j in range(1, len(data)):
            if len(data[j]) >= numChannels:
                crossChannel = [float(data[j][k]) for k in range(1, numChannels + 1)]
                if np.sum(crossChannel) != 0:
                    channelData.append(float(data[j][i]))
        ret.append(channelData)
    return np.array(ret)

def prepare_data(filePath, numChannels = 4, highPassOrder = 2, doHighPass = True, highPassFrequency = .1, rejectZCutoff = 2.5, rejectDivideFactor = 4, window = 100, returnFreq = False, normalize = False):
    data = parse(filePath, numChannels)
    if doHighPass:
        data = highPass(data, highPassOrder, highPassFrequency)
    data = peakReject(data, rejectZCutoff, rejectDivideFactor, window)
    if returnFreq:
        data = FFT(data, window)
    return np.array(data)

if __name__ == "__main__":
    data = prepare_data("/Users/williamlevine/Downloads/OpenBCI-RAW-Mixture-Trial-4.txt", returnFreq = True)[3]
    for i in range(len(data)):
        data[i] = np.max(data[i])
    plt.plot(data)
    plt.show()
