import numpy as np

import numpy as np

def parse(filename, numChannels):
    data = open(filename).read().split("\n")
    data = np.array([line.split(", ") for line in data if "%" not in line])
    #what we are returning
    ret = []
    for i in range(1, numChannels + 1):
        channelData = []
        for j in range(1, len(data)):
            if len(data[j]) > numChannels:
                crossChannel = [float(data[j][k]) for k in range(1, numChannels + 1)]
                if np.sum(crossChannel) != 0:
                    channelData.append(float(data[j][i]))
        ret.append(channelData)
    return np.array(ret)
