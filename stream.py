import sys
sys.path.append("./pre-processing/methods/")
sys.path.append("./learning_model/")
from timeDomain import *
from freqDomain import *
from parse import *
import numpy as np
import matplotlib.pyplot as plt
from analytical import *
import math
import time

'''
A master preprocessing method that includes high pass filtering, low pass
filtering, band stop filtering, peak rejection, normalization, a savgol
filter, and FFT transformations. It also reads/writes from the parse file.

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
def stream_detect(filePath, num_channels = 1, filter_order = 2,
                 do_high_pass = True, do_low_pass = True,
                 high_pass_critical_freq = .1, low_pass_critical_freq = .1,
                 band_stop_min_freq = 50, band_stop_max_freq = 60,
                 reject_z_cutoff = 2.5, reject_divide_factor = 4,
                 window = 225):
    an = analytical()
    mean = 100
    eeg_data = []
    while(True):
        parsed = parse(filePath, num_channels)
        #keep only the last 2000 lines
        if len(eeg_data) == 0:
            print "Calibrating. Please hold arm still."
            print str(np.array(parsed).shape[1]/2060.0) + "%"
            if np.array(parsed).shape[1] > 2000:
                if do_high_pass:
                    data = highPass(parsed, filter_order, high_pass_critical_freq)

                #low pass filter
                if do_low_pass:
                    data = lowPass(data, filter_order, low_pass_critical_freq)

                #band stop
                data = bandStop(data, band_stop_min_freq, band_stop_max_freq)

                #peak rejection
                data = peakReject(data, reject_z_cutoff, reject_divide_factor, window)

                #apply the Fourier transform
                data = FFT(data, window)
                maxes = [np.max(data[0, i]) for i in range(len(data[0]))]
                new_maxes = []
                for max in maxes:
                    if not math.isnan(max):
                        new_maxes.append(max)
                mean = np.mean(new_maxes)
                eeg_data = parsed
        else:
            length = np.array(parsed).shape[1]
            eeg_data = np.concatenate((np.array(eeg_data)[:, length:], parsed), axis = 1)
            data = eeg_data
            if do_high_pass:
                data = highPass(data, filter_order, high_pass_critical_freq)

            #low pass filter
            if do_low_pass:
                data = lowPass(data, filter_order, low_pass_critical_freq)

            #band stop
            data = bandStop(data, band_stop_min_freq, band_stop_max_freq)

            #peak rejection
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
                    two_dimension_data[j][k][1] = ((two_dimension_data[j][k][0] - two_dimension_data[j][k - window / 4][0])/(window / 4))

                    #the third feature is the timepoint in comparison to its local mean
                    local = two_dimension_data[j][k - window / 2: k, 0]
                    two_dimension_data[j][k][2] = (two_dimension_data[j][k][0] - np.mean(local))

            #get rid of nan values
            two_dimension_data = get_rid_nan_values(two_dimension_data)

            #smooth
            two_dimension_data = savgol(two_dimension_data)

            #normalize all the channels
            two_dimension_data = normalize(two_dimension_data, mean)

            #get rid of nan values
            two_dimension_data = get_rid_nan_values(two_dimension_data)

            for chan in range(num_channels):
                #get predictions for current timepoint
                data_2000 = two_dimension_data[chan]
                predictions = an.predict(data_2000)
                mean_predict = np.mean(predictions[-25:])
                print 'Prediction on Channel ' + str(chan) + ": " + \
                str(mean_predict)

            #keep only the last 2000 lines
            f = open(filePath, 'w+')
            f.truncate()
            f.close()
            time.sleep(.2)


stream_detect("/Users/williamlevine/Documents/BCI/SavedData/OpenBCI-RAW-2017-07-03_21-33-43.txt")
