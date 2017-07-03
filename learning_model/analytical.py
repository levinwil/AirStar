import numpy as np

'''
predict

predicts the "true" movement of the an EMG-attached limb

Parameters
____________
data : 3d array
    the input data
local_val_thresh : int
    the value a specific timepoint needs to exceed in comparison to its local
    values to be considered a positive or negative movement

Outputs
____________
predictions : 2d array
    the predictions
'''

def predict(data, local_val_thresh = 0):
    predictions = []
    for i in range(len(data)):
        if data[i, 0] > 0 and np.abs(data[i, 2]) > local_val_thresh:
            if data[i, 2] > 0:
                predictions.append(data[i, 0])
            else:
                predictions.append(data[i, 0])
        else:
            predictions.append(0)
    return np.array(predictions)
