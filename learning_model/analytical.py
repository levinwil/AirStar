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

def predict(data, locl_val_thres = 3):
    predictions = []
    for i in range(len(normal)):
        if normal[i, 0] > 0 and np.abs(normal[i, 2]) > local_val_thresh:
            if normal[i, 2] > 0:
                predictions.append(normal[i, 0])
            else:
                predictions.append(-1 * normal[i, 0])
        else:
            predictions.append(0)
