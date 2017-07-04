import numpy as np
import matplotlib.pyplot as plt
import pylab
from utils import accuracy

''' an analytical model for EMG signal classification'''
class analytical(object):

    '''
    Analytical classifier

    Parameters/Attributes
    ____________________
    local_val_thresh : int
        the value a specific timepoint needs to exceed in comparison to its local
        values to be considered a positive or negative movement
    '''
    def __init__(self, local_val_thresh = 0):
        self.local_val_thresh = local_val_thresh

    '''
    predict

    predicts the "true" movement of the an EMG-attached limb

    Parameters
    ____________
    data : 2d array
        the input data

    Outputs
    ____________
    predictions : 2d array
        the predictions
    '''

    def predict(self, data):
        predictions = []
        for i in range(len(data)):
            if data[i, 0] > 0 and np.abs(data[i, 2]) > self.local_val_thresh:
                predictions.append(data[i, 0])
            else:
                predictions.append(0)
        return np.array(predictions)

    '''
    Evaluate the Analytical model

    Inputs
    data: a 2d array, where x is time and y is value
    labels: a 2d array, where x is time and y is the label
    display_feat : Int (0 -> num_features)
        the feature you'd like to display on the graph w/ predictions & labels
    '''
    def evaluate(self,
                 data,
                 labels,
                 display_feat = 0):
        #predict the remaining data using the generated logistic regression model
        predictions = self.predict(data)
        pylab.plot(np.array(data[:,display_feat]) \
                            / np.max(np.array(data[:, display_feat])), \
                            label = "Raw Data Feature " + str(display_feat))
        pylab.plot(predictions / np.max(predictions), label="Predictions")
        pylab.plot(np.array(labels) , label="Labels")
        pylab.xlabel("Time")
        pylab.ylabel("Unit of Feature " + str(display_feat))
        pylab.legend(loc = 'upper right')
        pylab.ylim(-2, 2)
        pylab.show()
        classes = np.zeros_like(predictions)
        for i in range(len(predictions)):
            if predictions[i] > predictions[i - 5]:
                classes[i] = 1
            elif predictions[i] < predictions[i - 5]:
                classes[i] = -1
            else:
                classes[i] = 0
        print 'Performance on test data: '
        accuracy(classes, labels)

#unit test
if __name__ == "__main__":
    #loading the data
    import pickle
    an = analytical()
    #NOTE: you'll probably have to change the file path to get this unit test to run
    train_data = pickle.load(open("C:/Users/levinwv1/Downloads/2-Seconds-Will-Trial-1.MultFeat"))
    train_labels = train_data[1]
    train_x = train_data[0]
    an.evaluate(train_x, train_labels)

    #validation on a completely different data set
    test_data = pickle.load(open("C:/Users/levinwv1/Downloads/4-Seconds-Will-Trial-1.MultFeat"))
    test_labels = test_data[1]
    test_x = np.array(test_data[0])

    an.evaluate(test_x, test_labels)
