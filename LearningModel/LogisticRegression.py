from sklearn.linear_model import LogisticRegression
from utils import train_test_split
import math
import numpy as np
from utils import precision_recall_f1
import matplotlib.pyplot as plt

''' A general Logistic Regression model for EMG signal classification'''
class LR(object):

    '''
    Logistic Regression Classifier

    Attributes
    ____________
    model : keras model
        The model after fitting
    '''
    def __init__(self, data = None, labels = None):
        if (isinstance(data, list) or isinstance(data, np.ndarray)) and \
           (isinstance(labels, list) or isinstance(labels, np.ndarray)):
            self.fit(data, labels)
        else:
            self.model = None

    '''
    A basic logistic regression model classifier

    Inputs
    data: a 2d array, where x is time and y is value
    labels: a 2d array, where x is time and y is the label
    n_estimators: the number of trees in the logistic regression model
    n_jobs: the number of cores you want to use in parallel

    Outputs
    lr: the fitted logistic regression model
    '''
    def fit(self, data, labels):
        #if there is only 1 feature
        if len(np.array(data).shape) == 1:
            data = np.reshape(data, (-1, 1))
        lr = LogisticRegression()
        lr.fit(data, labels)
        self.model = lr

    '''
    A basic logistic regression model classifier

    Inputs
    data: a 2d array, where x is time and y is value
    lr: a pre-fitted logistic regression model

    Outputs
    lr: the predictions
    '''
    def getPredictions(self,
                data,
                verification_window = 5,
                continuity_window = 100,
                min_size = 65):
        #if there is only 1 feature
        if len(np.array(data).shape) == 1:
            data = np.reshape(data, (-1, 1))
        predictions =  self.model.predict(data)
        for i in range(verification_window, len(predictions)):
            std = np.std([predictions[i - verification_window: i]])
            if std > 0.5:
                for j in range(verification_window):
                    predictions[i - j] = -1
            if predictions[i] == 1:
                previous_occurence = predictions[i]
                for j in range(continuity_window):
                    if predictions[i - j] == 1:
                        previous_occurence = i - j
                    elif predictions[i - j] == -1:
                        break
                        break
                if (i - previous_occurence) < continuity_window:
                    for j in range(i - previous_occurence):
                        predictions[i - j] = 1
            if predictions[i] == -1:
                previous_occurence = predictions[i]
                for j in range(continuity_window):
                    if predictions[i - j] == -1:
                        previous_occurence = i - j
                    elif predictions[i - j] == 1:
                        break
                        break
                if (i - previous_occurence) < continuity_window:
                    for j in range(i - previous_occurence):
                        predictions[i - j] = -1
        for i in range(len(predictions)):
            if predictions[i - 1] == 1 and predictions[i] == 0:
                j = i - 1
                while predictions[j] == 1 and j > 0:
                    j = j -1
                if (i - j) < min_size:
                    for k in range(i - j):
                        predictions[i - k] = predictions[j - 1]
            elif predictions[i - 1] == -1 and predictions[i] == 0:
                j = i - 1
                while predictions[j] == -1 and j > 1:
                    j = j - 1
                if (i - j) < min_size:
                    for k in range(i - j):
                        predictions[i - k] = predictions[j - 1]
            elif predictions[i - 1] == 0 and predictions[i] == 1:
                j = i - 1
                while predictions[j] == 0 and j > 0:
                    j = j -1
                if (i - j) <  min_size:
                    for k in range(i - j):
                        predictions[i - k] = predictions[j - 1]
            elif predictions[i - 1] == 0 and predictions[i] == -1:
                j = i - 1
                while predictions[j] == 0 and j > 1:
                    j = j - 1
                if (i - j) <  min_size:
                    for k in range(i - j):
                        predictions[i - k] = predictions[j - 1]
        return predictions

    '''
    Evaluate the Logistic Regression model

    Inputs
    data: a 2d array, where x is time and y is value
    labels: a 2d array, where x is time and y is the label
    label_value: the value you assigned to your labels
    '''
    def evaluate(self, data, labels, label_value = 1):
        #predict the remaining data using the generated logistic regression model
        predictions = self.getPredictions(data)
        precision_recall_f1(np.abs(predictions), np.abs(labels), label_value)

#unit test
if __name__ == "__main__":
    #loading the data
    import pickle
    #NOTE: you'll probably have to change the file path to get this unit test to run
    train_data = pickle.load(open("/Users/williamlevine/Downloads/2-Seconds-6-Seconds-mixture-concat.MultFeat"))
    train_labels = train_data[1]
    train_x = train_data[0]


    #running the LR and producing results
    lr = LR(train_x, train_labels)

    #validation on a completely different data set
    test_data = pickle.load(open("/Users/williamlevine/Downloads/5-seconds-trial-1.MultFeat"))
    test_labels = test_data[1]
    test_x = np.array(test_data[0])
    predictions = lr.getPredictions(test_x)
    lr.evaluate(test_x, predictions)
    plt.plot(test_x[:, 1] * 1000)
    plt.plot(predictions)
    plt.show()
