from sklearn.linear_model import LogisticRegression
from utils import train_test_split
import math
import numpy as np
from utils import precision_recall_f1

''' A general Logistic Regression model for EMG signal classification'''
class LR(object):

    '''
    Logistic Regression Classifier

    Attributes
    ____________
    model : keras model
        The model after fitting
    '''
    def __init__(self):
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
                continuity_window = 25):
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
        precision_recall_f1(np.abs(predictions), np.abs(labels))

#unit test
if __name__ == "__main__":
    #loading the data
    import pickle
    #NOTE: you'll probably have to change the file path to get this unit test to run
    train_data = pickle.load(open("/Users/williamlevine/Downloads/6-seconds-trial-1.MultFeat"))
    train_labels = np.abs(train_data[1])
    train_x = np.array(train_data[0])


    #running the CNN and producing results
    lr = LR()
    lr.fit(train_x, train_labels)

    #validation on a completely different data set
    test_data = pickle.load(open("/Users/williamlevine/Downloads/mixture-trial-4.MultFeat"))
    test_labels = np.abs(test_data[1])
    test_x = np.array(test_data[0])

    lr.evaluate(test_x, test_labels)
