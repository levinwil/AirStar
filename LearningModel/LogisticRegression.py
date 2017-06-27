from sklearn.linear_model import LogisticRegression
from utils import train_test_split
import math
import numpy as np
from utils import precision_recall_f1
import pylab

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
    verification_window : Int
        the window you view behind to check for rapid changes between classes
        predictions to prevent the model being confused
    min_size : Int
        the smallest interval of detection you'd like (if this is a fast
        movement, this should be small. If this is a slow movement, this should
        be large.)


    Outputs
    lr: the predictions
    '''
    def getPredictions(self,
                data,
                verification_window = 5,
                min_size = 130):
        #if there is only 1 feature
        if len(np.array(data).shape) == 1:
            data = np.reshape(data, (-1, 1))
        predictions =  self.model.predict(data)
        #fix the intervals which rapidly change between classes
        for i in range(verification_window, len(predictions)):
            std = np.std([predictions[i - verification_window: i]])
            if std > 0.5:
                for j in range(verification_window):
                    predictions[i - j] = -1
        #getting rid of small intervals. Specifically, getting rid of less than
        #min_size / 200 second intervals
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
            elif predictions[i - 1] == -1 and predictions[i] == 1:
                j = i - 1
                while predictions[j] == -1 and j > 0:
                    j = j -1
                if (i - j) <  min_size:
                    for k in range(i - j):
                        predictions[i - k] = predictions[j - 1]
            elif predictions[i - 1] == 1 and predictions[i] == -1:
                j = i - 1
                while predictions[j] == 1 and j > 1:
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
    min_size : Int
        the minimum interval for a detection
    label_value: Any
        the value you assigned to your labels
    display_feat : Int (0 -> num_features)
        the feature you'd like to display on the graph w/ predictions & labels
    '''
    def evaluate(self,
                 data,
                 labels,
                 min_size = 130,
                 label_value = 1,
                 display_feat = 1):
        #predict the remaining data using the generated logistic regression model
        predictions = self.getPredictions(data, min_size = min_size)
        pylab.plot(np.array(data[:,display_feat]) \
                            / np.max(np.array(data[:, display_feat])), \
                            label = "Raw Data Feature " + str(display_feat))
        pylab.plot(np.array(labels) / 1.5, label="Labels")
        pylab.plot(np.array(predictions) / 1.5, label="Predictions")
        pylab.xlabel("Time")
        pylab.ylabel("Unit of Feature " + str(display_feat))
        pylab.legend(loc = 'upper right')
        pylab.ylim(-1.2, 1.2)
        pylab.show()
        print 'Performance on test data: '
        precision_recall_f1(np.abs(predictions), np.abs(labels), label_value)

#unit test
if __name__ == "__main__":
    #loading the data
    import pickle
    #NOTE: you'll probably have to change the file path to get this unit test to run
    train_data = pickle.load(open("/Users/williamlevine/Downloads/concat_train.MultFeat"))
    train_labels = train_data[1]
    train_x = train_data[0]


    #running the LR and producing results
    lr = LR(train_x, train_labels)

    #validation on a completely different data set
    test_data = pickle.load(open("/Users/williamlevine/Downloads/3-Seconds-Will-Trial-2.MultFeat"))
    test_labels = test_data[1]
    test_x = np.array(test_data[0])
    lr.evaluate(test_x, test_labels)
