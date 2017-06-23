from sklearn.linear_model import LogisticRegression
from utils import train_test_split
import math
import numpy as np
from utils import precision_recall_f1


'''
A basic logistic regression model classifier that will report precision, recall, and F1 score
on 25 percent of the data.

Inputs
data: a 2d array, where x is time and y is value
labels: a 2d array, where x is time and y is the label
label_value: the value you assigned to your labels
n_estimators: the number of trees in the logistic regression model
n_jobs: the number of cores you want to use in parallel
test_size: the proportion of the data you want to use for verification (precision, recall, F1)

Outputs
lr: the fitted logistic regression model

NOTE: if you get some freaky, error, it's probably in n_jobs. If n_jobs = -1,
the logistic regression model classifier trains itself in parallel using all its cores,
which sometimes gives you some pretty freaky errors. So, just set n_jobs = 1.
'''
def analyzeLR(data, labels, label_value = 1, test_size = 0.25, channels_present = False):
    #split the data
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size = test_size, channels_present = channels_present)
    #generate the logistic regression model
    lr = generateLR(x_train, y_train)
    #predict the remaining data using the generated logistic regression model
    predictions = getPredictions(x_test, lr)
    print predictions

    precision_recall_f1(predictions, y_test)
    return lr


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
def generateLR(data, labels):
    #if there is only 1 feature
    if len(np.array(data).shape) == 1:
        data = np.reshape(data, (-1, 1))
    lr = LogisticRegression()
    lr.fit(data, labels)
    return lr

'''
A basic logistic regression model classifier

Inputs
data: a 2d array, where x is time and y is value
lr: a pre-fitted logistic regression model

Outputs
lr: the predictions
'''
def getPredictions(data, lr):
    verification_window = 5
    continuity_window = 25
    #if there is only 1 feature
    if len(np.array(data).shape) == 1:
        data = np.reshape(data, (-1, 1))
    predictions =  lr.predict(data)
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
