from sklearn.ensemble import RandomForestClassifier
from utils import train_test_split
import math
import numpy as np
from utils import precision_recall_f1


'''
A basic random forest classifier that will report precision, recall, and F1 score
on 25 percent of the data.

Inputs
data: a 2d array, where x is time and y is value
labels: a 2d array, where x is time and y is the label
label_value: the value you assigned to your labels
n_estimators: the number of trees in the random forest
n_jobs: the number of cores you want to use in parallel
test_size: the proportion of the data you want to use for verification (precision, recall, F1)

Outputs
rf: the fitted random forest

NOTE: if you get some freaky, error, it's probably in n_jobs. If n_jobs = -1,
the random forest classifier trains itself in parallel using all its cores,
which sometimes gives you some pretty freaky errors. So, just set n_jobs = 1.
'''
def analyzeRF(data, labels, label_value = 1, n_estimators = 10, n_jobs = -1, test_size = 0.25):
    # if there is 1 feature,
    if len(np.array(data).shape) == 1:
        data = np.reshape(data, (-1, 1))
    #split the data
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size = test_size)
    #generate the random forest
    rf = generateRF(x_train, y_train, n_estimators = n_estimators, n_jobs = n_jobs)
    #predict the remaining data using the generated random forest
    predictions = getPredictions(x_test, rf)

    precision_recall_f1(y_test, predictions)
    return rf


'''
A basic random forest classifier

Inputs
data: a 2d array, where x is time and y is value
labels: a 2d array, where x is time and y is the label
n_estimators: the number of trees in the random forest
n_jobs: the number of cores you want to use in parallel

Outputs
rf: the fitted random forest
'''
def generateRF(data, labels, n_estimators = 10, n_jobs = 1):
    rf = RandomForestClassifier(n_estimators = n_estimators, n_jobs = n_jobs)
    rf.fit(data, labels)
    return rf

'''
A basic random forest classifier

Inputs
data: a 2d array, where x is time and y is value
rf: a pre-fitted random forest

Outputs
rf: the predictions
'''
def getPredictions(data, rf):
    return rf.predict(data)
