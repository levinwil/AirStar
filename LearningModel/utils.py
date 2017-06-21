import uuid
import tensorflow as tf
import numpy as np

"""
train_test_split

splits the data into training and testing data while still keeping context accross the time axis

Parameters
---------
array-like data           --- Either an np.ndarray or a pd.DataFrame containing the data
array-like labels         --- Either an np.ndarray or a pd.DataFrame containing the labels
numeric-type testSize    --- the proportion of the data & labels to set aside for testing purposes

Return
------
(in order) training_x, testing_x, training_y, testing_y

"""
def train_test_split(data, labels, testSize = 0.25):
    train_x = data[:, 0: int(len(data[0])*(1 - testSize)), :]
    test_x = data[:, int (len(data[0])*(1 - testSize)):, :]
    train_y = labels[0: int(len(labels)*(1 - testSize))]
    test_y = labels[int(len(labels)*(1 - testSize)):]
    return train_x, test_x, train_y, test_y

"""
precision_recall_f1

calculates the precision, recall, and f1 of your predictions in reference to your labels (note: does NOT calculate based on overlap, but ranges, rather)

Parameters
---------
array-like predictions       --- Either an np.ndarray or a pd.DataFrame containing the predictions
array-like labels            --- Either an np.ndarray or a pd.DataFrame containing the labels
numeric-type width           --- the window size to consider in labels when classifying a prediction, and in predictions when classifying a label

Return
------
(in order) precision, recall, F1

"""
def precision_recall_f1(predictions, labels, width = 50):
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    for i in range(len(predictions)):
        if predictions[i] == 1:
            lower_bound = -1 * width
            if (i + lower_bound < 0):
                lower_bound = 0
            upper_bound = width
            if (i + upper_bound > len(labels)):
                upper_bound = len(predictions) - 1 - i
            ra = [1 for j in range(lower_bound, upper_bound) if labels[i + j] == 1]
            if (np.sum(ra) > 0) :
                true_positives += 1
            else:
                false_negatives +=1
    for i in range(len(labels)):
        if labels[i] == 1:
            lower_bound = -1 * width
            if (i + lower_bound < 0):
                lower_bound = 0
            upper_bound = width
            if (i + upper_bound > len(predictions)):
                upper_bound = len(predictions) - 1 - i
            ra = [i for j in range(lower_bound, upper_bound) if predictions[i + j] == 1]
            if (np.sum(ra) <= 0) :
                false_positives +=1
    precision = true_positives * 1.0 / (true_positives + false_positives)
    recall = true_positives * 1.0 / (true_positives + false_negatives)
    F1 = 2 * precision * recall / (precision + recall)
    print "Precision: " + str(precision)
    print "Recall: " + str(recall)
    print "F1: " + str(F1)
    return precision, recall, F1


"""
batch_iter

A generator that yields one batch of a data set at a time, and does so for a given number of epochs. Useful for batch iteration in models that leverage mini-batch operations (i.e. gradient descent operations).

Parameters
---------
array-like data            --- Either an np.ndarray or a pd.DataFrame containing the data. Generally this is best used as a zipped x and y together, as a list
numeric-type batch_size    --- The size of the batches to use
numeric-type num_epochs    --- the number of training epochs
keyword bool shuffle       --- whether or not to shuffle batches before they are returned. Defaults to True

"""

def batch_iter(data, batch_size, num_epochs, shuffle=True):

    #convert internally to np array for uniformity
    if ( type(data) == type(pd.DataFrame) ):
        data = data.as_matrix()
    elif ( type(data) == type(list) ):
        data = np.array(data)

    data_size = len(data)
    batches_per_epoch = data_size // batch_size

    for i_epoch in range(0, num_epochs):

        #shuffle data randomly at each epoch (if specified)
        if shuffle:
            np.random.shuffle(data)

        #slice our data according into batches every epoch and yield the next one
        for j_batch in range(batches_per_epoch):
            start_idx = j_batch * batch_size
            end_idx = min( ( (j_batch + 1) * batch_size) , data_size )
            yield( data[start_idx:end_idx] )
