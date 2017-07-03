import numpy as np
from sklearn.metrics import accuracy_score

"""
train_test_split

splits the data into training and testing data while still keeping context accross the time axis

Parameters
---------
array-like data           --- Either an np.ndarray or a pd.DataFrame containing the data
array-like labels         --- Either an np.ndarray or a pd.DataFrame containing the labels
numeric-type test_size    --- the proportion of the data & labels to set aside for testing purposes

Return
------
(in order) training_x, testing_x, training_y, testing_y

"""
def train_test_split(data, labels, test_size = 0.25, channels_present = True):
    if channels_present:
        train_x = data[:, 0: int(len(data[0])*(1 - test_size)), :]
        test_x = data[:, int (len(data[0])*(1 - test_size)):, :]
        train_y = labels[0: int(len(labels)*(1 - test_size))]
        test_y = labels[int(len(labels)*(1 - test_size)):]
    else:
        train_x = data[0: int(len(data)*(1 - test_size)), :]
        test_x = data[int (len(data)*(1 - test_size)):, :]
        train_y = labels[0: int(len(labels)*(1 - test_size))]
        test_y = labels[int(len(labels)*(1 - test_size)):]
    return train_x, test_x, train_y, test_y


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

"""
accuracy

calculates the accuracy of your predictions in reference to your labels

Parameters
---------
array-like predictions       --- Either an np.ndarray or a pd.DataFrame containing the predictions
array-like labels            --- Either an np.ndarray or a pd.DataFrame containing the labels

Return
------
accuracy : float

"""
def accuracy(data, labels):
    accuracy = accuracy_score(data, labels, normalize = True)
    print 'Accuracy:' + str(accuracy)
    return accuracy
