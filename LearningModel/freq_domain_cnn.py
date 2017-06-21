import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import sys
from utils import train_test_split, precision_recall_f1
sys.path.append("../pre-processing")
import numpy as np

''' A general model for EMG signal classification'''
class CNN(object):

    '''
    Convolutional Neural Network Classifier
    Parameters
    ____________
    window : int
        The length of the window you consider for the FFT
        (MUST BE A PERFECT SQUARE)
        Generally, if you are going to do short bursts of muscle contraction,
        the window should be between 25 & 49
    Attributes
    ____________
    model : keras model
        The model after fitting
    '''
    def __init__(self, window=36):
        self.window = window
        self.model = None

    """
    fit
    fits the CNN to the input data and labels. Optimizes using gradient descent,
    and then evaluates how it did on a subset of your data.
    Parameters
    ---------
    X : 2d-array
        the data, where X[i] MUST BE a 1d array of size window
    labels  : 1d-array
        the labels
    numClasses : int
        the number of classes you'd like to predict
    testSize : Double
        the proportion of data you want to set aside for testing
        (MUST BE BETWEEN 0 and 1)
    batchSize : int
        the size of the batches used for gradient descent optimization
    epochs : int
        the number of iterations in which the model evaluates a batch to
        optimize
    Return
    ------
    self : object
        self is fitted
    """
    def fit(self, X, labels, numClasses=2, testSize=.25, batchSize=36, epochs=40):
        #splitting into training and testing
        x_train, x_test, y_train, y_test = train_test_split(X,
                                                            labels,
                                                            testSize = testSize)

        # input image dimensions
        img_rows, img_cols = int(np.sqrt(self.window)), int(np.sqrt(self.window))

        #some reshaping
        x_test = x_test.reshape(x_test.shape[1], img_rows, img_cols, 1)
        x_train = x_train.reshape(x_train.shape[1], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

        x_train = x_train
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')

        # convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(y_train, numClasses)
        y_test = keras.utils.to_categorical(y_test, numClasses)


        #the CNN model
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=input_shape))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(numClasses, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])

        model.fit(x_train, y_train,
                  batch_size=batchSize,
                  epochs=epochs,
                  verbose=1,
                  validation_data=(x_test, y_test))
        self.model = model
        labels = [1 if y_test[i][1] > y_test[i][0] else 0
            for i in range(10, len(y_test))]
        x_test = np.array(x_test).reshape(1, x_test.shape[0], window)
        self.evaluate(x_test, labels)

    '''
    predict_discrete
    makes predictions based off of a trained model (will produce 0 or 1 for
    each time point)
    Parameters
    ____________
    X : 2d array
        the data you want to predict
    Returns
    ____________
    predictions : 1d array
        discrete values (either a 0 or a 1 at each time point)
    '''
    def predict_discrete(self, X):
        #reshape to be sqrt(window) x sqrt(window)
        X = np.array(X).reshape(X.shape[1],
                                int(np.sqrt(self.window)),
                                int(np.sqrt(self.window)),
                                1)
        #get prediction probabilities
        probabilities = self.model.predict(X)
        #for each time step, find if it's more likely to be class 1 or 0
        predictions = [1 if probabilities[i][1] > probabilities[i][0] else 0
            for i in range(10, len(probabilities))]
        return np.array(predictions)

    '''
    predict_continuous
    makes predictions based off of a trained model (will produce doubles
    between 0 and 1)
    Parameters
    ____________
    X : 2d array
        the data you want to predict
    Returns
    ____________
    predictions : 2d array
        continuous values (each timepoint is a 1d array, where the value at
        index i is the probability that timepoint belongs to class i)
    '''
    def predict_continuous(self, X):
        #reshape to be sqrt(window) x sqrt(window)
        X = np.array(X).reshape(X.shape[1],
                                np.sqrt(self.window),
                                np.sqrt(self.window),
                                1)
        return np.array(self.model.predict(X))



    '''
    evaluate
    evaluates a model's prediction performance on a data set against its set of labels
    Parameters
    ____________
    X : 2d array
        the data the model will predict
        X[i] is of size window
    labels : 1d array
        the ground truth
    Returns
    ____________
    void
    '''
    def evaluate(self, X, labels):
        predictions = self.predict_discrete(X)
        print 'Performance on test data: '
        precision_recall_f1(predictions, labels)



#unit test
if __name__ == "__main__":
    #variables that you can play around with
    window = 36 #must be a perfect square
    epochs = 12 #you probably want to keep this between 0 and 100 if you want it running < 5 minutes


    #loading the data
    import pickle
    #NOTE: you'll probably have to change the file path to get this unit test to run
    train_data = pickle.load(open("/Users/williamlevine/Downloads/OpenBCI-RAW-Mixture-Trial-4.DatLabl"))
    train_labels = train_data[1]
    train_x = np.array([train_data[0]])


    #running the CNN and producing results
    cnn = CNN(window = window)
    cnn.fit(train_x, train_labels, epochs = epochs)

    #validation on a completely different data set
    test_data = pickle.load(open("/Users/williamlevine/Downloads/OpenBCI-RAW-2017-Fast-Richard-Trial-1.DatLabl"))
    test_labels = test_data[1]
    test_x = np.array([test_data[0]])

    cnn.evaluate(test_x, test_labels)
