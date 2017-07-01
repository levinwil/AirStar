import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from utils import train_test_split, precision_recall_f1
import pylab
import numpy as np

''' A general model for EMG signal classification'''
class MLP(object):

    '''
    Multi-Layer Perceptron Classifier
    Parameters
    ____________
    num_features : int
        the number of features that you will be using
    Attributes
    ____________
    model : keras model
        The model after fitting
    '''
    def __init__(self, num_features=3, model_path = None):
        self.num_features = num_features
        if model_path != None:
            self.model = load_model(model_path)
        else:
            self.model = None


    '''
    Set_Model

    Parameters
    ____________
    model_path : String
        the path of the model which you are setting

    '''
    def set_model(self, model_path):
        self.model = load_model(model_path)



    """
    fit
    fits the MLP to the input data and labels. Optimizes using gradient descent,
    and then evaluates how it did on a subset of your data.
    Parameters
    ---------
    X : 2d-array
        the data, where X[i] MUST BE a 1d array of size num_features
    labels  : 1d-array
        the labels
    num_classes : int
        the number of classes you'd like to predict
    test_size : Double
        the proportion of data you want to set aside for testing
        (MUST BE BETWEEN 0 and 1)
    batch_size : int
        the size of the batches used for gradient descent optimization
    epochs : int
        the number of iterations in which the model evaluates a batch to
        optimize
    Return
    ------
    self : object
        self is fitted
    """
    def fit(self, X, labels, num_classes=2, test_size=.25, batch_size=36,
            epochs=40):
        #splitting into training and testing
        x_train, x_test, y_train, y_test = train_test_split(X,
                                                            labels,
                                                            test_size = test_size)

        x_train = x_train[0]
        x_test = x_test.astype('float32')[0]
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')

        # convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)


        #the MLP model
        model = Sequential()
        model.add(Dense(1024, activation='relu', input_dim=self.num_features))
        model.add(Dropout(0.5))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(3, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])

        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  validation_data=(x_test, y_test))

        self.model = model
        labels = np.zeros((len(y_test)))
        for i in range(len(labels)):
            labels_at_time = y_test[i]
            max_index = 0
            if labels_at_time[1] > labels_at_time[max_index]:
                max_index = 1
            if labels_at_time[2] > labels_at_time[max_index]:
                max_index = -1
            labels[i] = max_index
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
    def predict_discrete(self,
                         X,
                         verification_window = 5,
                         continuity_window = 100,
                         min_size = 65):
        #get prediction probabilities
        probabilities = self.model.predict(X)
        #for each time step, find if it's more likely to be class 1 or 0
        predictions = [np.argmax(probabilities[i]) for i in range(len(probabilities))]
        predictions = [-1 if predictions[i] == 2 else predictions[i] for i in range(len(predictions))]
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
        return np.array(self.model.predict(X))



    '''
    evaluate

    evaluates a model's prediction performance on a data set against its set of
    labels

    Parameters
    ____________
    X : 2d array
        the data the model will predict
        X[i] is of size num_features
    labels : 1d array
        the ground truth
    display_feat : int
        the feature you'd like yo visualize to further evaluate if the
        predictions are or are not correct
    Returns
    ____________
    void
    '''
    def evaluate(self, X, labels, min_size = 65, display_feat = 1):
        predictions = self.predict_discrete(X, min_size = min_size)
        pylab.plot(np.array(X[:,display_feat]) \
                            / np.max(np.array(X[:, display_feat])), \
                            label = "Raw Data Feature " + str(display_feat))
        pylab.plot(np.array(labels) / 1.5, label="Labels")
        pylab.plot(np.array(predictions) / 1.5, label="Predictions")
        pylab.xlabel("Time")
        pylab.ylabel("Unit of Feature " + str(display_feat))
        pylab.legend(loc = 'upper right')
        pylab.ylim(-1.2, 1.2)
        pylab.show()
        print 'Performance on test data: '
        precision_recall_f1(predictions, labels)

    '''
    save_model

    saves the MLP model

    Parameters
    ____________
    file_path : String
        the file path to save the model at (must end in '.h5')

    Returns
    ____________
    void
    '''
    def save_model(self, file_path):
        self.model.save(file_path)



#unit test
if __name__ == "__main__":
    num_features = 3

    #variables that you can play around with
    epochs = 100 #you probably want to keep this between 0 and 100 if you want it running < 5 minutes
    batch_size = 64

    #loading the data
    import pickle
    #NOTE: you'll probably have to change the file path to get this unit test to run
    train_data = pickle.load(open("/Users/williamlevine/Downloads/concat_train.MultFeat"))
    train_labels = train_data[1]
    train_x = np.array([train_data[0]])


    #running the MLP and producing results
    MLP = MLP(num_features  = num_features)
    MLP.fit(train_x, train_labels, epochs = epochs, batch_size = batch_size, num_classes = 3, test_size = .05)
    MLP.save_model("/Users/williamlevine/Documents/BCI/AirStar/LearningModel/saved_models/MLP.h5")

    #validation on a completely different data set
    test_data = pickle.load(open("/Users/williamlevine/Downloads/3-Seconds-Will-Trial-2.MultFeat"))
    test_labels = test_data[1]
    test_x = np.array([test_data[0]])

    MLP.evaluate(test_x[0], test_labels, min_size = 170)
