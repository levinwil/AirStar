import sys
import numpy as np
import tensorflow as tf

from tensorflow.contrib import rnn
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/temp/data', one_hot=True)

#hyperparams
learningRate = .005
epochs = 3
batchSize = 128
displayStep = 100

#network config
inputSize = 28
unrollingLength = 28
hiddenSize = 128
numClasses = 10

inputBatch = tf.placeholder(tf.float32, [None, unrollingLength, inputSize])
labelBatch = tf.placeholder(tf.float32, [None, numClasses])


def lstm_layer(inputBatch):

    '''
    LSTM layer

    TAKES
    -----
    > inputBatch: a tensor of [batch, unrollingLength, inputSize]

    RETURNS: a tensor of [hiddenSize]
    '''

    reshapedInput = tf.unstack(inputBatch,
                               unrollingLength,
                               1)

    cell = rnn.LSTMCell(hiddenSize,
                        forget_bias=1.0,
                        activation=None,
                        use_peepholes=False)

    outputs, states = rnn.static_rnn(cell, reshapedInput, dtype=tf.float32)

    return outputs[-1] #we only want the most current prediction


def fc_layer(lstmOut):

    '''
    Fully Connected layer to reduce hidden size to num classes

    TAKES
    -----
    > lstmOut: a tensor of [hiddenSize]

    RETURNS: a tensor of [numClasses]
    '''

    fc = tf.layers.dense(lstmOut,
                         units=numClasses,
                         activation=tf.nn.relu)

    return fc


def model_function(inputBatch):

    '''
    Defines the model function for the network

    TAKES
    -----
    > inputBatch: a tensor of [batchSize, unrollingLength, inputSize]

    RETURNS: a tensor of [numClasses]
    '''

    lstmOut = lstm_layer(inputBatch)
    fcOut = fc_layer(lstmOut)

    return fcOut


#optimizer config
prediction = model_function(inputBatch)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,
                                                              labels=labelBatch))
optimizer = tf.train.AdamOptimizer(learningRate).minimize(cost)

correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(labelBatch, 1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))


#Launch Graph Session
init = tf.global_variables_initializer()
with tf.Session() as session:
    session.run(init)
    saver = tf.train.Saver()

    if sys.argv[1] == 'train':
        #TODO write actual file IO here
        batchNum = 55000/batchSize
        for epoch in range(epochs):
            for batch in range(batchNum):

                #TODO write actual file IO here
                curInputBatch, curLabelBatch = mnist.train.next_batch(batchSize)
                curInputBatch = curInputBatch.reshape((batchSize, unrollingLength, inputSize))
                session.run(optimizer, feed_dict={inputBatch: curInputBatch,
                                                  labelBatch: curLabelBatch})

                if not epoch*batch + batch:
                    acc = session.run(accuracy, feed_dict={inputBatch: curInputBatch,
                                                          labelBatch: curLabelBatch})
                    print 'Epoch: ', epoch, '\tBatch:, ', batch, '\tAccuracy: ', acc
        saver.save(session, '/models/lstm.ckpt')

    elif sys.argv[1] == 'test':
        saver.restore(session,'/models/lstm.ckpt' )

        #TODO write actual file IO here
        curInputBatch = mnist.test.images[:batchSize].reshape((-1, unrollingLength, inputSize))
        curLabelBatch = mnist.test.labels[:batchSize]
        acc = session.run(accuracy, feed_dict={inputBatch: curInputBatch,
                                               labelBatch: curLabelBatch})
        print 'Test set accuracy: ', acc


    else:
        print 'Usage: python lstm.py <train/test>'
        exit()
