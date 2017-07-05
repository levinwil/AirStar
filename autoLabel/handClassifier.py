import glob
import pickle

import numpy as np
import tensorflow as tf

from random import random, shuffle

EPOCHS = 1
TRAIN = True
LEARNING_RATE = .001
BATCH_SIZE = 16
DISPLAY_STEP = 10


def get_data(path):
    allFiles = glob.glob(path+'/*')
    myData = []

    for curFile in allFiles:
        curDat = pickle.load(open(curFile, 'r'))
        myData += curDat

    shuffle(myData)
    train = []
    val = []

    for i, point in enumerate(myData):
        if not i%10:
            val.append(point)
        else:
            train.append(point)

    return val, train


def get_batch(samples, batchSize):
    for i in range(0, len(samples), batchSize):
        try:
            curLabels = np.array([samples[i+j][0] for j in range(batchSize)])
            curInputs = np.array([samples[i+j][1] for j in range(batchSize)])
            yield curLabels, curInputs

        except IndexError:
            return


def model_function(batch, train=False):

    inputBatch = tf.reshape(batch, [-1, 256, 256, 1])

    conv1 = tf.layers.conv2d(inputBatch,
                             filters=4,
                             kernel_size=[8, 8],
                             activation=tf.nn.relu)

    pool1 = tf.layers.max_pooling2d(conv1,
                                    pool_size=[4, 4],
                                    strides=4)

    conv2 = tf.layers.conv2d(pool1,
                             filters=16,
                             kernel_size=[4, 4],
                             activation=tf.nn.relu)

    pool2 = tf.layers.max_pooling2d(conv2,
                                    pool_size=[8, 8],
                                    strides=8)

    flattened = tf.reshape(pool2, [-1, 7*7*16])

    dense = tf.layers.dense(flattened,
                            units=1024,
                            activation=tf.nn.relu)

    dropout = tf.layers.dropout(dense,
                                rate=.5,
                                training=train)

    reduction = tf.layers.dense(dropout,
                                units=5,
                                activation=tf.sigmoid)

    return reduction


def get_loss(y, y_):
    l2 = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(y, y_)),reduction_indices=1)))
    return l2

y = tf.placeholder(tf.float32, (None, 5))
x = tf.placeholder(tf.float32, (None, 256, 256))

y_ = model_function(x, TRAIN)
loss = get_loss(y, y_)
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

if __name__ == '__main__':

    val, train = get_data('./handData')

    if TRAIN:
        with tf.Session() as sess:
            sess.run(init)
            for epoch in range(EPOCHS):
                curStep = 0
                for curLabels, curSamples in get_batch(train, BATCH_SIZE):
                    sess.run(optimizer, feed_dict={y: curLabels,
                                                   x: curSamples})

                    if not curStep % DISPLAY_STEP:
                        batchLoss = sess.run(loss, feed_dict={y: curLabels,
                                                              x: curSamples})

                        print 'Epoch: ', epoch ,'\tBatch: ', curStep, '\tLoss: ', batchLoss
                    curStep +=1

                saver.save(sess, './models/rps-model', global_step=epoch)

            total = 0
            correct = 0
            for curLabels, curSamples in get_batch(val, 1):
                pred = np.rint(sess.run(y_, feed_dict={y: curLabels,
                                                       x: curSamples}))

                print pred[0]
                print curLabels[0]
                print all(pred[0] == curLabels[0])
                print '\n\n'

                if all(pred[0] == curLabels[0]):
                    correct+=1
                total+=1

            print 'Test Accuracy: ', correct/float(total)
