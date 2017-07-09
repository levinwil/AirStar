import cv2
import glob
import time
import pickle

import numpy as np
import tensorflow as tf

from skimage.measure import label
from random import random, shuffle
from sklearn.neighbors import KDTree
from skimage.transform import resize, rotate
from getHandLabels import open_capture, extract_hand, nothing

EPOCHS = 1
TRAIN = False
LEARNING_RATE = .001
BATCH_SIZE = 16
DISPLAY_STEP = 10

def open_capture(ignore = []):
    for i in range(10):
        if not i in ignore:
            try:
                cap = cv2.VideoCapture(i)
                return cap
            except:
                continue
    print 'Error: Could Not Open Video Capture!'
    return None

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

    with tf.Session() as sess:
        if TRAIN:
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

        else:
            saver.restore(sess, './models/rps-model-0')
            cap = open_capture()

            #set up required GUI elements
            start = time.time()


            cv2.namedWindow('Thresh')
            cv2.createTrackbar('HMin', 'Thresh', 0, 255, nothing)
            cv2.createTrackbar('SMin', 'Thresh', 0, 255, nothing)
            cv2.createTrackbar('VMin', 'Thresh', 92, 255, nothing)
            cv2.createTrackbar('HMax', 'Thresh', 184, 255, nothing)
            cv2.createTrackbar('SMax', 'Thresh', 55, 255, nothing)
            cv2.createTrackbar('VMax', 'Thresh', 255, 255, nothing)

            while 1:
                flag, frame = cap.read()

                dil, handImg = extract_hand(frame,
                                            cv2.getTrackbarPos('HMin', 'Thresh'),
                                            cv2.getTrackbarPos('SMin', 'Thresh'),
                                            cv2.getTrackbarPos('VMin', 'Thresh'),
                                            cv2.getTrackbarPos('HMax', 'Thresh'),
                                            cv2.getTrackbarPos('SMax', 'Thresh'),
                                            cv2.getTrackbarPos('VMax', 'Thresh'))


                positiveList = np.where(handImg)

                yMin = np.min(positiveList[0])
                xMin = np.min(positiveList[1])
                yMax = np.max(positiveList[0])
                xMax = np.max(positiveList[1])

                bbox = handImg[yMin:yMax, xMin:xMax]

                scaledBbox = resize(bbox, (256, 256))

                cv2.imshow('scaledBbox', scaledBbox)

                prediction = np.rint(sess.run([y_], feed_dict={x:np.array([scaledBbox])}))

                print prediction

                key = cv2.waitKey(100)
                if key == 13:
                    break
