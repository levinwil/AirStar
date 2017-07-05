import cv2
import time
import pickle
import numpy as np

from skimage.measure import label
from skimage.transform import resize, rotate

from sklearn.neighbors import KDTree


#becuse opencv requires a callback for trackbars
def nothing(x):
    pass

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


def extract_hand(frame, hMin, sMin, vMin, hMax, sMax, vMax):
    hsvFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    threshedFrame = cv2.inRange(hsvFrame,
                                np.array([hMin, sMin, vMin]),
                                np.array([hMax, sMax, vMax]))

    ero = threshedFrame
    for i in range(6):
        ero = cv2.erode(ero, (5, 5), 1)
    dil = ero
    for i in range(10):
        dil = cv2.dilate(dil, (5, 5), 1)

    labelImg = label(dil)

    maxArea = 0
    maxLabel = 0

    #assume 0 label is background
    for i in range(1, np.max(labelImg)):
        area = len(zip(*np.where(labelImg == i)))
        if area > maxArea:
            maxArea = area
            maxLabel = i

    return dil, (labelImg == maxLabel).astype(np.uint8)*255

'''
def close_bbox(bbox):

    xCandid = np.zeros_like(bbox)
    yCandid = np.zeros_like(bbox)

    bboxr = (rotate(bbox, 45)).astype(np.uint8)
    xrCandid = np.zeros_like(bboxr)
    yrCandid = np.zeros_like(bboxr)

    for y in range(bbox.shape[0]):
        xNonZero = np.where(bbox[y])
        if len(xNonZero[0]):
            minX = np.min(xNonZero)
            maxX = np.max(xNonZero)
            xCandid[y, minX:maxX] = 1

    for x in range(bbox.shape[1]):
        yNonZero = np.where(bbox[:,x])
        if len(yNonZero[0]):
            minY = np.min(yNonZero)
            maxY = np.max(yNonZero)
            yCandid[minY:maxY, x] = 1


    for y in range(bboxr.shape[0]):
        xNonZero = np.where(bboxr[y])
        if len(xNonZero[0]):
            minX = np.min(xNonZero)
            maxX = np.max(xNonZero)
            xrCandid[y, minX:maxX] = 1

    for x in range(bboxr.shape[1]):
        yNonZero = np.where(bboxr[:,x])
        if len(yNonZero[0]):
            minY = np.min(yNonZero)
            maxY = np.max(yNonZero)
            yrCandid[minY:maxY, x] = 1

    rotate(xrCandid, -45)
    rotate(yrCandid, -45)

    toActivateFlat = np.multiply(xCandid, yCandid)
    toActivateRot = np.multiply(xrCandid, yrCandid)
    toActivate = np.multiply(toActivateRot, toActivateFlat)

    return (np.logical_or(toActivate, bbox)).astype(np.uint8)*255
'''

if __name__ == '__main__':
    cap = open_capture(ignore=[0])
    if cap is None:
        exit()

    labelSeries = []
    start = time.time()


    cv2.namedWindow('Thresh')
    cv2.createTrackbar('HMin', 'Thresh', 0, 255, nothing)
    cv2.createTrackbar('SMin', 'Thresh', 0, 255, nothing)
    cv2.createTrackbar('VMin', 'Thresh', 92, 255, nothing)
    cv2.createTrackbar('HMax', 'Thresh', 184, 255, nothing)
    cv2.createTrackbar('SMax', 'Thresh', 55, 255, nothing)
    cv2.createTrackbar('VMax', 'Thresh', 255, 255, nothing)

    cv2.namedWindow('Writing')
    cv2.createTrackbar('OnOff', 'Writing', 0, 1, nothing)

    cv2.namedWindow('Label')
    cv2.createTrackbar('Thumb', 'Label', 0, 1, nothing)
    cv2.createTrackbar('Index', 'Label', 0, 1, nothing)
    cv2.createTrackbar('Middle', 'Label', 0, 1, nothing)
    cv2.createTrackbar('Ring', 'Label', 0, 1, nothing)
    cv2.createTrackbar('Little', 'Label', 0, 1, nothing)
    fingerList = ['Thumb', 'Index', 'Middle', 'Ring', 'Little']

    while 1:

        flag, frame = cap.read()
        dil, handImg = extract_hand(frame,
                                    cv2.getTrackbarPos('HMin', 'Thresh'),
                                    cv2.getTrackbarPos('SMin', 'Thresh'),
                                    cv2.getTrackbarPos('VMin', 'Thresh'),
                                    cv2.getTrackbarPos('HMax', 'Thresh'),
                                    cv2.getTrackbarPos('SMax', 'Thresh'),
                                    cv2.getTrackbarPos('VMax', 'Thresh'))

        #cv2.imshow('handImg', handImg)

        positiveList = np.where(handImg)

        yMin = np.min(positiveList[0])
        xMin = np.min(positiveList[1])
        yMax = np.max(positiveList[0])
        xMax = np.max(positiveList[1])

        bbox = handImg[yMin:yMax, xMin:xMax]

        scaledBbox = resize(bbox, (256, 256))

        cv2.imshow('scaledBbox', scaledBbox)

        if cv2.getTrackbarPos('OnOff', 'Writing'):

            handLabel = [0, 0, 0, 0, 0]
            for idx, fingerKey in enumerate(fingerList):
                handLabel[idx] = cv2.getTrackbarPos(fingerKey, 'Label')

            labelSeries.append([handLabel, scaledBbox])
            labelSeries.append([handLabel, scaledBbox[:,::-1]])

        key = cv2.waitKey(100)
        if key == 13:
            break

    pickle.dump(labelSeries, open(str(start) + '.pkl', 'w'))
