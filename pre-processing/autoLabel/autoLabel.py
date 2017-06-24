import cv2
import time
import numpy as np

from skimage.measure import label

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

def get_next_label(frame, hMin, sMin, vMin, hMax, sMax, vMax):
    hsvFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    threshedFrame = cv2.inRange(hsvFrame,
                                np.array([hMin, sMin, vMin]),
                                np.array([hMax, sMax, vMax]))

    ero = threshedFrame
    for i in range(5):
        ero = cv2.erode(ero, (5, 5), 1)
    dil = ero
    for i in range(7):
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

    return dil


if __name__ == '__main__':
    cap = open_capture(ignore=[0])
    if cap is None:
        exit()

    labelSeries = []
    start = time.time()


    cv2.namedWindow('Thresh')
    cv2.createTrackbar('HMin', 'Thresh', 0, 255, nothing)
    cv2.createTrackbar('SMin', 'Thresh', 16, 255, nothing)
    cv2.createTrackbar('VMin', 'Thresh', 18, 255, nothing)
    cv2.createTrackbar('HMax', 'Thresh', 48, 255, nothing)
    cv2.createTrackbar('SMax', 'Thresh', 150, 255, nothing)
    cv2.createTrackbar('VMax', 'Thresh', 79, 255, nothing)

    while 1:
        flag, frame = cap.read()
        img = get_next_label(frame,
                             cv2.getTrackbarPos('HMin', 'Thresh'),
                             cv2.getTrackbarPos('SMin', 'Thresh'),
                             cv2.getTrackbarPos('VMin', 'Thresh'),
                             cv2.getTrackbarPos('HMax', 'Thresh'),
                             cv2.getTrackbarPos('SMax', 'Thresh'),
                             cv2.getTrackbarPos('VMax', 'Thresh'))


        cv2.imshow('img', img)
        cv2.waitKey(100)
        #labelSeries.append(get_next_label(frame))

    #TODO write labels here
