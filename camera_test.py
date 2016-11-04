#!/usr/bin/env python

import numpy as np
import time
from PIL import Image
import cv2


# cap = cv2.VideoCapture(0)

def test_cam():
    #capture from camera at location 0
    cap = cv2.VideoCapture(1)
    # Change the camera setting using the set() function
    cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 160)
    cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 120)
    # cap.set(cv2.cv.CV_CAP_PROP_FPS, 60)
    # cap.set(cv2.cv.CV_CAP_PROP_EXPOSURE, -6.0)
    # cap.set(cv2.cv.CV_CAP_PROP_GAIN, 4.0)
    # cap.set(cv2.cv.CV_CAP_PROP_BRIGHTNESS, 144.0)
    # cap.set(cv2.cv.CV_CAP_PROP_CONTRAST, 27.0)
    # cap.set(cv2.cv.CV_CAP_PROP_HUE, 13.0) # 13.0
    # cap.set(cv2.cv.CV_CAP_PROP_SATURATION, 28.0)
    # Read the current setting from the camera
    test = cap.get(cv2.cv.CV_CAP_PROP_POS_MSEC)
    ratio = cap.get(cv2.cv.CV_CAP_PROP_POS_AVI_RATIO)
    frame_rate = cap.get(cv2.cv.CV_CAP_PROP_FPS)
    width = cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
    brightness = cap.get(cv2.cv.CV_CAP_PROP_BRIGHTNESS)
    contrast = cap.get(cv2.cv.CV_CAP_PROP_CONTRAST)
    saturation = cap.get(cv2.cv.CV_CAP_PROP_SATURATION)
    hue = cap.get(cv2.cv.CV_CAP_PROP_HUE)
    gain = cap.get(cv2.cv.CV_CAP_PROP_GAIN)
    exposure = cap.get(cv2.cv.CV_CAP_PROP_EXPOSURE)
    print("Test: ", test)
    print("Ratio: ", ratio)
    print("Frame Rate: ", frame_rate)
    print("Height: ", height)
    print("Width: ", width)
    print("Brightness: ", brightness)
    print("Contrast: ", contrast)
    print("Saturation: ", saturation)
    print("Hue: ", hue)
    print("Gain: ", gain)
    print("Exposure: ", exposure)
    while True:
        start_time = time.time()
        ret, img = cap.read()
        cv2.imshow("input", img)
        file = "out/fast+%s.png" % start_time
        # A nice feature of the imwrite method is that it will automatically choose the
        # correct format based on the file extension you provide. Convenient!
        cv2.imwrite(file, img)

        end_time = time.time()
        seconds = end_time - start_time
        print "Time: {0} seconds".format(seconds)
        # time.sleep(0.001)

        key = cv2.waitKey(10)
        if key == 27:
            break

    cv2.destroyAllWindows()
    cv2.VideoCapture(1).release()

test_cam()

# while(True):
#     # Capture frame-by-frame
#     ret, frame = cap.read()

#     print type(frame)
#     im = Image.fromarray(frame)
#     im.save("out.png")

#     time.sleep(1)


# # When everything done, release the capture
# cap.release()
# cv2.destroyAllWindows()

