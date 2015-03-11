#!/usr/bin/env python
"""
Tools to track position of objects (balls) in stratified fluid
"""

import numpy
from PIL import Image
import cv2
import cv
import argparse
import os
import pylab as py
import labdb


def estimate_position(image, threshold_val, blur_val):
    """
    Estimate the position (in pixels) of a ball in image
    """

    imgray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    imgray = cv2.blur(imgray, (1,blurval))
    ret, thresh = cv2.threshold(imgray, threshval, 255, cv2.THRESH_BINARY_INV)
    contours, hierarchy  = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    count = 0;
    for cnt in contours:

        if len(cnt) < 5:
            continue

        area = cv2.contourArea(cnt)

        if area < 1000:
            continue

        (x,y), radius = cv2.minEnclosingCircle(cnt)

        if radius > 100:
            continue

        center = (int(x), int(y))
        cv2.circle(imgray, center, int(radius), 180, -1)

        ellipse = cv2.fitEllipse(cnt)
        print ellipse,
        (x,y), (w,h), theta = ellipse
        #if x > 200 and x < 1000 and y > 200 and y < 600:
        cv2.ellipse(imgray, ellipse, 127, 2)
        count += 1
    print count

    return x, y

def track_object(video_id):

    # identify a video stream to use (video_id)
    db = labdb.LabDB()
    rows = db.execute("SELECT path FROM video WHERE video_id = %d" % video_id)
    if len(rows) == 0:
        print "video_id %d not found." % video_id
        return
    
    rootpath = rows[0][0]
    
    path = os.path.join(rootpath, "frame%05d.png")

    if not os.path.exists(path % 0):
        path = os.path.join(rootpath, "frame%05d.pgm")

    print "Using", path

    capture = cv2.VideoCapture(path)

    retval, frame = capture.read()
    thresh = 0
    imgray = 0
    thresh = 0

    def onChange(val):

        blurval = cv2.getTrackbarPos("blur","video")
        threshval = cv2.getTrackbarPos("threshold","video")
        imgray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        imgray = cv2.blur(imgray, (1,blurval))
        ret, thresh = cv2.threshold(imgray, threshval, 255, cv2.THRESH_BINARY_INV)
        contours, hierarchy  = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        #ret, imgray = cv2.threshold(imgray, threshval, 255, cv2.THRESH_BINARY_INV)

        print threshval, blurval,

        count = 0;
        for cnt in contours:

            if len(cnt) < 5:
                continue

            area = cv2.contourArea(cnt)

            if area < 1000:
                continue

            (x,y), radius = cv2.minEnclosingCircle(cnt)

            if radius > 100:
                continue

            center = (int(x), int(y))
            cv2.circle(imgray, center, int(radius), 180, -1)

            ellipse = cv2.fitEllipse(cnt)
            print ellipse,
            (x,y), (w,h), theta = ellipse
            #if x > 200 and x < 1000 and y > 200 and y < 600:
            cv2.ellipse(imgray, ellipse, 127, 2)
            count += 1
        print count
 
        cv2.imshow("video", imgray[:,:])

    cv2.namedWindow("video")
    cv2.createTrackbar("blur", "video", 16, 255, onChange)
    cv2.createTrackbar("threshold", "video", 127, 255, onChange)

    onChange(0)

    # see blog posts at
    # http://opencvpython.blogspot.ca/2012/06/hi-this-article-is-tutorial-which-try.html
    # for great info on what is going on here.

    c = cv2.waitKey(0)

    while retval:

        onChange(None)

        c = cv2.waitKey(1)

        if c == 27:  #ESC
            break

        retval, frame = capture.read()

   

    
def main():
    """
    UI for code
    """

    # parse command line arguements
    parser = argparse.ArgumentParser(description="identify circular object in a video clip")

    parser.add_argument("video_id", type=int, help = "video_id")
    args = parser.parse_args()

    track_object(args.video_id)

if __name__ == "__main__":
    main()
    #track_object()

