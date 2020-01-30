# ------------------------------------------------------------------------------------------------ IMPORT LIBRARIES ----
import numpy as np
import cv2 as cv
# from Flea2Camera import FleaCam
from typing import NamedTuple


# -------------------------------------------------------------------------------------------------- CUSTOM CLASSES ----
class ConfidenceDisp(NamedTuple):  # custom class to store pertinent information to displaying confidence info
    orig: list  # coordinates (top left corner) of display
    scale: list  # number of pixels in x and y direction per tick
    spacing: int  # horizontal spacing in pixels between bars
    filled: bool  # flag for filling in (or not) rectangle -- true: filled; false: outline
    thickness: int  # thickness of rectangle if not filled


# ------------------------------------------------------------------------------------------------- USER PARAMETERS ----
debugFlag = True  # flag indicating whether or not to printout debug statements to console
beltThreshold = 30  # average of red, green, and blue max values for belt thresholding/detection
YGThreshold = 70  # threshold between yellow and green values (since they have the same brightness channels)
erodeIterations = 2  # erode iteration number
dilateIterations = 10  # dilate iteration number
# initialize paramters for confidence display
dispParams = ConfidenceDisp(orig=[8, 50], scale=[5, 15], spacing=8, filled=False, thickness=2)
yellow = (31, 214, 239)  # yellow bgr value
orange = (44, 141, 239)  # orange bgr value
red = (75, 75, 207)  # red bgr value
green = (66, 183, 136)  # green bgr value


# ----------------------------------------------------------------------------------- VARIABLES AND INITIALIZATIONS ----
# cap = FleaCam()  # initialize flea camera to read video
cap = cv.VideoCapture('2.avi')
firstFrame = True  # flag indicating first tick of unconditional loop
previousFrame = []  # array to hold previous frame data
confidenceVals = [0, 0, 0, 0]  # confidence values for goldfish colors (yellow, orange, red, green)


# ---------------------------------------------------------------------------------------------- UNCONDITIONAL LOOP ----
while True:
    # check if user wants to quit program
    k = cv.waitKey(1) & 0xFF  # read key from keyboard (wait maximum of 1ms)
    if k == 27:  # see if read key is 'esc'
        break  # if so, break from unconditional while loop

    # frame = cap.getFrame()  # get frame for this tick of loop
    ret, frame = cap.read()
    if ret:
        # check if is not the first time through the loop - ie, there is previous frame data
        if not firstFrame:
            # process camera data
            diff = cv.absdiff(frame, previousFrame)  # take difference of previous and current frame
            blur = cv.GaussianBlur(diff, (5, 5), 0)  # blur image using gaussian distribution
            erode = cv.erode(blur, None, iterations=erodeIterations)  # erode to remove noise
            dilate = cv.dilate(erode, None, iterations=dilateIterations)  # dilate to enlarge information data
            cv.imshow('debug', dilate)
            b, g, r = cv.split(dilate)  # split frame into blue, green, and red data arrays
            b = np.max(np.uint32(b))  # find maximum blue value from frame
            g = np.max(np.uint32(g))  # find maximum green value from frame
            r = np.max(np.uint32(r))  # find maximum red value from frame

            bgrAvg = np.average([b, g, r])
            if bgrAvg > beltThreshold:  # goldfish found in frame
                if debugFlag:  # want print statments for debugging purposes
                    print("RGB:({},{},{})".format(r, g, b))  # print out red value for current frame
                # yellow if: B,G - avg over 70
                # orange if: B,R
                # red if:    G,R
                # green if:  B,G - avg under 70
                if b > r and g > r:  # yellow or green detected
                    if bgrAvg > YGThreshold:  # yellow detected
                        confidenceVals[0] += 1  # increment confidence
                    else:  # green detected
                        confidenceVals[3] += 1  # increment confidence
                elif b > g and r > g:  # orange detected
                    confidenceVals[1] += 1  # increment confidence
                elif g > b and r > b:  # red detected
                    confidenceVals[2] += 1  # increment confidence
                else:  # unable to detect color based on values
                    pass
            else:  # only belt found in frame
                confidenceVals = [0, 0, 0, 0]  # reset confidence values to prepare for next goldfish
        else:  # if first time through loop
            firstFrame = False  # unset flag to indicate no longer first time through loop
            confidenceVals = [0, 0, 0, 0]  # reset confidence values to prepare for next goldfish

        previousFrame = frame  # store current frame as previous, preparatory to receiving a new frame next tick

        displayFrame = frame  # store current frame into display frame for user feedback
        for i in range(len(confidenceVals)):  # work through possible goldfish colors for confidence display
            # calculate (x, y) for starting point
            p1x = dispParams.orig[0]  # top left x coord
            p1y = dispParams.orig[1] + ((dispParams.scale[1] + dispParams.spacing) * i)  # top left y coord
            # calculate (x, y) for ending point (relative to starting point)
            p2x = p1x + (dispParams.scale[0] * confidenceVals[i])  # bottom right x coord
            p2y = p1y + dispParams.scale[1]  # bottom right y coord
            if i == 0:  # displaying yellow bar
                clr = yellow  # set color to yellow
            elif i == 1:  # displaying orange bar
                clr = orange  # set color to orange
            elif i == 2:  # displaying red bar
                clr = red  # set color to red
            elif i == 3:  # displaying green bar
                clr = green  # set color to green
            else:  # uncertain case
                clr = (255, 255, 255)  # white color
            fill = -1 if dispParams.filled else dispParams.thickness  # determine if rectangle will be filled in or not
            displayFrame = cv.rectangle(displayFrame, (p1x, p1y), (p2x, p2y), clr, fill)  # draw rectangle on frame
        cv.imshow('Goldfish Color Detection', displayFrame)  # display frame
    else:
        cap.set(cv.CAP_PROP_POS_AVI_RATIO, 0)
        firstFrame = True
