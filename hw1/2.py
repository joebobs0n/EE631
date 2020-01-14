import imutils
import cv2 as cv
import numpy as np

tgt_fps = 60 # desired camera fps - doesn't seem to work on pc, but does on raspberry pi
tgt_height = 600 # change height of processed/displayed video - width changed proportionally

# print out instructions to console
print("\nUsage Instructions:\n -ESC: exit feed\n -0: original mode\n -1: binarize mode\n -2: canny edges mode\n -3: corner detect mode\n -4: line detect mode\n -5: difference mode\n")

mode = 0 # initialize current mode variable
cap = cv.VideoCapture(0) # initialize video capture
if not cap.isOpened(): # check if camera successfully opened
    print("Cannot open camera") # if not - inform user
    exit() # exit program
cap.set(cv.CAP_PROP_FPS, tgt_fps) # set target fps
print("Starting OpenCV program at {} FPS".format(cap.get(cv.CAP_PROP_FPS))) # inform user of actual fps

while True: # run continuously
    ret, frame = cap.read() # read frame from camera capture
    if not ret: # if no frame received
        print("Can't receive frame. Exiting...") # inform user
        break # break from unconditional loop and exit program
    frame = imutils.resize(frame, height=tgt_height) # resize received frame to target height

    k = cv.waitKey(1) & 0xFF # wait 1 ms for keypress
    if k == 27: # if 'esc' received
        break # break from unconditional loop and exit program
    elif k == ord('`'):
        mode = 0 # change to original video mode (no processing)
    elif k == ord('1'):
        mode = 1 # change to threshold mode
    elif k == ord('2'):
        mode = 2 # change to edge detection mode
    elif k == ord('3'):
        mode = 3 # change to corner detection mode
    elif k == ord('4'):
        mode = 4 # change to line detection mode
    elif k == ord('5'):
        mode = 5 # change to frame differencing mode
    else:
        pass # ignore all other keystrokes
    print("\rMode {} selected".format(mode), end="") # inform user of currently selected mode

    disp_frame = [] # initialize display frame (actual frame passed to user viewer)
    if mode == 0:
        # original
        disp_frame = frame # pass received frame to user viewer
    elif mode == 1:
        # binarize/threshold
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) # convert frame to grayscale
        disp_frame = cv.threshold(gray, 125, 250, cv.THRESH_BINARY_INV)[1] # convert grayscale to black and white
    elif mode == 2:
        # canny edge detection
        disp_frame = cv.Canny(frame, 30, 150) # convert frame to detected edges
    elif mode == 3:
        # corner detection
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) # convert frame to grayscale
        gray = np.float32(gray) # convert frame array to float32 for computation
        dst = cv.cornerHarris(gray, 2, 3, 0.04) # detect corners
        dst = cv.dilate(dst, None) # dilate detected corners
        ret, dst = cv.threshold(dst, 0.01 * dst.max(), 255, 0) # convert corner data to black and white
        dst = np.uint8(dst) # convert corner data to unsigned 8b ints for computation

        ret, labels, stats, centroids = cv.connectedComponentsWithStats(dst) # don't know :|

        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001) # create object of 'criteria' data
        corners = cv.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria) # find sub pixel corners

        res = np.hstack((centroids, corners)) # don't know :/
        res = np.int0(res) # don't know :(
        disp_frame = frame # pass received frame to user viewer frame
        if res is not None: # if there are corners detected
            for point in res[1:len(res)]: # work through each point
                cv.circle(disp_frame, (point[0], point[1]), 2, (0, 0, 255), 2) # draw a circle at the given point
    elif mode == 4:
        # line detection
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) # convert frame to grayscale
        edges = cv.Canny(gray, 50, 150, apertureSize=3) # detect edges
        lines = cv.HoughLines(edges, 1, np.pi / 180, 200)n # detect lines from edges data

        disp_frame = frame # pass received frame to user viewer frame
        if lines is not None: # if there are lines detected
            for oneLine in lines: # work through each line
                for r, theta in oneLine: # work through each radius and angle for each line
                    # calculate two x and y coordinates for each line
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a * r
                    y0 = b * r
                    x1 = int(x0 + 1000 * (-b))
                    y1 = int(y0 + 1000 * (a))
                    x2 = int(x0 - 1000 * (-b))
                    y2 = int(y0 - 1000 * (a))
                    # draw a line corresponding to each set of coordinates
                    cv.line(disp_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    elif mode == 5:
        # difference mode
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) # convert frame data to grayscale
        ret, frame2 = cap.read() # receive another frame from camera
        frame2 = imutils.resize(frame2, height=tgt_height) # resize second frame
        frame2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY) # convert second frame to grayscale
        disp_frame = cv.absdiff(frame2, frame) # use both frames to find their absolute difference
    else:
        mode = 0 # invalid mode encountered - change back to mode 0 (original capture)

    cv.imshow("monk-hw1_p2", disp_frame) # display processed frame
    disp_frame = None # empty frame data

# while loop broken
cap.release() # release camera
cv.destroyAllWindows() # close all windows generated by opencv