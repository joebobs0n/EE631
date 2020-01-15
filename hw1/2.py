import imutils  # import library for simple frame transforms
import cv2 as cv  # import openCV
import numpy as np  # import numpy for math capabilities

tgt_width = 800 # change width of processed/displayed video
tgt_height = 600  # change height of processed/displayed video

currentMode = 'Unmodified'  # text to print
mode = 0  # initialize current mode variable
disp_frame = []  # initialize display frame (actual frame passed to user viewer)
prev_frame = []  # initialize previous display frame
captureFlag = False  # initialize program to not intially capture video
needPrevFlag = False  # flag to determine if previous flag is needed (for difference mode)
cap = cv.VideoCapture(0)  # initialize video capture
if not cap.isOpened():  # check if camera successfully opened
    print("Cannot open camera")  # if not - inform user
    exit()  # exit program
fps = cap.get(cv.CAP_PROP_FPS)  # frame rate of camera
# configure video out writer at capture's height, width, and fps
out_vid = cv.VideoWriter('2.avi', cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (tgt_width, tgt_height))

while True:  # run continuously
    ret, frame = cap.read()  # read frame from camera capture
    if not ret:  # if no frame received
        print("Can't receive frame. Exiting...")  # inform user
        break  # break from unconditional loop and exit program
    frame = imutils.resize(frame, height=tgt_height, width=tgt_width)  # resize received frame

    # check key for key press and change operating mode accordingly
    k = cv.waitKey(1) & 0xFF  # wait 1 ms for keypress
    if k == 27:  # if 'esc' received
        break  # break from unconditional loop and exit program
    elif k == ord('`'):
        currentMode = 'Unmodified'  # set text to be print on frame
        mode = 0  # change to original video mode (no processing)
        needPrevFlag = False  # reset previous frame needed flag
    elif k == ord('1'):
        currentMode = 'Binarized/Threshold'  # set text to be print on frame
        mode = 1  # change to threshold mode
        needPrevFlag = False  # reset previous frame needed flag
    elif k == ord('2'):
        currentMode = 'Canny/Edge Detection'  # set text to be print on frame
        mode = 2  # change to edge detection mode
        needPrevFlag = False  # reset previous frame needed flag
    elif k == ord('3'):
        currentMode = 'Corner Detection'  # set text to be print on frame
        mode = 3  # change to corner detection mode
        needPrevFlag = False  # reset previous frame needed flag
    elif k == ord('4'):
        currentMode = 'Line Detection'  # set text to be print on frame
        mode = 4  # change to line detection mode
        needPrevFlag = False  # reset previous frame needed flag
    elif k == ord('5'):
        currentMode = 'Differencing'  # set text to be print on frame
        mode = 5  # change to frame differencing mode
    elif k == ord('c'):
        captureFlag = not captureFlag  # toggle capture flag
    else:
        pass  # ignore all other keystrokes

    # perform appropriate actions based on flags and current mode
    if mode == 0:
        # original
        disp_frame = frame  # pass received frame to user viewer
    elif mode == 1:
        # binarize/threshold
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)  # convert frame to grayscale
        disp_frame = cv.threshold(gray, 125, 250, cv.THRESH_BINARY_INV)[1]  # convert grayscale to black and white
        disp_frame = cv.cvtColor(disp_frame, cv.COLOR_GRAY2BGR)  # convert back to "BGR" (still looks grayscale)
    elif mode == 2:
        # canny edge detection
        disp_frame = cv.Canny(frame, 30, 150)  # convert frame to detected edges
        disp_frame = cv.cvtColor(disp_frame, cv.COLOR_GRAY2BGR)  # convert back to "BGR" (still looks grayscale)
    elif mode == 3:
        # corner detection
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)  # convert frame to grayscale
        gray = np.float32(gray)  # convert frame array to float32 for computation
        dst = cv.cornerHarris(gray, 2, 3, 0.04)  # detect corners
        dst = cv.dilate(dst, None)  # dilate detected corners
        dst = cv.threshold(dst, 0.01 * dst.max(), 255, 0)[1]  # convert corner data to black and white
        dst = np.uint8(dst)  # convert corner data to unsigned 8b ints for computation

        _, labels, stats, centroids = cv.connectedComponentsWithStats(dst)  # don't know :|

        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)  # create object of 'criteria' data
        corners = cv.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria)  # find sub pixel corners

        res = np.hstack((centroids, corners))  # flip the two arrays to be horizontally stacked
        res = np.int0(res)  # convert corner data to ints for computation
        disp_frame = frame  # pass received frame to user viewer frame
        if res is not None:  # if there are corners detected
            for point in res[1:len(res)]:  # work through each point
                cv.circle(disp_frame, (point[0], point[1]), 2, (0, 0, 255), 2)  # draw a circle at the given point
    elif mode == 4:
        # line detection
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)  # convert frame to grayscale
        edges = cv.Canny(gray, 50, 150, apertureSize=3)  # detect edges
        lines = cv.HoughLines(edges, 1, np.pi / 180, 200)  # detect lines from edges data

        disp_frame = frame  # pass received frame to user viewer frame
        if lines is not None:  # if there are lines detected
            for oneLine in lines:  # work through each line
                for r, theta in oneLine:  # work through each radius and angle for each line
                    # calculate two x and y coordinates for each line
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a * r
                    y0 = b * r
                    x1 = int(x0 - 1000 * b)
                    y1 = int(y0 + 1000 * a)
                    x2 = int(x0 + 1000 * b)
                    y2 = int(y0 - 1000 * a)
                    # draw a line corresponding to each set of coordinates
                    cv.line(disp_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    elif mode == 5:
        # difference mode
        # check if the need previous flag is asserted (essentially check if "first frame" of differencing)
        if needPrevFlag:
            # not first frame of differencing
            disp_frame = cv.absdiff(prev_frame, frame)  # use both frames to find their absolute difference
        else:
            disp_frame = frame  # is first frame of differencing

        if not needPrevFlag:
            needPrevFlag = True  # assert need previous flag since we want to calculate difference frames on next frame

        prev_frame = frame  # store current frame as previous frame for next difference frame
    else:
        mode = 0  # invalid mode encountered - change back to mode 0 (original capture)

    if captureFlag:  # check if currently capturing
        # put text on frame to indicate that video currently capturing
        cv.putText(disp_frame, '{}'.format(currentMode), (15, 30), cv.QT_FONT_NORMAL, .6, (50, 50, 255), 1, cv.LINE_AA)
        out_vid.write(disp_frame)  # write current display frame to video output
    else:
        # put text on frame to indicate that video currently capturing
        cv.putText(disp_frame, '{}'.format(currentMode), (15, 30), cv.QT_FONT_NORMAL, .6, (255, 50, 50), 1, cv.LINE_AA)
    cv.imshow("monk-hw1_p2", disp_frame)  # display processed frame
    disp_frame = None  # empty frame data

# while loop broken
cap.release()  # release camera
out_vid.release()  # release video out
cv.destroyAllWindows()  # close all windows generated by opencv
