import cv2 as cv
import glob

images = glob.glob('calibration_imgs/*.jpg')  # create array of all image filenames
width = 10  # width of chessboard corners
height = 7  # height of chessboard corners
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)  # criteria for subpixel detection
saved_flag = False  # check if one image has been saved (for documentation)

for img in images:  # work through each image in calibration pictures array
    frame = cv.imread(img)  # pull image
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)  # convert to grayscale
    corners = cv.findChessboardCorners(gray, (width, height), None)[1]  # detect chessboard corners

    if corners is not None:  # check if there are corners
        subpix = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)  # refine corner detection
        disp = cv.drawChessboardCorners(frame, (width, height), subpix, True)  # draw corners on frame

        cv.imshow('chessboard corner detection', disp)  # show frame

    if not saved_flag:  # check if an image has already been saved
        cv.imwrite('task1.jpg', disp)  # save image
        saved_flag = True  # set image saved flag

    cv.waitKey(100)  # wait for any key to be pressed or 100 ms to elapse

cv.destroyAllWindows()  # destroy all windows generated by script
exit()  # close out script
