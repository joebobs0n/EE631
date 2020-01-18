# convert pictures to avi
import cv2 as cv  # import opencv for video capture/processing
import imutils  # import imutils for frame manipulation
import numpy as np  # import numpy for numerical calculations

numFrames = 36  # indicate number of frames for the video
tgt_width = 640  # target width of video
tgt_height = 480  # target height of video
fps = 60  # target frames per second
# configure left video output file
leftVid = cv.VideoWriter('ball_left_vid.avi', cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (tgt_width, tgt_height))
# configure right video output file
rightVid = cv.VideoWriter('ball_right_vid.avi', cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (tgt_width, tgt_height))
# configure video output file for both right and left
fullVid = cv.VideoWriter('ball_both_vid.avi', cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (2*tgt_width, tgt_height))

for i in range(numFrames):  # work through all frames
    leftFrame = ''  # initialize left frame variable
    rightFrame = ''  # initialize right frame variable

    # generate file names to pull from picture path
    if i < 5:
        leftFrame = ("Baseball Practice Images/1L0{}.jpg".format(i + 5))
        rightFrame = ("Baseball Practice Images/1R0{}.jpg".format(i + 5))
    else:
        leftFrame = ("Baseball Practice Images/1L{}.jpg".format(i + 5))
        rightFrame = ("Baseball Practice Images/1R{}.jpg".format(i + 5))

    leftFrame = cv.imread(leftFrame)  # read left frame data
    rightFrame = cv.imread(rightFrame)  # read right frame data

    leftFrame = imutils.resize(leftFrame, height=tgt_height, width=tgt_width)  # resize frame to target dimensions
    rightFrame = imutils.resize(rightFrame, height=tgt_height, width=tgt_width)  # resize frame to target dimensions

    fullFrame = np.hstack((leftFrame, rightFrame))  # place right and left frames next to each other

    leftVid.write(leftFrame)  # write left frame to its video file
    rightVid.write(rightFrame)  # write right frame to its video file
    fullVid.write(fullFrame)  # write full frame to its video file

leftVid.release()  # release left video output stream
rightVid.release()  # release right video output stream
fullVid.release()  # release full video output stream
exit()  # exit program
