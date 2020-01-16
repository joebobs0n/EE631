# convert pictures to avi
import cv2 as cv

tgt_width = 640
tgt_height = 480
fps = 60
leftVid = cv.VideoWriter('ball_left_vid.avi', cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (tgt_width, tgt_height))
rightVid = cv.VideoWriter('ball_right_vid.avi', cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (tgt_width, tgt_height))

for i in range(46):
    leftFrame = ''
    rightFrame = ''
    if i < 5:
        leftFrame = ("Baseball Practice Images/1L0{}.jpg".format(i + 5))
        rightFrame = ("Baseball Practice Images/1R0{}.jpg".format(i + 5))
    else:
        leftFrame = ("Baseball Practice Images/1L{}.jpg".format(i + 5))
        rightFrame = ("Baseball Practice Images/1R{}.jpg".format(i + 5))

    leftFrame = cv.imread(leftFrame)
    rightFrame = cv.imread(rightFrame)

    leftVid.write(leftFrame)
    rightVid.write(rightFrame)

leftVid.release()
rightVid.release()
exit()
