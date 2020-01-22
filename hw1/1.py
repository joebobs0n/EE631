import cv2 as cv
import numpy as np

target_width = 800
target_height = 600

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

if not (target_width == cap.get(cv.CAP_PROP_FRAME_WIDTH)):
    cap.set(cv.CAP_PROP_FRAME_WIDTH, target_width)
if not (target_height == cap.get(cv.CAP_PROP_FRAME_HEIGHT)):
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, target_height)

if not (target_width == cap.get(cv.CAP_PROP_FRAME_WIDTH)):
    print("Width wrong and unset")
if not (target_height == cap.get(cv.CAP_PROP_FRAME_HEIGHT)):
    print("Height wrong and unset")

ret, frame = cap.read()
if not ret:
    print("Can't receive frame (stream end?). Exiting...")
    exit()
# gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
cv.imshow('Monk', frame)
cv.imwrite('1.jpg', frame)

cv.waitKey(0)
cap.release()
cv.destroyAllWindows()
exit()
