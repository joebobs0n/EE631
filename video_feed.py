import cv2 as cv
import numpy as np

print("Hit ESC to exit feed")

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting...")
        break
    cv.imshow('Monk', frame)
    if cv.waitKey(1) == 27:
        break

cap.release()
cv.destroyAllWindows()
exit()
