import cv2 as cv
import numpy as np
import imutils

print("Hit ESC to exit feed")

greenLower = (29,86,6)
greenUpper = (64,255,255)

cap = cv.VideoCapture(0)
if not cap.isOpened():
	print("Cannot open camera")
	exit()
cap.set(cv.CAP_PROP_FPS,15)

while True:
	ret, frame = cap.read()
	if not ret:
		print("Can't receive frame (stream end?). Exiting...")
		break
	if cv.waitKey(1) == (27 & 0xFF):
		break
		
	frame = imutils.resize(frame,width=600)
	disp_frame = frame
	blurred = cv.medianBlur(frame,5)
	gray = cv.cvtColor(blurred,cv.COLOR_BGR2GRAY)
	
	circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1.2, 100)
	
	if circles is not None:
		for (x,y,r) in circles:
			cv.circle(disp_frame,(x,y),r,(0,0,255),2)

	cv.imshow('monk-hw1_p3',disp_frame)

cap.release()
cv.destroyAllWindows()
