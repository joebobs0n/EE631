import imutils
import cv2 as cv
import numpy as np

tgt_fps = 60
tgt_height = 600

print("\nUsage Instructions:\n -ESC: exit feed\n -0: original mode\n -1: binarize mode\n -2: canny edges mode\n -3: "
      "corner detect mode\n -4: line detect mode\n -5: difference mode\n")

mode = 0
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
first_flag = True
cap.set(cv.CAP_PROP_FPS, tgt_fps)
print("Starting OpenCV program at {} FPS".format(cap.get(cv.CAP_PROP_FPS)))

while True:
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame. Exiting...")
        break

    frame = imutils.resize(frame, height=tgt_height)

    k = cv.waitKey(1) & 0xFF
    if k == 27:
        break
    elif k == ord('`'):
        mode = 0
    elif k == ord('1'):
        mode = 1
    elif k == ord('2'):
        mode = 2
    elif k == ord('3'):
        mode = 3
    elif k == ord('4'):
        mode = 4
    elif k == ord('5'):
        mode = 5
    else:
        pass

    print("\rMode {} selected".format(mode), end="")

    disp_frame = []
    if mode == 0:
        # original
        disp_frame = frame
    elif mode == 1:
        # binarize/threshold
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        disp_frame = cv.threshold(gray, 125, 250, cv.THRESH_BINARY_INV)[1]
    elif mode == 2:
        # canny edge detection
        disp_frame = cv.Canny(frame, 30, 150)
    elif mode == 3:
        # corner detection
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gray = np.float32(gray)
        dst = cv.cornerHarris(gray, 2, 3, 0.04)
        dst = cv.dilate(dst, None)
        ret, dst = cv.threshold(dst, 0.01 * dst.max(), 255, 0)
        dst = np.uint8(dst)

        ret, labels, stats, centroids = cv.connectedComponentsWithStats(dst)

        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        corners = cv.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria)

        res = np.hstack((centroids, corners))
        res = np.int0(res)
        disp_frame = frame
        # disp_frame[res[:, 1], res[:, 0]] = [0, 0, 255]
        if res is not None:
            for point in res[1:len(res)]:
                cv.circle(disp_frame, (point[0], point[1]), 2, (0, 0, 255), 2)
    elif mode == 4:
        # line detection
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        edges = cv.Canny(gray, 50, 150, apertureSize=3)
        lines = cv.HoughLines(edges, 1, np.pi / 180, 200)

        disp_frame = frame
        if lines is not None:
            for oneLine in lines:
                for r, theta in oneLine:
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a * r
                    y0 = b * r
                    x1 = int(x0 + 1000 * (-b))
                    y1 = int(y0 + 1000 * (a))
                    x2 = int(x0 - 1000 * (-b))
                    y2 = int(y0 - 1000 * (a))
                    cv.line(disp_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    elif mode == 5:
        # difference mode
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        ret, frame2 = cap.read()
        frame2 = imutils.resize(frame2, height=tgt_height)
        frame2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
        disp_frame = cv.absdiff(frame2, frame)
    else:
        pass

    cv.imshow("monk-hw1_p2", disp_frame)
    disp_frame = None

print("")
cap.release()
cv.destroyAllWindows()