import cv2 as cv
import numpy as np

file = np.load('params.npz')  # read in parameters from task 2 calibration

fl = file['fl']  # read in focal length
intr = file['intr']  # read in intrinsic parameters
dist = file['dist']  # read in distortion parameters

frame = cv.imread('obj_w_corners/Object with Corners.jpg')  # read in image
f = open('obj_w_corners/Data Points.txt')  # open file stream for data

image_points = []  # create vector for image points data
world_points = []  # create vector for world points data

for line in f:  # work through each line
    data = [float(x) for x in line.split()]  # convert each line into array of elements
    if len(data) is 2:  # if line has two elements
        image_points.append(data)  # append it to image points
    elif len(data) is 3:  # if line has three elements
        world_points.append(data)  # append it to world points
    else:  # line doesn't have two or three elements
        print('Error in data import.')  # inform user of illformed data
        exit()  # exit program

image_points = np.asarray(image_points, np.float64)  # convert vector to array of float64
world_points = np.asarray(world_points, np.float64)  # convert vector to array of float64

rvec, tvec = cv.solvePnP(world_points, image_points, intr, dist)[1:3]
rmat = cv.Rodrigues(rvec)[0]

print(f'rotation vector:\n{rvec}\n\nrotation matrix:\n{rmat}\n')
print(f'translation vector:\n{tvec}')

f = open('task4.txt', 'w')  # open file for writing results
f.write(f'rotation vector:\n{rvec}\n\nrotation matrix:\n{rmat}\n\n')  # rotation vector and matrix
f.write(f'translation vector:\n{tvec}')  # store translation vector
f.close()  # close file

np.savez('task4.npz', rvec=rvec, rmat=rmat, tvec=tvec)

exit()  # close out script
