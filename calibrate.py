import argparse
import os
import sys
import cv2 as cv
import numpy as np
import imutils

os.chdir(sys.path[0])
ap = argparse.ArgumentParser()
ap.add_argument('-f', '--frames', help='Number of frames for each view (left, right, and stereo) to use for calibration. Default = 32')
ap.add_argument('-l', '--left', help='Camera index of left camera. Default = 0')
ap.add_argument('-r', '--right', help='Camera index of right camera. Default = 1')
ap.add_argument('-p', '--pixel', help='Width of sensor\'s individual pixel in millimeters. Default = 7.4e-3')
ap.add_argument('-v', '--verification', help='Turns on and off verification viewer after calibration. 0: off, 1: on. Default = 0')
ap.add_argument('--cbwidth', help='Width of calibration chessboard in intersections. Default = 9')
ap.add_argument('--cbheight', help='Height of calibration chessboard in intersections. Default = 7')
ap.add_argument('--gridwidth', help='Width of chessboard grid (from intersection to adjacent intersection) in whatever units you want measurements in. Default = 1')
ap.add_argument('--rotleft', help='Rotate left camera view. Integer value from 0-3. 0: 0 deg, 1: 90 deg, 2: 180 deg, 3: 270 deg. Default = 0')
ap.add_argument('--rotright', help='Rotate right camera view. Integer value from 0-3. 0: 0 deg, 1: 90 deg, 2: 180 deg, 3: 270 deg. Default = 0')
args = vars(ap.parse_args())

calib_max_images = 32
left_camera_index = 0
right_camera_index = 1
cb_width = 9
cb_height = 7
grid_width = 1
pixel = 7.4e-3
rot_left = 0
rot_right = 0
verify = False

if args.get('frames'):
	calib_max_images = int(args['frames'])
if args.get('left'):
	left_camera_index = int(args['left'])
if args.get('right'):
	right_camera_index = int(args['right'])
if args.get('cbwidth'):
	cb_width = int(args['cbwidth'])
if args.get('cbheight'):
	cb_height = int(args['cbheight'])
if args.get('pixel'):
	pixel = float(args['pixel'])
if args.get('rotleft'):
	rot_left = int(args['rotleft'])
if args.get('rotright'):
	rot_right = int(args['rotright'])
if args.get('gridwidth'):
    grid_width = float(args['gridwidth'])
if args.get('verification'):
    verify = bool(args['verification'])

print('Starting stereo calibration script.\n')
print(f'Frames per calibration step: {calib_max_images}')
print(f'Left camera index: {left_camera_index}')
print(f'Right camera index: {right_camera_index}')
print(f'Left camera rotate: {rot_left * 90} deg')
print(f'Right camera rotate: {rot_right * 90} deg')
print(f'Chessboard dimensions: ({cb_width}, {cb_height})')
print(f'Chessboard grid width: {grid_width}')
print(f'Sensor pixel size: {pixel} mm')
print(f'User verification after calibration: {"True" if verify else "False"}\n')

print(f'Usage:\n  Space bar: capture image\n  ESC: exit script\n')

print('Initializing left camera...')
cam_l = cv.VideoCapture(left_camera_index)
print('Initializing right camera...\n')
cam_r = cv.VideoCapture(right_camera_index)

chessboard = [cb_width, cb_height]
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
board_points = np.zeros((chessboard[0] * chessboard[1], 3), np.float32)
board_points[:, :2] = np.mgrid[0:chessboard[0], 0:chessboard[1]].T.reshape(-1, 2) * grid_width
image_count = 0

frame_shape = None

for side in ['left', 'right']:
	world_points = []
	image_points = []

	print(f'Starting calibration of {side} side...')
	while image_count < calib_max_images:
		frame = cam_l.read()[1] if side == 'left' else cam_r.read()[1]
		frame = imutils.rotate_bound(frame, 90 * (rot_left if side == 'left' else rot_right))
		cv.imshow(f'Mono Selection Frame: {side}', frame)

		k = cv.waitKey(1) & 0xff
		if k == 27:
			cv.destroyAllWindows()
			exit()
		elif k == ord(' '):
			if image_count == 0 and side == 'left':
				frame_shape = tuple(np.flip(np.shape(frame)[0:2]))
   
			gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
			corners = cv.findChessboardCorners(gray, (chessboard[0], chessboard[1]), None)[1]

			if corners is None:
				print('  No chessboard detected. Try again.')
			else:
				subpix = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
				image_points.append(subpix)
				world_points.append(board_points)
				image_count += 1
				print(f'  {side}: {image_count}/{calib_max_images}')
				disp_frame = cv.drawChessboardCorners(frame, (chessboard[0], chessboard[1]), subpix, True)
				cv.imshow(f'Calibration Frame: {side}', disp_frame)

	print(f'  Calculating intrinsic and distortion parameters of {side} camera.\n')
	ret, mat, dist = cv.calibrateCamera(world_points, image_points, frame_shape, None, None)[0:3]
	dist = dist.T

	fSx = mat[0, 0]
	focal_length = fSx * pixel if pixel > 0 else -1

	f = open(side + '_params.txt', 'w')
	f.write(f'focal length in pixels:\n{fSx}\n\n')
	f.write(f'focal length in mm:\n{focal_length}\n\n')
	f.write(f'intrinsic paramters:\n{mat}\n\n')
	f.write(f'distortion paramters:\n{dist}')
	f.close()
	np.savez(side + '_params.npz', flpx=fSx, flmm=focal_length, mat=mat, dist=dist)

	cv.destroyAllWindows()
	image_count = 0

world_points_stereo = []
image_points_left = []
image_points_right = []

print('Starting calibration and rectification of stereo configuration.')
while image_count < calib_max_images:
	frame_l = cam_l.read()[1]
	frame_r = cam_r.read()[1]
	frame_l = imutils.rotate_bound(frame_l, 90 * rot_left)
	frame_r = imutils.rotate_bound(frame_r, 90 * rot_right)

	full_frame = np.hstack([frame_l, frame_r])
	cv.imshow('Stereo Selection Frame', full_frame)

	k = cv.waitKey(1) & 0xff
	if k == ord(' '):
		gray_l = cv.cvtColor(frame_l, cv.COLOR_BGR2GRAY)
		gray_r = cv.cvtColor(frame_r, cv.COLOR_BGR2GRAY)

		corners_l = cv.findChessboardCorners(gray_l, (chessboard[0], chessboard[1]), None)[1]
		corners_r = cv.findChessboardCorners(gray_r, (chessboard[0], chessboard[1]), None)[1]

		if (corners_l is not None) and (corners_r is not None):
			subpix_l = cv.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), criteria)
			subpix_r = cv.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), criteria)

			image_points_left.append(subpix_l)
			image_points_right.append(subpix_r)
			world_points_stereo.append(board_points)

			disp_l = cv.drawChessboardCorners(frame_l, (chessboard[0], chessboard[1]), subpix_l, True)
			disp_r = cv.drawChessboardCorners(frame_r, (chessboard[0], chessboard[1]), subpix_r, True)
			disp_frame = np.hstack([disp_l, disp_r])
			cv.imshow('Calibration Frame', disp_frame)

			image_count += 1
			print(f'  stereo: {image_count}/{calib_max_images}')
		else:
			print('  No chessboard corners found in stereo view. Try again.')
	elif k == 27:
		cv.destroyAllWindows()
		exit()

cv.destroyAllWindows()

print(f'  Calculating stereo calibration parameters.')
left = np.load('left_params.npz')
mat_l = left['mat']
dist_l = left['dist']
right = np.load('right_params.npz')
mat_r = right['mat']
dist_r = right['dist']
(_, _, _, _, _, R, T, E, F) = cv.stereoCalibrate(world_points_stereo, image_points_left, image_points_right, mat_l, dist_l, mat_r, dist_r, frame_shape, criteria=criteria, flags=cv.CALIB_FIX_INTRINSIC)
f = open('stereo_params.txt', 'w')
f.write(f'R:\n{R}\n\n')
f.write(f'T:\n{T}\n\n')
f.write(f'E:\n{E}\n\n')
f.write(f'F:\n{F}')
f.close()
np.savez('stereo_params.npz', R=R, T=T, E=E, F=F)

print(f'  Calculating stereo rectification paramters.\n')
R1, R2, P1, P2, Q = cv.stereoRectify(mat_l, dist_l, mat_r, dist_r, frame_shape, R, T)[0:5]
f = open('rectify_params.txt', 'w')
f.write(f'R1:\n{R1}\n\n')
f.write(f'R2:\n{R2}\n\n')
f.write(f'P1:\n{P1}\n\n')
f.write(f'P2:\n{P2}\n\n')
f.write(f'Q:\n{Q}\n\n')
f.close()
np.savez('rectify_params.npz', R1=R1, R2=R2, P1=P1, P2=P2, Q=Q)

print('Full calibration complete.')

while verify:
	ret_l, frame_l = cam_l.read()
	ret_r, frame_r = cam_r.read()
	frame_l = imutils.rotate_bound(frame_l, 90 * rot_left)
	frame_r = imutils.rotate_bound(frame_r, 90 * rot_right)

	k = cv.waitKey(1) & 0xff
	if k == 27:
		break

	if ret_l is False or ret_r is False:
		break

	gray_l = cv.cvtColor(frame_l, cv.COLOR_BGR2GRAY)
	gray_r = cv.cvtColor(frame_r, cv.COLOR_BGR2GRAY)

	ret_l, corners_l = cv.findChessboardCorners(gray_l, (chessboard[0], chessboard[1]), None)
	ret_r, corners_r = cv.findChessboardCorners(gray_r, (chessboard[0], chessboard[1]), None)

	if ret_l is True and ret_r is True:
		point_l = np.array([corners_l[0]])
		point_r = np.array([corners_r[0]])
  
		dist_points_l = cv.undistortPoints(point_l, mat_l, dist_l, R=R1, P=P1)
		dist_points_r = cv.undistortPoints(point_r, mat_r, dist_r, R=R2, P=P2)

		disparity = np.array([[dist_points_l[0][0][0] - dist_points_r[0][0][0]]])
  
		dist_points_l = np.array(dist_points_l).reshape((1, 2))
		dist_points_l = np.hstack([dist_points_l, disparity]).reshape((1, 1, 3))
		dist_points_r = np.array(dist_points_r).reshape((1, 2))
		dist_points_r = np.hstack([dist_points_r, disparity]).reshape((1, 1, 3))

		obj_dist_l = cv.perspectiveTransform(dist_points_l, Q)
		obj_dist_r = cv.perspectiveTransform(dist_points_r, Q)

		frame_l = cv.putText(frame_l, f'({int(obj_dist_l[0][0][0])}, {int(obj_dist_l[0][0][1])}, {int(obj_dist_l[0][0][2])})', (int(point_l[0][0][0]), int(point_l[0][0][1])), cv.QT_FONT_NORMAL, 0.5, (0, 0, 255), 1, cv.LINE_AA)
		frame_r = cv.putText(frame_r, f'({int(obj_dist_r[0][0][0])}, {int(obj_dist_r[0][0][1])}, {int(obj_dist_r[0][0][2])})', (int(point_r[0][0][0]), int(point_r[0][0][1])), cv.QT_FONT_NORMAL, 0.5, (0, 0, 255), 1, cv.LINE_AA)

	disp_frame = np.hstack([frame_l, frame_r])
	cv.imshow('Verification', disp_frame)

cam_l.release()
cam_r.release()
cv.destroyAllWindows()
exit()
