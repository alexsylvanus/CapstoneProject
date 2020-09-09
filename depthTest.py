import cv2
import numpy as np
import sys
from matplotlib import pyplot as plt
import time

left = cv2.VideoCapture(1)
right = cv2.VideoCapture(2)

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

#left.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
#left.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
#right.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
#right.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

LEFT_PATH = "capture/left/{:06d}.jpg"
RIGHT_PATH = "capture/right/{:06d}.jpg"

# Filenames are just an increasing number
frameId = 0

calibration = np.load(sys.argv[1], allow_pickle=False)
imageSize = tuple(calibration["imageSize"])
leftMapX = calibration["leftMapX"]
leftMapY = calibration["leftMapY"]
leftROI = tuple(calibration["leftROI"])
rightMapX = calibration["rightMapX"]
rightMapY = calibration["rightMapY"]
rightROI = tuple(calibration["rightROI"])
mtx = calibration["rightMat"]
dist = calibration["rightCo"]

while(True):

	_, leftFrame = left.read()
	_, rightFrame = right.read()

	#right
	h,w = rightFrame.shape[:2]
	camMat, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
	dst = cv2.undistort(rightFrame, mtx, dist, None, camMat)
	x,y,w,h = roi
	#print type(dst)	
	rightF = dst[y:y+h, x:x+w]

	#left
	h,w = leftFrame.shape[:2]
	camMat, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
	dst = cv2.undistort(leftFrame, mtx, dist, None, camMat)
	x,y,w,h = roi
	#print type(dst)	
	leftF = dst[y:y+h, x:x+w] 
	 
	#fixedLeft = cv2.remap(leftFrame, leftMapX, leftMapY,cv2.INTER_LINEAR)
	#fixedRight = cv2.remap(rightFrame, rightMapX, rightMapY,cv2.INTER_LINEAR)
	#cv2.imshow('left', leftF)
	#cv2.imshow('right', rightF)

	#gleftFrame = cv2.cvtColor(leftFrame, cv2.COLOR_BGR2GRAY)
	#grightFrame = cv2.cvtColor(rightFrame, cv2.COLOR_BGR2GRAY)
	#stereo = cv2.StereoBM_create(numDisparities=128, blockSize=5)
	#disparity = stereo.compute(gleftFrame,grightFrame)
	#cv2.imshow('disparity', disparity)

	
	stereoMatcher = cv2.StereoBM_create(numDisparities=64, blockSize=27)
#	stereoMatcher = cv2.StereoSGBM_create(8,48,27)
#	stereoMatcher.setNumDisparities(16)
#	stereoMatcher.setBlockSize(15)
#	stereoMatcher.setSpeckleRange(32)
#	stereoMatcher.setSpeckleWindowSize(50)
	#stereoMatcher.setROI1(leftROI)
	#stereoMatcher.setROI2(rightROI)


	#fixedLeft = cv2.remap(leftFrame, leftMapX, leftMapY,cv2.INTER_LINEAR)
	#fixedRight = cv2.remap(rightFrame, rightMapX, rightMapY,cv2.INTER_LINEAR)

	grayLeft = cv2.cvtColor(leftF, cv2.COLOR_BGR2GRAY)
	grayRight = cv2.cvtColor(rightF, cv2.COLOR_BGR2GRAY)
	grayLeft = cv2.resize(grayLeft,(500,300))
	grayRight = cv2.resize(grayRight,(500,300))
	depth = stereoMatcher.compute(grayRight,grayLeft)
	#res = cv2.convertScaleAbs(depth)
	minD = depth.min()
	maxD = depth.max()
	disp = np.uint8(255 * (depth - minD) / (maxD - minD))
	DEPTH_VISUALIZATION_SCALE = 2048
	#plt.imshow(depth, 'gray')
	cv2.imshow('depth', np.hstack((grayRight, grayLeft, disp)))
	#plt.show(False)
	#plt.pause(0.75)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
	if cv2.waitKey(1) & 0xFF == ord('t'):
		cv2.imwrite(LEFT_PATH.format(frameId), depth)
		cv2.imwrite(RIGHT_PATH.format(frameId), depth)
		frameId+=1

	#time.sleep(.001)

left.release()
right.release()
cv2.destroyAllWindows()
