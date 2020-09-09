import cv2
import numpy as np
import sys
from matplotlib import pyplot as plt

left = cv2.VideoCapture(1)
right = cv2.VideoCapture(2)

def cut(disparity, image, threshold):
 for i in range(0, image.height):
  for j in range(0, image.width):
   # keep closer object
   if cv.GetReal2D(disparity,i,j) > threshold:
    cv.Set2D(disparity,i,j,cv.Get2D(image,i,j))

while(True):

	_, leftFrame = left.read()
	_, rightFrame = right.read()

	grayleft = cv2.cvtColor(leftFrame, cv2.COLOR_BGR2GRAY)
	grayright = cv2.cvtColor(rightFrame, cv2.COLOR_BGR2GRAY)

	stereo = cv2.StereoSGBM_create(minDisparity=5, numDisparities=16, blockSize = 5)
	disparity = stereo.compute(grayleft,grayright)
	cv2.imshow('right', disparity)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

left.release()
right.release()
cv2.destroyAllWindows()
