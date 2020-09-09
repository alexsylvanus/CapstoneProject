"""
copyright (c) 2015 PAL Robotics SL.
Released under the BSD License.

Created on 7/14/15

@author: Sammy Pfeiffer

test_video_resource.py contains
a testing code to see if opencv can open a video stream
useful to debug if video_stream does not work
"""

import cv2
import sys
from time import sleep

if __name__ == '__main__':

	# If we are given just a number, interpret it as a video device

	cap1 = cv2.VideoCapture(1)
	cap2 = cv2.VideoCapture(2)

	rval1, frame1 = cap1.read()
	rval2, frame2 = cap2.read()

	
	
	while rval1:
		cv2.imshow('left', frame1)
		cv2.imshow('right', frame2)
		rval1, frame1 = cap1.read()
		rval2, frame2 = cap2.read()
		key = cv2.waitKey(20)
        # print "key pressed: " + str(key)
        # exit on ESC, you may want to uncomment the print to know which key is ESC for you
		if key == 27 or key == 1048603:
			break
		cv2.destroyWindow("preview")

