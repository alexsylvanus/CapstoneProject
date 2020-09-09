import numpy as np
import cv2

capR = cv2.VideoCapture(1)
capL = cv2.VideoCapture(2)
orbR = cv2.ORB()
orbL = cv2.ORB()

kpR_L = []
kpR_R = []
kpL_L = []
kpL_R = []

while True:
    kpR_L = []
    kpR_R = []
    kpL_L = []
    kpL_R = []

    # Capture frame-by-frame
    retR, frameR = capR.read()
    retL, frameL = capL.read()
    # Our operations on the frame come here
    # find the keypoints with ORB
    orbR = cv2.ORB_create(nfeatures = 500)  # Initiate SIFT detector
    orbL = cv2.ORB_create(nfeatures = 500)

    # find the keypoints and descriptors with SIFT
    kpR = orbR.detect(frameR)
    kpL = orbL.detect(frameL)
    kpR, desR = orbR.compute(frameR, kpR)
    kpL, desL = orbL.compute(frameL, kpL)

    for each in kpR:
        if each.pt[0] <= 320:
            kpR_L.append(each)
        else:
            kpR_R.append(each)
    for each in kpL:
        if each.pt[0] <= 320:
	    kpL_L.append(each)
	else:
            kpL_R.append(each)


    # draw only keypoints location,not size and orientation
    img2 = cv2.drawKeypoints(frameR, kpR_L, None, color=(255,0,0), flags=0) #right
    img3 = cv2.drawKeypoints(img2, kpR_R, None, color=(0,255,0), flags=0)
    img4 = cv2.drawKeypoints(frameL, kpL_L, None, color=(255,0,0), flags=0)
    img5 = cv2.drawKeypoints(img4, kpL_R, None, color=(0,255,0), flags=0)

    # Display the resulting frame
    cv2.imshow('right',img3)
    cv2.imshow('left',img5)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
