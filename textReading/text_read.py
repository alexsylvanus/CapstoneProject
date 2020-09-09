# File: text_read.py
# Author: Alex Sylvanus

# Imports
from imutils.object_detection import non_max_suppression
import numpy as np
import pytesseract
import argparse
import cv2
import pyttsx3

# Define the text prediction decoding function
def decode_predictions(scores, geometry):
    # Get the number of rows and columns
    (numRows, numCols) = scores.shape[2:4]

    # Initialize the bounding rectangles
    rects = []

    # Initialize the confidence scores vector
    confidences = []

    # Loop over the rows
    for y in range(0, numRows):
        # Extract the probabilities
        scoresData = scores[0, 0, y]

        # Extract geometrical data containing text locations
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        # Loop over the columns
        for x in range(0, numCols):
            # Ignore scores with small probabilities
            if scoresData[x] < 0.5:
                continue

            # Compute offset factor
            (offsetX, offsetY) = (x*4.0, y*4.0)

            # Exctract rotation angle to compute sin and cos
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # Derive width and height of bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            # Compute the starting and ending points
            endX = int(offsetX + (cos*xData1[x]) + (sin*xData2[x]))
            endY = int(offsetY - (sin*xData1[x]) + (cos*xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            # Add box coordinates to rectangle list
            rects.append((startX, startY, endX, endY))

            # Add probability score to confidence vector
            confidences.append(scoresData[x])

    # Return the resulting box data and confidence vector
    return (rects, confidences)

# Define function for running Deep Learning algorithm on image
def get_image_text(image, n_net, width=608, height=320, padding=0.1):
    # Copy the image
    orig = image.copy()
    (origH, origW) = image.shape[:2]

    # Set new width and height
    (newW, newH) = (width, height)
    rW = origW/float(newW)
    rH = origH/float(newH)

    # Resize the image and get new dimensions
    image = cv2.resize(image, (newW, newH))
    (H, W) = image.shape[:2]

    # Define layer names for Deep Learning model
    layerNames = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"
    ]

    # Load the pre-trained text detector

    # net = cv2.dnn.readNet("frozen_east_text_detection.pb")
    net = n_net
    # Construct a blob and perform a forward pass of the model
    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H), (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)

    # Obtain scores and locations from neural network
    (scores, geometry) = net.forward(layerNames)

    # Decode the predictions
    (rects, confidences) = decode_predictions(scores, geometry)

    # Suppress overlapping bounding boxes
    boxes = non_max_suppression(np.array(rects), probs=confidences)

    # Initialize the list of results
    results = []

    # Loop over the bounding boxes
    for (startX, startY, endX, endY) in boxes:
        # Scale boxes based on ratios
        startX = int(startX*rW)
        startY = int(startY*rH)
        endX = int(endX*rW)
        endY = int(endY*rH)

        # Compute deltas from the padding
        dX = int((endX - startX)*padding)
        dY = int((endY - startY)*padding)

        # Apply padding to the bounding box
        startX = max(0, startX - dX)
        startY = max(0, startY - dY)
        endX = min(origW, endX + (dX*2))
        endY = min(origH, endY + (dY*2))

        # Extract original padded ROI
        roi = orig[startY:endY, startX:endX]

        # Define arguments for tesseract command
        config = ("-l eng --oem 1 --psm 3")
        text = pytesseract.image_to_string(roi, config=config)

        # Append the extracted text and box coordinates to the results
        results.append(((startX, startY, endX, endY), text))

    # Sort the results by box coordinates
    results = sorted(results, key=lambda r:r[0][1])

    # Return the results
    return results

# Define main script
def main():
    # Get video capture
    capR = cv2.VideoCapture(0)
    engine = pyttsx3.init()

    # Load neural network
    print("Loading the EAST text detector...")
    net = cv2.dnn.readNet("frozen_east_text_detection.pb")
    while(True):
        # Read the image
    	ret, image = capR.read()

        # Display the image
    	cv2.imshow('frame', image)

    	if cv2.waitKey(1) & 0xFF == ord('q'):
    		break
        if cv2.waitKey(1) & 0xFF == ord('c'):
            # Get the text from the current image
            res = get_image_text(image, net)

            # Loop over the results
            for ((A, B, C, D), text) in res:
                # Print out the text
                print(text)
                engine.say(text)

            # Say the text
            engine.runAndWait()

    capR.release()
    cv2.destroyAllWindows()

# Run main
if __name__ == '__main__':
    main()
