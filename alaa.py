import os

import cv2

import numpy as np


path = 'Images'
for filename in os.listdir(path):
    print(str(path+'/'+filename))
    # Read input image and resize 3shan yb2o kolohom haga wahda bnfs el size for any filters b3d kda:
    inputImage = cv2.imread(path + '/' + filename)
    inputCopy = inputImage.copy()
    inputImage = cv2.resize(inputImage, (100, 75))
    inputCopy = cv2.resize(inputCopy, (100, 75))

    # Convert BGR to grayscale:
    grayscaleImage = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)

    # Set the adaptive thresholding:
    windowSize = 25
    windowConstant = -1

    # Apply the threshold:
    binaryImage = cv2.adaptiveThreshold(grayscaleImage, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                        windowSize, windowConstant)

    # Perform Connected Components:
    componentsNumber, labeledImage, componentStats, componentCentroids = cv2.connectedComponentsWithStats(binaryImage,
                                                                                                          connectivity=4)

    # Set the minimum pixels for the area filter to filter connected components:
    minArea = 20

    # Get the indices/labels of the remaining components based on the area stat
    # (skip the background component at index 0)
    remainingComponentLabels = [i for i in range(1, componentsNumber) if componentStats[i][4] >= minArea]

    # Filter the labeled pixels based on the remaining labels,
    # assign pixel intensity to 255 (uint8) for the remaining pixels
    # y3ni b3d de el mafood yfdal the largest connected components that hopefully contains the digits
    # one problem is that background connected components may still exist
    filteredImage = np.where(np.isin(labeledImage, remainingComponentLabels) == True, 255, 0).astype('uint8')

    # Set kernel (structuring element) size:
    kernelSize = 3

    # Set operation iterations:
    opIterations = 1

    # Get the structuring element:
    maxKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernelSize, kernelSize))

    # Perform closing:
    closingImage = cv2.morphologyEx(filteredImage, cv2.MORPH_CLOSE, maxKernel, None, None, opIterations,
                                    cv2.BORDER_REFLECT101)

    # perform smoothing
    smoothedImage = cv2.GaussianBlur(closingImage, (13, 13), 0)

    # perform canny edge detection:
    edge_image = cv2.Canny(smoothedImage, 120, 200)

    # Get each bounding box
    # Find the big contours on the filtered image:
    contours, hierarchy = cv2.findContours(edge_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    contours_poly = [None] * len(contours)
    # The Bounding Rectangles will be stored here:
    boundRect = []

    # Alright, just look for the outer bounding boxes:
    for i, c in enumerate(contours):

        if hierarchy[0][i][3] == -1:
            contours_poly[i] = cv2.approxPolyDP(c, 3, True)
            boundRect.append(cv2.boundingRect(contours_poly[i]))

    # Get the dimensions of the image
    height, width, channels = inputCopy.shape

    # Sort the bounding boxes based on their areas in descending order
    boundRect = sorted(boundRect, key=lambda x: x[2] * x[3], reverse=True)

    # Create an empty list to store the selected bounding boxes
    selectedBoundRect = []

    # Iterate through each bounding box
    for i, box in enumerate(boundRect):
        # Check if the current bounding box overlaps with any of the previously selected bounding boxes
        overlaps = False
        for selectedBox in selectedBoundRect:
            if box[0] < selectedBox[0] + selectedBox[2] and box[0] + box[2] > selectedBox[0] and \
                    box[1] < selectedBox[1] + selectedBox[3] and box[1] + box[3] > selectedBox[1]:
                overlaps = True
                break

        # If the current bounding box doesn't overlap with any of the previously selected bounding boxes, add it to the list
        if not overlaps:
            selectedBoundRect.append(box)
    # print(len(boundRect))
    # Draw the bounding boxes on the (copied) input image:
    for i in range(len(boundRect)):
        color = (0, 255, 0)
        area= int(boundRect[i][2]) * int(boundRect[i][3])
        x, y, w, h = boundRect[i]

        # Calculate the center of the bounding box
        center_x = x + (w / 2)
        center_y = y + (h / 2)

        # filter contours according to the area of the rectangle (parameter re5em)
        if ( area > 225 and area < 3000 and center_x > (width / 3) and center_x < (2 * width / 3) and center_y > (height / 3) and center_y < (2 * height / 3)):
            cv2.rectangle(inputCopy, (int(boundRect[i][0]), int(boundRect[i][1])),
                          (int(boundRect[i][0] + boundRect[i][2]), int(boundRect[i][1] + boundRect[i][3])), color, 2)



    # Show the output image
    cv2.imshow("Image", inputCopy)
    cv2.waitKey(0)




