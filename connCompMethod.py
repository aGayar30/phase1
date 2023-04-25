import os

import cv2
import numpy as np
from PIL import Image
from adjustcontrast import automatic_brightness_and_contrast
from matplotlib import pyplot as plt
from backgroundremover import backgroundremover

path = 'Images'
for filename in os.listdir(path):
    print(str(path+'/'+filename))
    # Read input image and resize 3shan yb2o kolohom haga wahda bnfs el size for any filters b3d kda:
    inputImage = cv2.imread(path + '/' + filename)

#removing background
img=backgroundremover(inputImage)
#cv2.imshow("Image",img)

#adjusting contrast
img,a,b= automatic_brightness_and_contrast(img)
#cv2.imshow("Image",img)

# Resize the image to a smaller size
# img = cv2.resize(img, (50, 30))  # adjust the size as needed

# Convert the image to grayscale
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)



# Threshold the image to create a binary image
ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Find connected components
output = cv2.connectedComponentsWithStats(thresh, connectivity=8)

# Get the number of connected components
num_labels = output[0]

# Get the connected components
labels = output[1]

# Create an empty image to draw the connected components on
output_img = np.zeros_like(img)

# Loop over each connected component
for label in range(1, num_labels):
    # Create a mask for the current component
    mask = labels == label

    # Get the area of the current component
    area = output[2][label, cv2.CC_STAT_AREA]

    # Draw the mask on the output image if area is greater than 30
    if area > 10:
        output_img[mask] = img[mask]

        # Get the bounding box of the current component
        x, y, w, h = output[2][label, cv2.CC_STAT_LEFT:cv2.CC_STAT_HEIGHT + 1]

        # Draw a rectangle around the current component on the output image
        cv2.rectangle(output_img, (x, y), (x + w, y + h), (255, 255, 255), 2)

# Display the output image
cv2.imshow('Connected Components', output_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
