import cv2
import numpy as np
from PIL import Image
from adjustcontrast import automatic_brightness_and_contrast
from matplotlib import pyplot as plt
from backgroundremover import backgroundremover

# Set image path
path = "Images/"
fileName = "img (1).png"

# Load the image
img1 = cv2.imread(path+fileName)

#adjusting contrast
# img1,a,b= automatic_brightness_and_contrast(img1)

# img1 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 10)

# Convert the image to grayscale
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

# Apply edge detection
img1 = cv2.Canny(img1, 150, 200)

# Find the contours of the connected edges
contours, hierarchy = cv2.findContours(img1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw boxes around the contours
for contour in contours:

    # Get the area of the contour
    area = cv2.contourArea(contour)

    # Get the perimeter of the contour
    perimeter = cv2.arcLength(contour, True)

    # Filter contours based on area size
    if perimeter > 85 and area > 20:  # or any other threshold value
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(img1, (x, y), (x + w, y + h), (255, 255, 255), 2)

# # Resize the image to a smaller size
# img = cv2.resize(img, (150, 100))  # adjust the size as needed

# Display the image
cv2.imshow('Digits', img1)
cv2.waitKey(0)
cv2.destroyAllWindows()


