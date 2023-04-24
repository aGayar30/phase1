import cv2

import numpy as np

# Load the image
img = cv2.imread('digits9.png')

# Resize the image to a smaller size
# img = cv2.resize(img, (50, 30))  # adjust the size as needed

# Convert the image to grayscale
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 10)

# Apply edge detection
img = cv2.Canny(img, 30, 70)

# # Find the contours of the connected edges
contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw boxes around the contours
for contour in contours:

    # Get the area of the contour
    area = cv2.contourArea(contour)

    # Get the perimeter of the contour
    perimeter = cv2.arcLength(contour, True)

    # Filter contours based on area size
    if perimeter > 85 and area > 20:  # or any other threshold value
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)

# Resize the image to a smaller size
img = cv2.resize(img, (150, 100))  # adjust the size as needed

# Display the image
cv2.imshow('Digits', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
