import cv2
import numpy as np

# Read the grayscale image
gray_image = cv2.imread("C:\\Users\\ADMIN\\Desktop\\flask project\\y186LSWMD.png", cv2.IMREAD_GRAYSCALE)

# Create a three-channel image from the grayscale image
colored_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)

# Save the colored image
cv2.imwrite("C:\\Users\\ADMIN\\Desktop\\flask project\\yc186LSWMD.png", colored_image)
