import cv2
import numpy as np
import matplotlib.pyplot as plt
image = cv2.imread('C:/Users/Ramanathan/OneDrive/Pictures/21627.jpg', cv2.IMREAD_GRAYSCALE)
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
sobel_edge = np.sqrt(sobel_x**2 + sobel_y**2)
sobel_edge = np.uint8(sobel_edge)
plt.subplot(1, 2, 1), plt.title('Sobel X'), plt.imshow(sobel_x, cmap='gray')
plt.subplot(1, 2, 2), plt.title('Sobel Y'), plt.imshow(sobel_y, cmap='gray')
plt.show()
