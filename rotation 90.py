import cv2 
path='C:/Users/Ramanathan/OneDrive/Pictures/Leaving Fingers.jpg' 
src = cv2.imread(path) 
window_name = 'Image' 
image = cv2.rotate(src, cv2.ROTATE_180) 
cv2.imshow(window_name, image) 
cv2.waitKey(0) 
