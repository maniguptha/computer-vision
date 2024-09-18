import cv2
import numpy as np

# Define HSV range for detecting blue color (you can adjust for different colors)
lower_blue = np.array([100, 150, 0])
upper_blue = np.array([140, 255, 255])

# Path to the video file
video_path = 'C:/Users/Ramanathan/Music/video.mp4'  # Replace with your video file path

# Initialize the video capture object with the video file
cap = cv2.VideoCapture(video_path)

# Initialize canvas (to draw on) and previous coordinates
canvas = None
prev_x, prev_y = 0, 0

# Counter for saving different versions of the drawing
save_counter = 1

while True:
    # Capture the video frame
    ret, frame = cap.read()

    # Break the loop if no more frames are available
    if not ret:
        break

    # Flip the frame horizontally (for mirror effect)
    frame = cv2.flip(frame, 1)

    # Create a blank canvas if it doesn't exist
    if canvas is None:
        canvas = np.zeros_like(frame)

    # Convert the frame from BGR to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a mask for detecting blue color
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Find contours on the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If contours are detected, track the marker's position
    if contours:
        # Get the largest contour (assuming it's the marker)
        c = max(contours, key=cv2.contourArea)

        # Find the circle that encloses the contour
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        x, y = int(x), int(y)

        # If we have previous coordinates, draw a line from previous to current
        if prev_x != 0 and prev_y != 0:
            cv2.line(canvas, (prev_x, prev_y), (x, y), (255, 0, 0), 5)  # Draw blue line

        # Update the previous coordinates
        prev_x, prev_y = x, y
    else:
        # Reset previous coordinates if no object is detected
        prev_x, prev_y = 0, 0

    # Merge the canvas with the frame to display the drawing
    combined = cv2.add(frame, canvas)

    # Show the combined output
    cv2.imshow('Virtual Paint', combined)

    # Key controls:
    # 'c' to clear the canvas, 'q' to quit, 's' to save
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        canvas = None  # Clear the canvas
    elif key == ord('s'):
        # Save the current canvas as a .jpg file
        filename = f'painting_{save_counter}.jpg'
        cv2.imwrite(filename, canvas)
        print(f'Saved {filename}')
        save_counter += 1
    elif key == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
