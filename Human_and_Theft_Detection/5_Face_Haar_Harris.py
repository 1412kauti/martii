import cv2
import numpy as np

# Load the Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Start the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # Convert the frame to grayscale (required for face detection and Harris corner detection)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Harris Corner Detection
    # Convert to float32 for Harris corner detection
    gray_float = np.float32(gray)
    
    # Perform Harris corner detection
    corners = cv2.cornerHarris(gray_float, 2, 3, 0.04)
    
    # Dilate the corners to enhance them
    corners = cv2.dilate(corners, None)
    
    # Threshold to identify strong corners
    frame[corners > 0.01 * corners.max()] = [0, 0, 255]  # Red for corners
    
    # Draw rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    # Display the resulting frame
    cv2.imshow("Face and Harris Corner Detection", frame)
    
    # Break the loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()


# Explanation:
# This script performs both face detection and Harris corner detection in real-time using a webcam feed.

# 1. Haar Cascade Classifier for Face Detection:
#    - Haar cascades are machine learning-based classifiers used to detect objects (faces, in this case).
#    - The classifier is trained using positive and negative images and detects faces in real-time.
#    - detectMultiScale detects faces in the image by scanning at multiple scales and detecting faces based on features.

# 2. Harris Corner Detection:
#    - Harris corner detection is used to identify "corner-like" regions of an image.
#    - A corner is defined as an area where there are significant changes in the intensity in all directions.
#    - The cornerHarris function computes corner strength for each pixel, and the result is then dilated to make the corners more visible.

# 3. Grayscale Conversion:
#    - Both face detection and Harris corner detection work better on grayscale images because color information is less critical for these tasks.
#    - The frame is converted to grayscale using cvtColor.

# 4. Dilating Corners:
#    - After detecting corners, dilating (enlarging) them helps make them more prominent, making it easier to visualize them.

# 5. Thresholding:
#    - Thresholding the corners allows us to highlight only the strongest corners by checking if their intensity exceeds a certain value (0.01 times the maximum corner strength).

# 6. Display:
#    - The processed frame is displayed with rectangles around faces and red marks for strong corners.
#    - Pressing the 'q' key will exit the loop and close the window.

# 7. Cleanup:
#    - The video capture is released, and all OpenCV windows are closed when the program exits.