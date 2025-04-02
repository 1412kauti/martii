import cv2
import numpy as np

# Load the Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Initialize the HOG descriptor for pedestrian detection
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Start the webcam
cap = cv2.VideoCapture(0)

# ORB detector
orb = cv2.ORB_create()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the frame to grayscale (required for face detection, Harris corner detection, and ORB)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image using Haar Cascade
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(30, 30))

    # Harris Corner Detection
    gray_float = np.float32(gray)
    corners = cv2.cornerHarris(gray_float, 2, 3, 0.04)
    corners = cv2.dilate(corners, None)

    # Threshold to identify strong corners
    frame[corners > 0.01 * corners.max()] = [0, 0, 255]  # Red for corners

    # ORB Feature Detection
    kp, des = orb.detectAndCompute(gray, None)  # Detect ORB keypoints and compute descriptors

    # Draw the ORB keypoints
    frame = cv2.drawKeypoints(frame, kp, None, color=(0, 255, 0), flags=0)  # Green keypoints

    # Detect people using HOG
    # HOG detects pedestrians, but can also be used for other objects
    boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8), padding=(8, 8), scale=1.05)

    # Draw the HOG bounding boxes
    for (x, y, w, h) in boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)  # Yellow for HOG detections

    # Draw rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue for faces

    # Display the resulting frame
    cv2.imshow("Face, Harris Corner, ORB, and HOG Detection", frame)

    # Break the loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()

# Changes compared to previous script:
# 1. Added ORB feature detection: 
#    - ORB (Oriented FAST and Rotated BRIEF) detects keypoints and computes descriptors for feature matching.
#    - Keypoints are drawn as green dots on the frame. It is similar to SIFT.
# 
# 2. Added HOG (Histogram of Oriented Gradients) for pedestrian detection:
#    - HOG detects pedestrians (or similar objects) and is used with an SVM classifier for detection.
#    - Pedestrian detection results are shown with yellow bounding boxes.
#
# 3. Adjusted the minNeighbors parameter in face detection:
#    - This was increased to 6 for better face detection results (may help reduce false positives).