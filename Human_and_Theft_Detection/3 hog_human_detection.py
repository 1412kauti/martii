import cv2

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cap = cv2.VideoCapture("shopping_test_1.mp4")
if not cap.isOpened():
    print("Error: Webcam not found")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.resize(frame, (640, 480))
    (boxes, _) = hog.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.05)

    for (x, y, w, h) in boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Human Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Explanation:

# What is the HOG Descriptor?
# The Histogram of Oriented Gradients (HOG) is a method used in computer vision to detect objects, like people.
# It works by:
# 0.1. Splitting an image into small parts (cells).
# 0.2. Finding edges and their directions in each cell.
# 0.3. Making a histogram that counts how often edges appear in different directions.
# 0.4. Adjusting these values to handle lighting changes.
# 0.5. Using this information to help a classifier (like an SVM) recognize objects.

# 1. HOG Descriptor: The HOG algorithm extracts features from an image by capturing the distribution of gradient orientations.
#    It is widely used in object detection tasks, particularly for detecting pedestrians.

# 2. Pre-trained SVM Detector: The cv2.HOGDescriptor_getDefaultPeopleDetector() provides a trained SVM classifier that recognizes
#    human shapes based on HOG features.

# 3. Video Capture: The script reads frames from a video file ("shopping_test_1.mp4") using OpenCV’s cv2.VideoCapture.
#    If the file is not found or cannot be opened, an error message is displayed.

# 4. Frame Processing: Each frame is resized to (640, 480) to standardize input size and improve performance.

# 5. Human Detection:
#    - detectMultiScale scans the frame at different scales and positions to detect humans.
#    - winStride=(4, 4): Determines how much the detection window moves between scans.
#    - padding=(8, 8): Adds padding around detection windows for better results.
#    - scale=1.05: Controls the image scaling between detections (lower values make detection more accurate but slower).

# 6. Drawing Bounding Boxes: Detected human regions are enclosed in green rectangles.

# 7. Displaying Output: The processed frame is displayed in a window labeled "Human Detection."

# 8. Exiting the Program: The loop continues until the 'q' key is pressed, at which point resources are released.