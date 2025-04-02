import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLO Face Model
face_model = YOLO('weights_person_detection_2.pt')  # Replace with your trained weights

# Load Haar Cascade for Eye Detection
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Webcam not accessible")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Step 1: Face Detection using YOLO
    face_results = face_model(frame)
    for face in face_results[0].boxes:
        x1, y1, x2, y2 = map(int, face.xyxy[0])
        roi_gray = gray[y1:y2, x1:x2]
        roi_color = frame[y1:y2, x1:x2]

        # Step 2: Eye Detection inside detected face
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 4)
        for (ex, ey, ew, eh) in eyes:
            eye_gray = roi_gray[ey:ey+eh, ex:ex+ew]
            eye_color = roi_color[ey:ey+eh, ex:ex+ew]

            # Step 3: Threshold to find pupil
            _, thresh = cv2.threshold(eye_gray, 50, 255, cv2.THRESH_BINARY_INV)

            # Step 4: Find largest contour = pupil
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

            if contours:
                (cx, cy), radius = cv2.minEnclosingCircle(contours[0])
                center = (int(cx), int(cy))

                # Step 5: Draw
                cv2.circle(eye_color, center, int(radius), (0, 0, 255), 2)
                cv2.circle(eye_color, center, 2, (255, 0, 0), -1)  # Pupil dot

        # Draw face bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    
    cv2.imshow("YOLO Face & Haar Cascade Eye Tracker", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Updates from Previous Code:
# This script integrates two types of models for detecting faces and eyes, followed by tracking the pupils inside the eyes.
# Key updates and explanations:

# 1. Face Detection with YOLO:
#    - The script uses a YOLO (You Only Look Once) model (`weights_person_detection_2.pt`) to detect faces in real-time.
#    - The model's output provides bounding boxes around detected faces.
#    - The detected face region is cropped for further processing.

# 2. Eye Detection within the Face using Haar Cascade:
#    - After detecting the face, the script uses OpenCV's Haar Cascade classifier (`haarcascade_eye.xml`) to detect eyes inside the face bounding box.
#    - The face region is converted to grayscale, and eye detection is performed on this gray image.
#    - The eyes are detected and extracted from the color image for the next steps.

# 3. Pupil Detection using Thresholding and Contours:
#    - Inside each detected eye, the script applies thresholding (`cv2.threshold`) to segment the pupil region from the rest of the eye.
#    - The thresholded image is then processed to find the largest contour, which is assumed to be the pupil.
#    - The pupil is approximated as a circle using `cv2.minEnclosingCircle`.

# 4. Drawing Pupil and Face Boundaries:
#    - The center and radius of the detected pupil are drawn as a red circle on the eye.
#    - A small blue dot is drawn at the center of the pupil for better visualization.
#    - The bounding box around the detected face is drawn in blue.

# 5. Real-time Webcam Feed:
#    - The script captures video from the webcam using `cv2.VideoCapture(0)` and processes each frame.
#    - Detected faces, eyes, and pupils are drawn on each frame as it is displayed in a window.
#    - The `q` key is used to exit the program.

# These updates combine face detection using YOLO with eye detection via Haar cascades and pupil tracking, providing a more detailed analysis of the face and eye regions in real-time.

# Notes:
# - You will need to replace `weights_person_detection_2.pt` with your custom YOLO model weights for face detection.
# - The Haar Cascade for eye detection works well in most scenarios but might not be perfect in very complex environments.
# - This approach relies on real-time video processing with webcam input.
