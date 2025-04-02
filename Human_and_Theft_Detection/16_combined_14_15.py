import cv2
import numpy as np
import time
from ultralytics import YOLO
from collections import deque

# ----------------------------
# CONFIGURATION
# ----------------------------
STEP_THRESHOLD = 1.5
MAX_HISTORY = 15

# ----------------------------
# Load Models
# ----------------------------
pose_model = YOLO('yolov8n-pose.pt')   # Skeleton detector
face_model = YOLO('weights.pt')        # Face detector
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")  # Eye detector

# ----------------------------
# Video Source
# ----------------------------
cap = cv2.VideoCapture("Supermarket Footage.mp4")
if not cap.isOpened():
    print("Error: Video not accessible")
    exit()

# ----------------------------
# Background Subtractor
# ----------------------------
backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50)

# ----------------------------
# Helper Variables
# ----------------------------
prev_gray = None
fps_counter = []

# ----------------------------
# Main Loop
# ----------------------------
while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    frame_motion = frame.copy()
    frame_pose = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ----------------------------
    # MOTION SYSTEM
    # ----------------------------
    fg_mask = backSub.apply(frame_motion)
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt) > 500:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame_motion, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame_motion, "Motion Detected", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # ----------------------------
    # FACE + PUPIL SYSTEM
    # ----------------------------
    face_results = face_model(frame_pose)
    for r in face_results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame_pose, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame_pose, "Face", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Pupil detection inside face ROI
            face_roi_gray = gray[y1:y2, x1:x2]
            face_roi_color = frame_pose[y1:y2, x1:x2]

            eyes = eye_cascade.detectMultiScale(face_roi_gray)
            for (ex, ey, ew, eh) in eyes:
                eye_gray = face_roi_gray[ey:ey+eh, ex:ex+ew]
                eye_color = face_roi_color[ey:ey+eh, ex:ex+ew]

                _, thresh = cv2.threshold(eye_gray, 50, 255, cv2.THRESH_BINARY_INV)
                contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

                if contours:
                    (cx, cy), radius = cv2.minEnclosingCircle(contours[0])
                    center = (int(cx), int(cy))
                    cv2.circle(eye_color, center, int(radius), (0, 0, 255), 2)
                    cv2.circle(eye_color, center, 2, (255, 0, 0), -1)

    # ----------------------------
    # SKELETON SYSTEM
    # ----------------------------
    pose_results = pose_model(frame_pose)
    for person in pose_results[0].keypoints.xy:
        person = person.cpu().numpy()
        for (x, y) in person:
            cv2.circle(frame_pose, (int(x), int(y)), 3, (0, 255, 0), -1)

    # ----------------------------
    # Merge Views
    # ----------------------------
    combined = cv2.addWeighted(frame_motion, 0.5, frame_pose, 0.5, 0)

    # ----------------------------
    # FPS Counter
    # ----------------------------
    fps_counter.append(1 / (time.time() - start_time))
    if len(fps_counter) > 30: fps_counter.pop(0)
    fps = np.mean(fps_counter) if fps_counter else 0
    cv2.putText(combined, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # ----------------------------
    # Show
    # ----------------------------
    cv2.imshow("Combined System", combined)
    cv2.imshow("Foreground Mask", fg_mask)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Overview of the script:
# This script is designed to perform several real-time computer vision tasks on a video source:
# 1. **Motion Detection:**
#    - The script uses a background subtractor (`MOG2`) to detect moving objects in the video feed.
#    - Detected motion is highlighted with green rectangles, and the message "Motion Detected" is displayed.
#    - The motion system helps track dynamic objects in the scene.

# 2. **Face and Pupil Detection:**
#    - The script uses a YOLO model (`weights.pt`) to detect faces in the video feed.
#    - Once a face is detected, it uses the Haar Cascade classifier to detect eyes within the detected face region.
#    - It then applies thresholding and contour analysis to detect the pupils within the eyes and highlights them with red circles.
#    - Face and eye detection are annotated with bounding boxes and text labels.

# 3. **Pose Estimation (Skeleton Detection):**
#    - The YOLO pose model (`yolov8n-pose.pt`) is used to detect human keypoints (such as joints and limbs) in the frame.
#    - Detected keypoints are visualized with green circles to highlight the skeleton structure.

# 4. **Combined View:**
#    - The frames from motion detection, face and pupil detection, and skeleton detection are merged into a combined view.
#    - This helps visualize all processes together in one image, with detected objects, faces, eyes, and skeleton keypoints.

# 5. **FPS Display:**
#    - The script calculates the FPS (Frames Per Second) and displays it on the video frame to monitor the performance of the system.

# 6. **Key Features:**
#    - **Motion Detection:** Green rectangles and "Motion Detected" message for detected movements.
#    - **Face Detection:** Bounding boxes and labels around faces, with pupil detection inside eyes.
#    - **Pose Estimation:** Keypoints detected and visualized to represent a skeleton structure.
#    - **Combined View:** Merging of all detection outputs into a single frame.
#    - **FPS Tracking:** Displays the average FPS to monitor system performance.

# 7. **User Interaction:**
#    - Press the 'q' key to quit the application.

# Key Variables and Functions:
#  - `STEP_THRESHOLD`: Defines the threshold for detecting motion based on background subtraction.
#  - `MAX_HISTORY`: Used to store recent FPS values to calculate average FPS.
#  - `backSub`: A background subtractor to detect moving objects.
#  - `pose_model`, `face_model`: YOLO models used for detecting poses and faces.
#  - `eye_cascade`: A Haar Cascade classifier used for eye detection inside the face region.
