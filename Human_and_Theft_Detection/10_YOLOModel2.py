# -----------------------------------
# FULL MOTION + FACE + PUPIL TRACKER
# -----------------------------------

import cv2
import numpy as np
import time
import csv
from ultralytics import YOLO
from collections import deque

# ----------------------------
# CONFIGURATION
# ----------------------------
STEP_THRESHOLD = 1.5
MAX_HISTORY = 15
CSV_LOG_PATH = "motion_log.csv"

# ----------------------------
# Load Models
# ----------------------------
pose_model = YOLO('yolov8n-pose.pt')   # Skeleton detector
face_model = YOLO('weights.pt')        # Your face detector (YOLO)
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")  # For eyes

# ----------------------------
# Webcam Setup
# ----------------------------
cap = cv2.VideoCapture("shopping_test_1.mp4")
if not cap.isOpened():
    print("Error: Webcam not accessible")
    exit()

# ----------------------------
# Initialize
# ----------------------------
prev_gray = None
person_trackers = {}
person_counter = 0
fps_counter = []

# ----------------------------
# CSV Logger
# ----------------------------
with open(CSV_LOG_PATH, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Timestamp", "PersonID", "Steps", "State"])

# ----------------------------
# Assign ID
# ----------------------------
def assign_id(nose_pos, existing_ids, threshold=50):
    for pid, data in existing_ids.items():
        prev_nose = data['history'][-1]
        if np.linalg.norm(nose_pos - prev_nose) < threshold:
            return pid
    global person_counter
    person_counter += 1
    return person_counter

# ----------------------------
# MAIN LOOP
# ----------------------------
while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ----------------------------
    # Face Detection (YOLO)
    # ----------------------------
    face_results = face_model(frame)
    detected_faces = []
    for r in face_results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detected_faces.append((x1, y1, x2, y2))

    # ----------------------------
    # Pose Detection (Skeleton)
    # ----------------------------
    pose_results = pose_model(frame)

    people_detected = False

    # ----------------------------
    # Process Each Skeleton
    # ----------------------------
    for person in pose_results[0].keypoints.xy:
        if person.shape[0] < 17:
            continue

        person = person.cpu().numpy()
        nose = person[0]
        left_ankle = person[15]
        right_ankle = person[16]

        # ----------------------------
        # Match skeleton only if nose inside a face box
        # ----------------------------
        matched_face = None
        for (x1, y1, x2, y2) in detected_faces:
            if x1 <= nose[0] <= x2 and y1 <= nose[1] <= y2:
                matched_face = (x1, y1, x2, y2)
                break

        if matched_face is None:
            continue

        people_detected = True

        # ----------------------------
        # Assign ID
        # ----------------------------
        pid = assign_id(nose, person_trackers)
        if pid not in person_trackers:
            person_trackers[pid] = {'history': deque(maxlen=MAX_HISTORY), 'steps': 0, 'state': 'Idle'}

        person_trackers[pid]['history'].append(nose)

        # ----------------------------
        # Step Detection
        # ----------------------------
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            motions = []
            for point in [left_ankle, right_ankle]:
                x, y = map(int, point)
                if 0 <= x < flow.shape[1] and 0 <= y < flow.shape[0]:
                    dx, dy = flow[y, x]
                    motions.append(np.sqrt(dx ** 2 + dy ** 2))
            mean_motion = np.median(motions) if motions else 0
            if mean_motion > STEP_THRESHOLD:
                person_trackers[pid]['steps'] += 1
                person_trackers[pid]['state'] = "Walking"
            else:
                person_trackers[pid]['state'] = "Not Walking"

        # ----------------------------
        # Draw Face Box
        # ----------------------------
        cv2.rectangle(frame, (matched_face[0], matched_face[1]), (matched_face[2], matched_face[3]), (0, 255, 255), 2)

        # ----------------------------
        # PUPIL DETECTION (inside YOLO face box)
        # ----------------------------
        face_roi_gray = gray[matched_face[1]:matched_face[3], matched_face[0]:matched_face[2]]
        face_roi_color = frame[matched_face[1]:matched_face[3], matched_face[0]:matched_face[2]]

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
        # CSV Logging
        # ----------------------------
        with open(CSV_LOG_PATH, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([time.strftime("%H:%M:%S"), pid, person_trackers[pid]['steps'], person_trackers[pid]['state']])

        # ----------------------------
        # Draw ID and State
        # ----------------------------
        cv2.putText(frame, f"ID:{pid} Steps:{person_trackers[pid]['steps']} State:{person_trackers[pid]['state']}",
                    (int(nose[0]), int(nose[1]) - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # ----------------------------
        # Draw Skeleton Keypoints
        # ----------------------------
        for (x, y) in person:
            cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)

    # ----------------------------
    # No Person Case
    # ----------------------------
    if not people_detected:
        cv2.putText(frame, "No faces in frame", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    prev_gray = gray.copy()

    # ----------------------------
    # FPS
    # ----------------------------
    fps_counter.append(1 / (time.time() - start_time))
    if len(fps_counter) > 30: fps_counter.pop(0)
    fps = np.mean(fps_counter)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow("Full System: Face-Guided + Skeleton + Pupil Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Changes from the last script:

# 1. Video Source: 
#   - Replaced webcam with a video file input: cap = cv2.VideoCapture("shopping_test_1.mp4").
#   - Error handling for webcam accessibility removed and now uses the video file.

# 2. Face Detection (YOLO):
#   - Integrated YOLO-based face detection (`face_model`) to detect and track faces along with skeletons.
#   - Replaced previous face detection method with YOLO detection for more accurate results.

# 3. Pose Detection:
#   - Kept YOLO model for pose detection (`pose_model`), but now the skeleton is matched to a detected face (if the nose is inside the face bounding box).

# 4. Step Detection:
#   - Added optical flow-based step detection, measuring motion at ankle positions (left_ankle, right_ankle) to determine if the person is walking or standing.
#   - Updated the person's state to "Walking" or "Not Walking" based on the detected motion.

# 5. Pupil Detection:
#   - Added eye detection using Haar cascades (`eye_cascade`) inside the detected face region to track pupils.
#   - Used contour detection to identify and mark the center of the pupil in the eye region.

# 6. CSV Logging:
#   - Continuously logs the person's state, step count, and timestamp into a CSV file (`motion_log.csv`) for tracking.

# 7. Drawing & Visualization:
#   - Displays face bounding boxes, skeleton keypoints, eye detection circles, and updated state information (ID, steps, state) on the frame.
#   - No detection message ("No faces in frame") added when no people or faces are detected.

# 8. FPS Calculation:
#   - Introduced FPS calculation to monitor the frame processing speed, displayed at the top left of the window.

# 9. Exit Condition:
#   - Loop exits when 'q' is pressed, closing the video feed window.
