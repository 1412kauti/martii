# -----------------------------------
# FULL PROFESSIONAL MOTION ANALYSIS SYSTEM (SAFE VERSION)
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
EYE_MOVEMENT_THRESHOLD = 0.5
MAX_HISTORY = 15
CSV_LOG_PATH = "motion_log.csv"

# ----------------------------
# Load Models
# ----------------------------
pose_model = YOLO('yolov8n-pose.pt')   # For skeleton
face_model = YOLO('weights.pt')        # For face detection

# ----------------------------
# Webcam Setup
# ----------------------------
cap = cv2.VideoCapture(0)
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
# Prepare CSV Log
# ----------------------------
with open(CSV_LOG_PATH, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Timestamp", "PersonID", "Steps", "State", "EyeMovement"])

# ----------------------------
# Helper: Assign ID
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
# Main Loop
# ----------------------------
while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ----------------------------
    # Detect Skeletons and Faces
    # ----------------------------
    pose_results = pose_model(frame)
    face_results = face_model(frame)

    detected_faces = []
    for r in face_results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detected_faces.append((x1, y1, x2, y2))

    people_detected = False

    # ----------------------------
    # Process Each Person
    # ----------------------------
    for person in pose_results[0].keypoints.xy:
        if person.shape[0] < 17:
            continue  #  Safe check to skip incomplete detections

        people_detected = True  # At least one valid person detected

        person = person.cpu().numpy()

        nose = person[0]
        left_eye = person[1]
        right_eye = person[2]
        left_ankle = person[15]
        right_ankle = person[16]

        pid = assign_id(nose, person_trackers)

        if pid not in person_trackers:
            person_trackers[pid] = {'history': deque(maxlen=MAX_HISTORY), 'steps': 0, 'state': 'Idle', 'last_eye': ''}

        person_trackers[pid]['history'].append(nose)

        # ----------------------------
        # Step-Cycle Detection
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
                person_trackers[pid]['state'] = "Standing"

        # ----------------------------
        # Eye Movement Detection
        # ----------------------------
        matched_face = None
        for (x1, y1, x2, y2) in detected_faces:
            if x1 <= nose[0] <= x2 and y1 <= nose[1] <= y2:
                matched_face = (x1, y1, x2, y2)
                break

        eye_direction = ""
        if matched_face:
            x1, y1, x2, y2 = matched_face
            fx, fy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            eye_roi = gray[y1:y2, x1:x2]
            if eye_roi.size > 0 and prev_gray is not None:
                prev_eye_roi = prev_gray[y1:y2, x1:x2]
                flow_eye = cv2.calcOpticalFlowFarneback(prev_eye_roi, eye_roi, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                dx = np.median(flow_eye[..., 0])
                dy = np.median(flow_eye[..., 1])

                if abs(dx) > EYE_MOVEMENT_THRESHOLD or abs(dy) > EYE_MOVEMENT_THRESHOLD:
                    eye_direction = "Left" if dx < 0 else "Right" if dx > 0 else ""
                    eye_direction += "/Up" if dy < 0 else "/Down" if dy > 0 else ""

                if eye_direction:
                    person_trackers[pid]['last_eye'] = eye_direction

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)
                cv2.putText(frame, f"Eye: {person_trackers[pid]['last_eye']}",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        # ----------------------------
        # Log to CSV
        # ----------------------------
        with open(CSV_LOG_PATH, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([time.strftime("%H:%M:%S"), pid, person_trackers[pid]['steps'],
                             person_trackers[pid]['state'], person_trackers[pid]['last_eye']])

        # ----------------------------
        # Visualization
        # ----------------------------
        cv2.putText(frame, f"ID:{pid} Steps:{person_trackers[pid]['steps']} State:{person_trackers[pid]['state']}",
                    (int(nose[0]), int(nose[1]) - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        for (x, y) in person:
            cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)

    # ----------------------------
    # Handle No Person Detected
    # ----------------------------
    if not people_detected:
        cv2.putText(frame, "Nobody in frame", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    prev_gray = gray.copy()

    # ----------------------------
    # FPS Counter
    # ----------------------------
    fps_counter.append(1 / (time.time() - start_time))
    if len(fps_counter) > 30: fps_counter.pop(0)
    fps = np.mean(fps_counter)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow("Full System: Multi-Person Tracking + Step + Eye Direction", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# 1. CONFIGURATION: 
# Defines thresholds for step and eye movement detection, max history for tracking, and CSV log path.

# 2. MODEL LOADING: 
# Loads YOLO models for human pose detection ('yolov8n-pose.pt') and face detection ('weights.pt').

# 3. WEBCAM SETUP: 
# Initializes the webcam for video capture and exits if not accessible.

# 4. INITIALIZATION: 
# Sets up variables for tracking detected people, FPS calculation, and previous frame storage.

# 5. CSV LOGGING: 
# Creates a CSV file to log timestamped data on detected persons (ID, steps, state, eye movement).

# 6. ASSIGN ID: 
# Assigns or reuses unique IDs for persons based on nose position and movement.

# 7. MAIN LOOP: 
# Captures frames, performs pose and face detection, and updates movement tracking.

# 8. MOTION DETECTION: 
# Detects walking/standing based on optical flow at ankle positions and updates person states.

# 9. EYE MOVEMENT: 
# Tracks eye direction by analyzing changes in face regions and updates the eye movement state.

# 10. LOGGING: 
# Logs detected movement and eye data to CSV for each person.

# 11. VISUALIZATION: 
# Displays webcam feed with ID, step count, state, and eye movement info overlaid on person keypoints.

# 12. NO PERSON DETECTED: 
# Displays a message when no people are detected in the frame.

# 13. FPS COUNTER: 
# Calculates and displays FPS based on frame processing time.

# 14. LOOP TERMINATION: 
# Exits the loop and closes the webcam feed when the user presses 'q'.
