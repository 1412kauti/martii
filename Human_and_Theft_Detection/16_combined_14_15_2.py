import cv2
import time
import numpy as np
from ultralytics import YOLO

class SimpleTracker:
    def __init__(self, max_distance=50):
        self.nextID = 1
        self.objects = {}  # objectID -> centroid
        self.max_distance = max_distance

    def update(self, detections):
        new_objects = {}
        input_centroids = []

        for (x, y, w, h) in detections:
            cX = int(x + w / 2)
            cY = int(y + h / 2)
            input_centroids.append((cX, cY))

        if len(self.objects) == 0:
            for centroid in input_centroids:
                new_objects[self.nextID] = centroid
                self.nextID += 1
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            used = set()

            for centroid in input_centroids:
                distances = [np.linalg.norm(np.array(centroid) - np.array(c)) for c in objectCentroids]
                if len(distances) == 0:
                    continue
                min_idx = int(np.argmin(distances))
                if distances[min_idx] < self.max_distance and objectIDs[min_idx] not in used:
                    objectID = objectIDs[min_idx]
                    new_objects[objectID] = centroid
                    used.add(objectID)
                else:
                    new_objects[self.nextID] = centroid
                    self.nextID += 1

        self.objects = new_objects
        return self.objects

# -------------------------
# Main
# -------------------------

# Load Models
pose_model = YOLO('yolov8n-pose.pt')  # Skeleton detector
cap = cv2.VideoCapture("shopping_test_1.mp4")  # or 0 for webcam

backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50)
tracker = SimpleTracker(max_distance=60)

# Lock system
lock_id = None
lock_centroid_history = []

# Cooldown system for "Hello, can I help?"
message_active = False
last_message_time = 0
cooldown_seconds = 10

# Simulated "hello"
simulate_hello_triggered = False
hello_trigger_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    fg_mask = backSub.apply(frame)

    # Morphological filtering
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)

    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 500:
            x, y, w, h = cv2.boundingRect(cnt)
            detections.append((x, y, w, h))

    objects = tracker.update(detections)
    object_detected = len(objects) > 0
    current_time = time.time()

    # === AUTO LOCK ===
    if lock_id is None:
        biggest_area = 0
        for objectID, centroid in objects.items():
            for (x, y, w, h) in detections:
                if abs(centroid[0] - (x + w // 2)) < 1 and abs(centroid[1] - (y + h // 2)) < 1:
                    area = w * h
                    if area > biggest_area:
                        biggest_area = area
                        lock_id = objectID
                        lock_centroid_history = []  # reset history
        if lock_id is not None:
            print(f"LOCKED on ID: {lock_id}")

    # === AUTO UNLOCK ===
    if lock_id is not None and lock_id not in objects:
        print(f"ID {lock_id} left the scene. Unlocking.")
        lock_id = None
        lock_centroid_history = []
        simulate_hello_triggered = False  # reset hello if lock breaks

    # === SKELETON SYSTEM ===
    detected_skeletons = []
    pose_results = pose_model(frame)
    for person in pose_results[0].keypoints.xy:
        person = person.cpu().numpy()
        detected_skeletons.append(person)
        for (x, y) in person:
            cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)  # Draw skeleton keypoints

    # === DRAW ===
    stationary = False
    for objectID, centroid in objects.items():
        for (x, y, w, h) in detections:
            if abs(centroid[0] - (x + w // 2)) < 1 and abs(centroid[1] - (y + h // 2)) < 1:
                # Check if there's a skeleton point inside the bounding box
                box_center = (x + w // 2, y + h // 2)
                has_skeleton_point = False
                for skeleton in detected_skeletons:
                    for (sx, sy) in skeleton:
                        if x <= sx <= x + w and y <= sy <= y + h:
                            has_skeleton_point = True
                            break
                if has_skeleton_point:
                    if objectID == lock_id:
                        color = (0, 0, 255)  # RED for locked
                        # --- STATIONARY CHECK ---
                        lock_centroid_history.append(centroid)
                        if len(lock_centroid_history) > 2:
                            lock_centroid_history.pop(0)

                        if len(lock_centroid_history) == 2:
                            dist = np.linalg.norm(np.array(lock_centroid_history[0]) - np.array(lock_centroid_history[1]))
                            if dist < 2:
                                stationary = True

                    else:
                        color = (0, 255, 0)  # GREEN for others

                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, f"ID {objectID}", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # === Display "Hello, can I help?" ===
    if object_detected and (not message_active or (current_time - last_message_time) > cooldown_seconds):
        message_active = True
        last_message_time = current_time

    if message_active and (current_time - last_message_time < 3):
        cv2.putText(frame, "Hey there, can I help anyone?", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    elif message_active and (current_time - last_message_time >= cooldown_seconds):
        message_active = False

    # === HELLO TRIGGER auto-reset ===
    if simulate_hello_triggered and (current_time - hello_trigger_time > 5):
        simulate_hello_triggered = False

    # === Display "Hi there, how can I help you today?" ONLY when LOCKED and (stationary OR hello)
    if lock_id is not None and (stationary or simulate_hello_triggered):
        cv2.putText(frame, "Hi there, how can I help you today?", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

    # === Show Frames ===
    cv2.imshow("Closest Object Highlighted + Lock + Hello + Stationary", frame)
    cv2.imshow("Foreground Mask", fg_mask)

    # === Simulate "hello" trigger ===
    key = cv2.waitKey(30) & 0xFF
    if key == ord('h') and lock_id is not None:
        simulate_hello_triggered = True
        hello_trigger_time = time.time()
        print("Simulated: Heard 'hello'")

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Updates from Previous Code:
# This script extends the previous motion, face, and skeleton detection with several new features:
#
# 1. Simple Object Tracker (SimpleTracker class):
#    - Introduced a custom object tracker to track moving objects across frames by associating them with unique object IDs.
#    - The tracker updates the positions of objects based on their centroids and keeps track of them until they leave the frame or get too far apart.
#    - The tracker uses a maximum distance threshold (`max_distance`) to decide whether an object has moved too far and should be re-identified.
#
# 2. Auto Locking Mechanism:
#    - When no object is locked, the script automatically locks onto the largest object in the frame based on bounding box area.
#    - The lock is maintained as long as the object is detected, and it is automatically unlocked when the object leaves the frame.
#    - This is useful for focusing attention on a specific object in the scene.
#
# 3. Stationary Check:
#    - A "stationary" check is added, where the script monitors if the locked object remains still for a period of time.
#    - If the locked object stays in the same position (within a small distance threshold), it is considered "stationary."
#    - This helps identify when the locked object might be standing still, triggering specific behavior.
#
# 4. Dynamic Message System:
#    - The system will display the message "Hey there, can I help anyone?" when an object is detected.
#    - After a short cooldown period, it will show the message for 3 seconds to simulate a greeting interaction.
#    - If an object is locked and either stationary or a simulated "hello" trigger occurs, a follow-up message "Hi there, how can I help you today?" is displayed.
#    - The simulated "hello" trigger can be manually activated by pressing the 'h' key when an object is locked.
#
# 5. Skeleton Detection (Pose Estimation):
#    - Added pose detection to the script using a YOLO pose model (`yolov8n-pose.pt`) to detect human keypoints and skeletons.
#    - Skeleton keypoints are overlaid on the frame to visualize human poses.
#    - If a skeleton point is inside the detected bounding box of an object, it helps confirm that the object is indeed a human.
#
# 6. Foreground Mask and Morphological Filtering:
#    - The foreground mask is generated using the background subtractor (`MOG2`), and morphological operations are applied to reduce noise and improve detection quality.
#    - Dilation and opening operations are used to refine the motion detection output.
#
# 7. Key Interaction:
#    - Press 'h' to simulate a "hello" trigger and activate the simulated interaction system.
#    - Press 'q' to exit the application and close the video capture.
#
# These updates enhance the interaction capabilities of the system by automatically detecting and tracking objects, providing greetings, and focusing attention on locked objects in the scene.
