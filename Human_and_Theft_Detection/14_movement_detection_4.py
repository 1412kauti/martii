import cv2
import time
import numpy as np

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

    # === DRAW ===
    stationary = False

    for objectID, centroid in objects.items():
        for (x, y, w, h) in detections:
            if abs(centroid[0] - (x + w // 2)) < 1 and abs(centroid[1] - (y + h // 2)) < 1:
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
        cv2.putText(frame, "Hey there, can I help, approach me if you need any?", (50, 50),
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
 
 # Overview of the script:
# This script is an enhanced version of the object tracking and interaction system that includes:
# 1. Auto Lock System:
#    - The system automatically locks onto the object with the largest bounding box (assumed to be the closest one).
#    - The "LOCKED" object is highlighted with a RED box.
#    - If the locked object leaves the scene, the system automatically unlocks and resets the process.
#  
# 2. Stationary Detection:
#    - The script tracks the centroid history of the locked object to detect if it becomes stationary.
#    - If the locked object remains in a small area (distance less than 2 pixels), it is considered stationary, and the system flags this status.
#    - The stationary status helps trigger a specific message ("Hi there, how can I help you today?").

# 3. Simulated "Hello" Trigger:
#    - Users can simulate a "hello" trigger by pressing the 'h' key when an object is locked.
#    - This triggers a message ("Hi there, how can I help you today?") if the object is stationary or the "hello" event is triggered.

# 4. Message Display System:
#    - A message "Hey there, can I help, approach me if you need any?" is displayed when an object is detected, and the cooldown period is reset.
#    - A second message "Hi there, how can I help you today?" is displayed when the object is locked and either stationary or a "hello" trigger has occurred.
#    - The cooldown system ensures that these messages are not shown too frequently.

# 5. Object Detection and Tracking:
#    - The `SimpleTracker` class is used to assign unique IDs to detected objects and track them across frames.
#    - Detected objects are assigned a unique ID, and their centroids are used for tracking.

# 6. Foreground Masking and Morphological Operations:
#    - A background subtractor (`MOG2`) is used to detect moving objects in the frame.
#    - Morphological operations (opening and dilation) are applied to improve the detection by reducing noise.

# 7. User Interaction:
#    - Press 'h' to simulate a "hello" trigger while the object is locked.
#    - Press 'q' to quit the program.

# Key Features:
#  - Auto-lock to the closest object.
#  - Stationary detection for triggering specific responses.
#  - Simulated "hello" trigger for user interaction.
#  - Cooldown system for managing messages displayed to the user.
#  - Robust object detection and tracking via SimpleTracker.
