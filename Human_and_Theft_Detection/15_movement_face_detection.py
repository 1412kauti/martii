import cv2
import time
import numpy as np
from ultralytics import YOLO

# ----------------------------------------
# Load YOLO person detector
# ----------------------------------------
face_model = YOLO('weights_person_detection_2.pt')  # <-- replace with your trained weights

# ----------------------------------------
# Background Subtractor + Tracker
# ----------------------------------------
backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50)

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

tracker = SimpleTracker(max_distance=60)

# ----------------------------------------
# Lock system
# ----------------------------------------
lock_id = None
lock_centroid_history = []

# Cooldown system
message_active = False
last_message_time = 0
cooldown_seconds = 10

# Simulated hello
simulate_hello_triggered = False
hello_trigger_time = 0

# ----------------------------------------
# Video source
# ----------------------------------------
cap = cv2.VideoCapture("shopping_test_1.mp4")  # Or 0 for webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    fg_mask = backSub.apply(frame)

    # ------------------------
    # YOLO Person Detection
    # ------------------------
    results = face_model(frame)
    detected_people = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detected_people.append((x1, y1, x2, y2))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # optional: draw YOLO box

    # ------------------------
    # Filter motion only inside YOLO boxes
    # ------------------------
    detections = []

    for (x1, y1, x2, y2) in detected_people:
        roi = fg_mask[y1:y2, x1:x2]
        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) > 500:
                cx, cy, cw, ch = cv2.boundingRect(cnt)
                detections.append((x1 + cx, y1 + cy, cw, ch))  # map ROI coords to full frame

    # ------------------------
    # Tracker update
    # ------------------------
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

    # ------------------------
    # Drawing
    # ------------------------
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

    # === Message ===
    if object_detected and (not message_active or (current_time - last_message_time) > cooldown_seconds):
        message_active = True
        last_message_time = current_time

    if message_active and (current_time - last_message_time < 3):
        cv2.putText(frame, "Hey there, can I help, approach me if you need any?", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    elif message_active and (current_time - last_message_time >= cooldown_seconds):
        message_active = False

    if simulate_hello_triggered and (current_time - hello_trigger_time > 5):
        simulate_hello_triggered = False

    if lock_id is not None and (stationary or simulate_hello_triggered):
        cv2.putText(frame, "Hi there, how can I help you today?", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

    # === Show
    cv2.imshow("YOLO-filtered Lock Tracker", frame)
    cv2.imshow("Foreground Mask", fg_mask)

    key = cv2.waitKey(30) & 0xFF
    if key == ord('h') and lock_id is not None:
        simulate_hello_triggered = True
        hello_trigger_time = time.time()
        print("Simulated: Heard 'hello'")

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# This model iterates on 14_movement_detection_4 by including a person detection model i made on YOLO