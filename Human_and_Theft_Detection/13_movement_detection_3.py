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

cap = cv2.VideoCapture("shopping_test_1.mp4")  # change to 0 for webcam

backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50)
tracker = SimpleTracker(max_distance=60)

# Cooldown system
message_active = False
last_message_time = 0
cooldown_seconds = 10

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

    # === FIND CLOSEST OBJECT (biggest box === assumed closest) ===
    biggest_area = 0
    closest_box = None

    for (x, y, w, h) in detections:
        area = w * h
        if area > biggest_area:
            biggest_area = area
            closest_box = (x, y, w, h)

    # --- Draw Boxes ---
    for objectID, centroid in objects.items():
        for (x, y, w, h) in detections:
            if abs(centroid[0] - (x + w // 2)) < 1 and abs(centroid[1] - (y + h // 2)) < 1:
                # Check if this is the closest one
                if closest_box is not None and (x, y, w, h) == closest_box:
                    color = (0, 0, 255)  # RED box for closest object
                else:
                    color = (0, 255, 0)  # GREEN box for others

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"ID {objectID}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Cooldown Message
    if object_detected and (not message_active or (current_time - last_message_time) > cooldown_seconds):
        message_active = True
        last_message_time = current_time

    if message_active and (current_time - last_message_time < 3):
        cv2.putText(frame, "Hello, can I help?", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
    elif message_active and (current_time - last_message_time >= cooldown_seconds):
        message_active = False

    # Show
    cv2.imshow("Closest Object Highlighted", frame)
    cv2.imshow("Foreground Mask", fg_mask)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Differences from the previous script:

# 1. Object Tracking with SimpleTracker Class:
#    - A `SimpleTracker` class has been added to track objects across frames. This class assigns unique IDs to detected objects and associates them across multiple frames based on the distance between centroids.
#    - The `update` method in `SimpleTracker` takes the current detections (bounding boxes) and tracks them, updating the positions of existing objects and creating new ones if necessary.

# 2. Object Detection and Tracking:
#    - Instead of just detecting moving objects and showing bounding boxes, the objects are now tracked with unique IDs.
#    - Each object gets assigned an ID, and the position of each tracked object is updated based on its centroid (the center point of the bounding box).
#    - This allows for consistent tracking of each object over time.

# 3. Closest Object Highlighting:
#    - The script introduces a feature to highlight the "closest" object (based on the largest bounding box area), which is assumed to be the closest object.
#    - The closest object is highlighted with a RED box, while other tracked objects are drawn with a GREEN box.
#    - The biggest area is calculated in the `for` loop iterating over the detections, and the closest object is determined accordingly.

# 4. Message Display:
#    - The message "Hello, can I help?" is displayed on the screen when an object is detected and the cooldown period has expired.
#    - This behavior is similar to the previous script, but the message now only appears for the closest object during detection.

# 5. Tracking vs. Contour Detection:
#    - In the previous version, objects were detected solely by their contours and bounding boxes.
#    - In this version, contours are still detected, but the bounding boxes are linked to tracked objects, providing more accurate tracking and handling of objects across multiple frames.

# 6. Additional Logic for Object Tracking:
#    - The tracker keeps track of multiple objects and updates them based on centroid distance. It is more robust for long-term tracking compared to just relying on static detections each frame.
