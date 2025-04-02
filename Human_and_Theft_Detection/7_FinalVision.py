import cv2
import numpy as np
from ultralytics import YOLO

# --------------------------------
# Load YOLO Face Detector
# --------------------------------
model = YOLO('weights.pt')  # Your trained face detection weights

# --------------------------------
# Initialize SIFT and Harris Detector
# --------------------------------
sift = cv2.SIFT_create()

# --------------------------------
# Webcam Setup
# --------------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Webcam not accessible")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # -------------------------------
    # Step 1: YOLO Face Detection
    # -------------------------------
    results = model(frame)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Face box coordinates

            # Draw Face Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red Box for Face
            cv2.putText(frame, "Face", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # -------------------------------
            # Step 2: Define Region Below Face
            # -------------------------------
            # We assume the body is below the face box
            body_top = y2
            body_bottom = y2 + int((y2 - y1) * 5)  # Extend 5x the face height downward
            body_bottom = min(body_bottom, frame.shape[0] - 1)  # Limit to image bounds

            body_region = frame[body_top:body_bottom, x1:x2]
            if body_region.size == 0:
                continue

            gray_body = cv2.cvtColor(body_region, cv2.COLOR_BGR2GRAY)

            # -------------------------------
            # Step 3: Harris Corner Detection in Body Region
            # -------------------------------
            harris = cv2.cornerHarris(np.float32(gray_body), 2, 3, 0.04)
            harris = cv2.dilate(harris, None)

            # -------------------------------
            # Step 4: SIFT Keypoints in Body Region
            # -------------------------------
            kp, des = sift.detectAndCompute(gray_body, None)

            # Collect all detected points (Harris corners + SIFT keypoints)
            points = []

            # Harris Points
            y_idxs, x_idxs = np.where(harris > 0.01 * harris.max())
            for (x, y) in zip(x_idxs, y_idxs):
                points.append([x, y])

            # SIFT Points
            for point in kp:
                x, y = point.pt
                points.append([int(x), int(y)])

            # -------------------------------
            # Step 5: Create Body Bounding Box if Enough Points
            # -------------------------------
            if len(points) >= 5:  # Avoid creating box if too few points
                points = np.array(points)
                x_min = np.min(points[:, 0]) + x1
                y_min = np.min(points[:, 1]) + body_top
                x_max = np.max(points[:, 0]) + x1
                y_max = np.max(points[:, 1]) + body_top

                # Draw the body bounding box
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # Green box
                cv2.putText(frame, "Body Outline", (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Optional: visualize the points
                for p in points:
                    cv2.circle(frame, (p[0] + x1, p[1] + body_top), 2, (255, 0, 255), -1)  # Magenta dots

    # -------------------------------
    # Step 6: Display Result
    # -------------------------------
    cv2.imshow("Face & Body Outline Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# 1. Load 'Humand Detection' YOLO model I made for face detection using pre-trained weights ('weights.pt')
# 2. Detect faces in each frame with YOLO and draw a red rectangle around detected faces
# 3. Define the region below the face as the body area (extend 5 times the face height downward)
# 4. Convert the body region to grayscale for Harris Corner Detection and apply Harris Corner Detection
# 5. Use SIFT to detect keypoints in the body region, then combine Harris corner points and SIFT keypoints
# 6. If enough points are found, create and draw a bounding box around the body and mark keypoints with magenta dots

