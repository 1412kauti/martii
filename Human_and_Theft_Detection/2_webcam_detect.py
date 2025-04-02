import cv2
import requests
import json
import numpy as np

# API Details
api_url = "https://detect.roboflow.com/people-detection-o4rdr/8"
api_key = "vH4U3BcmyG0u1B7zDloX"

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame to reduce API load (adjust as needed)
    frame_resized = cv2.resize(frame, (640, 480))
    
    # Encode frame as JPEG
    _, img_encoded = cv2.imencode(".jpg", frame_resized)
    
    # Send frame to Roboflow API
    response = requests.post(
        f"{api_url}?api_key={api_key}",
        files={"file": img_encoded.tobytes()}
    )
    
    # Parse API response
    try:
        data = response.json()
    except:
        continue

    # Draw bounding boxes on frame
    for prediction in data.get("predictions", []):
        x, y, w, h = prediction["x"], prediction["y"], prediction["width"], prediction["height"]
        x1, y1 = int(x - w / 2), int(y - h / 2)
        x2, y2 = int(x + w / 2), int(y + h / 2)
        
        # Draw rectangle
        cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Add label
        cv2.putText(frame_resized, "Person", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("People Detection", frame_resized)

    # Break on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

# This script runs the same way as 1_People_detection_API.py but instead of intaking an image it takes live webcam feed if cap = cv2.VideoCapture(0)
