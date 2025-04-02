import cv2
import time
import os
from ultralytics import YOLO

# Load YOLOv8 model (replace 'weights.pt' with your own)
model = YOLO("weights.pt")

frame_path = "latest_frame.jpg"
last_mod_time = 0

print(" Waiting for new frames...")

while True:
    if os.path.exists(frame_path):
        mod_time = os.path.getmtime(frame_path)

        # Skip if frame hasn't changed
        if mod_time == last_mod_time:
            time.sleep(0.05)
            continue

        last_mod_time = mod_time

        frame = cv2.imread(frame_path)

        # Skip if frame couldn't be read (e.g., still being written)
        if frame is None:
            time.sleep(0.05)
            continue

        # Run YOLO detection
        results = model(frame)

        # Draw bounding boxes
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                label = result.names[int(box.cls[0])]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Show output
        cv2.imshow("YOLO Detection", frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        time.sleep(0.05)

cv2.destroyAllWindows()
