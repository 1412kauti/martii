import cv2
from ultralytics import YOLO

# Load YOLO model
face_model = YOLO('theft_detection.pt')  # Replace with your trained weights

# Open video file
cap = cv2.VideoCapture("theft.mp4")
if not cap.isOpened():
    print("Error: Video file not accessible")
    exit()

# Define colors and labels for different classes
class_labels = {0: "Normal", 1: "Possible Theft"}
colors = {0: (0, 255, 0), 1: (0, 0, 255)}  # Green for Normal, Red for Possible Theft

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run YOLO detection
    results = face_model(frame)
    
    # Draw bounding boxes
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            color = colors.get(class_id, (255, 255, 255))  # Default to white if class unknown
            label = class_labels.get(class_id, "Unknown")
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    cv2.imshow("Theft Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()