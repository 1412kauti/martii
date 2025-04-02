import cv2
import time
import os

print(" Starting webcam capture...")

# Open the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print(" Error: Could not open webcam.")
    exit()

print(" Webcam started... saving frames to 'latest_frame.jpg'")

while True:
    ret, frame = cap.read()
    
    if not ret:
        print(" Error: Could not read frame.")
        break

    # Atomic write: write to temp, then rename
    cv2.imwrite("temp_frame.jpg", frame)
    os.replace("temp_frame.jpg", "latest_frame.jpg")

    # Delay to avoid excessive writes
    time.sleep(0.1)

cap.release()
