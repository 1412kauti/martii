import cv2
import time

# Load video or webcam
cap = cv2.VideoCapture("shopping_test_1.mp4")  # change to 0 for webcam

# Background Subtractor
backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50)

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

    # Contours
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    object_detected = False

    for cnt in contours:
        if cv2.contourArea(cnt) > 500:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            object_detected = True

    current_time = time.time()

    # Trigger message only if cooldown expired
    if object_detected and (not message_active or (current_time - last_message_time) > cooldown_seconds):
        message_active = True
        last_message_time = current_time

    # Display message for 3 seconds after triggering
    if message_active and (current_time - last_message_time < 3):
        cv2.putText(frame, "Hello, can I help?", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
    elif message_active and (current_time - last_message_time >= cooldown_seconds):
        message_active = False  # reset for next opportunity

    # Show
    cv2.imshow("Moving Object Detection", frame)
    cv2.imshow("Foreground Mask", fg_mask)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Changes in the script:

# 1. Video Source:
#   - Loads the video using `cv2.VideoCapture("shopping_test_1.mp4")`, and you can switch to webcam input by using `0`.

# 2. Background Subtraction:
#   - Uses `cv2.createBackgroundSubtractorMOG2()` for background subtraction with a history of 500 frames and a sensitivity threshold (`varThreshold=50`).

# 3. Morphological Filtering:
#   - Applies morphological operations to clean the foreground mask:
#     - `cv2.MORPH_OPEN` is used to remove small noise, followed by dilation to fill small gaps, both using a (5, 5) elliptical kernel for 2 iterations each.

# 4. Contour Detection:
#   - Detects contours in the foreground mask using `cv2.findContours()` and draws bounding boxes around contours with an area greater than 500 pixels.

# 5. Cooldown System:
#   - A cooldown mechanism is implemented to prevent the trigger message from being shown repeatedly too quickly.
#   - The system tracks the last message time and ensures the message is only shown if a specified cooldown (`cooldown_seconds=10`) has passed since the last message.
#   - If an object is detected and the cooldown has expired, a message ("Hello, can I help?") is displayed on the frame for 3 seconds.

# 6. Display:
#   - The video frame with bounding boxes and any displayed messages are shown in a window titled "Moving Object Detection".
#   - The foreground mask (binary mask showing detected objects) is displayed in another window called "Foreground Mask".

# 7. Exit Condition:
#   - The loop exits if the 'q' key is pressed.
