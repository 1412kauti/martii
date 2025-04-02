# https://learnopencv.com/moving-object-detection-with-opencv/
import cv2

# Load video (or replace with 0 for webcam)
cap = cv2.VideoCapture("shopping_test_1.mp4")  # <-- change to your video file

# Create Background Subtractor
backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Background Subtraction
    fg_mask = backSub.apply(frame)

    # Morphological filtering to clean noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)

    # Contour Detection
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt) > 500:  # Minimum area filter
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display
    cv2.imshow("Moving Object Detection", frame)
    cv2.imshow("Foreground Mask", fg_mask)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Changes in the script:

# 1. Video Source:
#   - The video source is loaded using `cv2.VideoCapture("shopping_test_1.mp4")`. You can replace it with 0 to use the webcam.

# 2. Background Subtraction:
#   - Utilizes OpenCV's `cv2.createBackgroundSubtractorMOG2()` for moving object detection by comparing the current frame to a background model.
#   - `history=500` stores the last 500 frames, and `varThreshold=50` adjusts the sensitivity of background subtraction.

# 3. Morphological Filtering:
#   - Applies morphological operations (`cv2.MORPH_OPEN` and `cv2.dilate`) to clean up noise in the foreground mask.
#   - A kernel of size (5, 5) is used to perform the operations, and they are applied for 2 iterations each.

# 4. Contour Detection:
#   - `cv2.findContours()` is used to detect contours in the foreground mask.
#   - Only contours with an area greater than 500 pixels are considered (to filter out small noise).

# 5. Bounding Box:
#   - For each valid contour, a bounding box is drawn using `cv2.boundingRect()`, marking the detected moving object in green on the original frame.

# 6. Display:
#   - The original frame with detected moving objects is displayed in a window named "Moving Object Detection".
#   - The foreground mask (binary mask showing detected objects) is displayed in a separate window named "Foreground Mask".

# 7. Exit Condition:
#   - The loop breaks and the program exits if the 'q' key is pressed.
