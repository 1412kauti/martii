import cv2
import numpy as np

# Load Haar Cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Webcam not accessible")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Step 1: Face Detection
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Step 2: Eye Detection inside face
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            eye_gray = roi_gray[ey:ey+eh, ex:ex+ew]
            eye_color = roi_color[ey:ey+eh, ex:ex+ew]

            # Step 3: Threshold to find pupil
            _, thresh = cv2.threshold(eye_gray, 50, 255, cv2.THRESH_BINARY_INV)

            # Step 4: Find largest contour = pupil
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

            if contours:
                (cx, cy), radius = cv2.minEnclosingCircle(contours[0])
                center = (int(cx), int(cy))

                # Step 5: Draw
                cv2.circle(eye_color, center, int(radius), (0, 0, 255), 2)
                cv2.circle(eye_color, center, 2, (255, 0, 0), -1)  # pupil dot

    cv2.imshow("Simple Live Pupil Tracker", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# 1. Load Haar Cascades for face and eye detection
#    - Haar cascades are pre-trained classifiers used for object detection, such as detecting faces and eyes.
#    - 'haarcascade_frontalface_default.xml' and 'haarcascade_eye.xml' are XML files that define these classifiers.

# 2. Capture frames from the webcam and convert them to grayscale for detection
#    - OpenCV's VideoCapture reads video from the webcam.
#    - Convert the captured frames to grayscale using `cv2.cvtColor()` because Haar cascades work on grayscale images for better performance.

# 3. Detect faces in the frame using the face cascade, and for each face, define a region of interest (ROI)
#    - `detectMultiScale()` is used to find faces in the grayscale image.
#    - For each detected face, extract the portion of the image containing the face (the region of interest).

# 4. Detect eyes within the detected face using the eye cascade
#    - Once the face region is obtained, the same method (`detectMultiScale()`) is used to find eyes within the face ROI.
#    - This step ensures that eyes are only detected within the context of the detected face.

# 5. For each detected eye, threshold the eye region to find the pupil by applying binary inverse thresholding
#    - Convert the eye image into a binary image using `cv2.threshold()`. This step enhances the pupil's visibility by setting a specific threshold value for pixel intensity.
#    - The `THRESH_BINARY_INV` flag inverts the binary image, making the pupil (usually darker) become white and everything else black.

# 6. Find the largest contour in the thresholded eye region, which corresponds to the pupil
#    - `cv2.findContours()` finds all contours (connected boundary lines) in the binary image.
#    - The largest contour, which should correspond to the pupil, is identified by sorting contours based on area (`cv2.contourArea()`).

# 7. Draw a circle around the pupil and mark the center with a smaller dot to track it
#    - `cv2.minEnclosingCircle()` is used to fit a circle around the largest contour (the pupil).
#    - `cv2.circle()` is used to draw the circle, and a smaller dot at the center of the pupil is drawn to indicate the precise location.

