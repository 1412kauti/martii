import cv2
import multiprocessing

def face_detection(face_detected):
    print("Starting webcam...")

    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture("shopping_test_1.mp4")

    if not cap.isOpened():
        print("Error: Cannot open webcam.")
        return

    print("Webcam opened. Detecting faces...")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        if len(faces) > 0:
            face_detected.value = True
        else:
            face_detected.value = False

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Face Detection (Haar)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    face_detected = multiprocessing.Value('b', False)  # Boolean shared variable
    process = multiprocessing.Process(target=face_detection, args=(face_detected,))
    process.start()

# Explanation:
# This script performs real-time face detection using OpenCV’s Haar cascade classifier 
# and runs the detection in a separate process using multiprocessing.

# 1. Haar Cascade Classifier:
#    - Haar cascades are pre-trained object detection models that use edge and texture features.
#    - The "haarcascade_frontalface_default.xml" file contains data for frontal face detection.
#    - The detectMultiScale function scans an image at multiple scales to find faces.
#      - scaleFactor=1.1: Reduces the image by 10% at each scale.
#      - minNeighbors=5: Requires at least 5 neighboring rectangles to confirm a detection.

# 2. Video Capture:
#    - cv2.VideoCapture("shopping_test_1.mp4") opens a video file instead of a webcam.
#    - If the file cannot be opened, an error message is displayed.

# 3. Grayscale Conversion:
#    - Haar cascades perform better on grayscale images, so the frame is converted using cvtColor.

# 4. Face Detection and Processing:
#    - If a face is detected, a rectangle is drawn around it.
#    - The `face_detected` shared variable is updated to indicate whether a face is present.

# 5. Multiprocessing:
#    - The `multiprocessing` module is used to run face detection in a separate process.
#    - `multiprocessing.Value('b', False)` creates a shared boolean variable.
#    - A separate process is started with `multiprocessing.Process(target=face_detection, args=(face_detected,))`.

# 6. Display and Exit:
#    - The processed video is displayed in a window labeled "Face Detection (Haar)."
#    - The program exits when 'q' is pressed, releasing resources properly.