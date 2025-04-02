import time
import multiprocessing

def check_face_detection(face_detected, duration=3):
    detected_time = 0

    while True:
        if face_detected.value:
            if detected_time == 0:
                detected_time = time.time()  # Start timer
            elif time.time() - detected_time >= duration:
                print("Face detected for", duration, "seconds!")
                detected_time = 0  # Reset timer after printing
        else:
            detected_time = 0  # Reset if face disappears

        time.sleep(0.5)  # Check every 0.5 seconds

if __name__ == "__main__":
    face_detected = multiprocessing.Value('b', False)  # Shared boolean variable
    check_face_detection(face_detected)
