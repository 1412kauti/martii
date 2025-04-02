import threading
from qibullet import SimulationManager
import pybullet as p
from qibullet import PepperVirtual, Camera
import cv2
from tensorflow.keras import models
import numpy as np
from PIL import Image
import os
import time
import sys
import math

# Create a threading lock to prevent concurrent access to shared resources (e.g., camera frames)
lock = threading.Lock()

def predict_object(img, model, labels):
    """
    Process the input image to match the model's expected format,
    perform prediction, and return the predicted label and confidence.
    """
    # Resize image to 224x224 pixels as required by the model
    resized_img = cv2.resize(img, (224, 224))
    # Normalize the image values to [0, 1]
    normalized_img = resized_img / 255.0
    # Expand dimensions to add batch size dimension
    normalized_img = np.expand_dims(normalized_img, 0)
    # Get predictions from the model
    y_pred = model.predict(normalized_img, verbose=0)
    predicted_index = np.argmax(y_pred[0])
    confidence = y_pred[0][predicted_index]
    predicted_label = labels[predicted_index]
    return predicted_label, confidence

def prepare_images(images_path, output_path):
    """
    Create the output directory (if needed) and save the original images
    as textures without any resizing or padding.
    """
    os.makedirs(output_path, exist_ok=True)
    # Process only the first 3 images
    for i, img_path in enumerate(images_path[:3]):
        img = Image.open(img_path)
        # Save the raw image directly to output_path with a new name
        img.save(os.path.join(output_path, f"texture_{i}.jpg"))

def spawn_textured_walls(output_path, fixed_positions):
    """
    Create visual walls in the simulation and apply textures from the saved images.
    """
    for i in range(3):
        # Load texture image from disk
        texture_id = p.loadTexture(os.path.join(output_path, f"texture_{i}.jpg"))
        # Create a visual shape representing the wall
        visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[1.5, 0.01, 1.5])
        # Create a multi-body object for the wall at a fixed position
        picture = p.createMultiBody(baseVisualShapeIndex=visual_shape, basePosition=fixed_positions[i])
        # Apply the texture to the wall
        p.changeVisualShape(picture, -1, textureUniqueId=texture_id)

def initialize_pepper(simulation_manager, client_id):
    """
    Spawn the Pepper robot in the simulation at the origin with a ground plane.
    """
    pepper = simulation_manager.spawnPepper(client_id, translation=[0, 0, 0], spawn_ground_plane=True)
    return pepper

def pepperCamera(pepper, lock):
    """
    Continuously capture frames from Pepper's top camera and display them.
    This function runs in a separate thread.
    """
    # Subscribe to the top camera with QVGA resolution and 15 fps
    pepper_camera = pepper.subscribeCamera(PepperVirtual.ID_CAMERA_TOP, resolution=Camera.K_QVGA, fps=15.0)
    try:
        while True:
            with lock:
                # Retrieve the current camera frame
                img = pepper.getCameraFrame(pepper_camera)
                # Display the frame in a window titled "Camera Feed"
                cv2.imshow("Camera Feed", img)
                cv2.waitKey(1)
    except KeyboardInterrupt:
        # Graceful exit on keyboard interrupt
        pass
    return pepper_camera

def load_model_and_labels(model_path):
    """
    Load the pre-trained Keras model and define the associated labels.
    """
    labels = [
        "Pepsi", "milk", "Coffee Nescafe",
        "SNICKERS", "Cheese", "water bottle", "walnuts"
    ]
    model = models.load_model(model_path)
    return model, labels

def detect(pepper, pepper_camera, model, labels):
    """
    Continuously capture frames from the camera, run object detection,
    annotate the image with the detected object and its confidence,
    and display the result. Runs in a separate thread.
    """
    while True:
        with lock:
            img = pepper.getCameraFrame(pepper_camera)
            predicted_label, confidence = predict_object(img, model, labels)
            display_text = "{} ({:.2f})".format(predicted_label, confidence)
            # Draw the detection result on the image
            cv2.putText(img, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.imshow("Object Detection", img)
            # Exit detection loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return

def move_and_detect(pepper):
    """
    Command the Pepper robot to move through a series of predefined positions.
    At each movement step, the robot moves forward and performs a turn.
    """
    fixed_positions = [
        [0,  2, 1],
        [3,  2, 1],
        [7,  2, 1],
        [10, 2, 1]
    ]
    for pos in range(1, len(fixed_positions)):
        # Calculate distance to move based on x-coordinate differences
        dist = abs(fixed_positions[pos][0] - fixed_positions[pos - 1][0])
        # Move forward in x-direction at a fixed speed
        pepper.move(0.35, 0, 0)
        time.sleep(dist / (0.35 * 0.5))  # Adjust sleep time based on distance and speed
        
        # Rotate left (assumed to be 1 radian per second)
        pepper.move(0, 0, 1)
        time.sleep(2 * (math.radians(90)))  # Wait for the rotation to complete
        
        # Rotate back to original orientation
        pepper.move(0, 0, -1)
        time.sleep(2 * (math.radians(90)))  # Wait for the rotation to complete

def main():
    """
    Main function to set up the simulation environment, prepare textured walls,
    initialize the Pepper robot, start camera and detection threads, and perform robot movement.
    """
    # Define base paths and directories
    base_path = "/home/mousa08/demo/"
    textures_path = os.path.join(base_path, "textures")
    output_path = os.path.join(base_path, "pictures_sim")
    
    # List of image file paths
    images_path = [
        os.path.join(textures_path, "photo1.jpg"),
        os.path.join(textures_path, "photo2.jpg"),
        os.path.join(textures_path, "photo3.jpg"),
        os.path.join(textures_path, "photo4.jpg")
    ]
    
    # Fixed positions for placing textured walls in the simulation
    fixed_positions = [
        [3,  2, 1],
        [7,  2, 1],
        [10, 2, 1]
    ]
    
    # Launch the simulation with GUI enabled
    simulation_manager = SimulationManager()
    client_id = simulation_manager.launchSimulation(gui=True)
    
    # Prepare images by saving them directly for textures
    prepare_images(images_path, output_path)
    
    # Create walls with the prepared textures
    spawn_textured_walls(output_path, fixed_positions)
    
    # Initialize the Pepper robot in the simulation
    pepper = initialize_pepper(simulation_manager, client_id)
    
    # Start the camera thread to display live camera feed
    camera_thread = threading.Thread(target=pepperCamera, args=(pepper, lock), daemon=True)
    camera_thread.start()
    
    # Subscribe to the top camera for detection purposes
    pepper_camera = pepper.subscribeCamera(PepperVirtual.ID_CAMERA_TOP, resolution=Camera.K_QVGA, fps=15.0)
    
    # Load the Keras model and associated labels for object detection
    print("Loading keras models")
    model, labels = load_model_and_labels(os.path.join(base_path, "model.savedmodel"))
    
    # Start the detection thread to process and display object detection results
    detect_thread = threading.Thread(target=detect, args=(pepper, pepper_camera, model, labels), daemon=True)
    detect_thread.start()
    
    try:
        # Wait a few seconds before starting robot movement
        print("Waiting 3 seconds before starting movement...")
        time.sleep(3)
        # Execute robot movement and detection
        move_and_detect(pepper)
        print("Pepper finished moving and detecting.")
    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        # Clean up the simulation and close OpenCV windows
        simulation_manager.stopSimulation(client_id)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
