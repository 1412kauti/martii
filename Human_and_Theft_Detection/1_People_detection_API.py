import requests
import cv2
import numpy as np
from PIL import Image
from io import BytesIO

api_url = "https://detect.roboflow.com/people-detection-o4rdr/8"
api_key = "vH4U3BcmyG0u1B7zDloX"
image_path = "asda.webp"

# Open image file in binary mode
with open(image_path, "rb") as image_file:
    image_data = image_file.read()

# Prepare the request
response = requests.post(
    f"{api_url}?api_key={api_key}",
    files={"file": image_data}
)

# Load the image
image = cv2.imread(image_path)

# Parse response
if response.status_code == 200:
    results = response.json()
    for obj in results.get("predictions", []):
        x, y, w, h = int(obj["x"]), int(obj["y"]), int(obj["width"]), int(obj["height"])
        cv2.rectangle(image, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), (0, 255, 0), 2)
        cv2.putText(image, obj["class"], (x - w // 2, y - h // 2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Display image
cv2.imshow("Detected People", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# This script detects people in a grocery store using a pre-trained model from Roboflow's Universe.
# It sends an image to the API, retrieves detection results, and draws bounding boxes around detected individuals.
# The final output is displayed in an OpenCV window with labeled detections, helping in tracking foot traffic.