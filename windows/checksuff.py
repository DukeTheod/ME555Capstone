import torch
import cv2
import numpy as np
from pathlib import Path

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)  # using the small model for quick performance

# Set the device to GPU if available, otherwise use CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Initialize webcam feed
cap = cv2.VideoCapture(0)  # 0 is usually the default webcam; change if needed

# Check if webcam opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to quit the webcam feed.")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Perform object detection
    results = model(frame)

    # Render results
    annotated_frame = np.squeeze(results.render())  # Rendered frame is returned as a numpy array

    # Display the frame with annotations
    cv2.imshow("YOLOv5 Webcam Detection", annotated_frame)

    # Exit loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
