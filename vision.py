import torch
import cv2

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='/Users/theoy/Documents/GitHub/ME555Capstone/weapon_training/yolov5s.pt')

# Initialize the webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform detection
    results = model(frame)

    # Render results on the frame
    annotated_frame = results.render()[0]

    # Display the frame
    cv2.imshow('Weapon Detection', annotated_frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()