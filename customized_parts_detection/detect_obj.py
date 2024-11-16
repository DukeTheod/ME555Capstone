import torch
import cv2
import sys
import numpy as np
import lgpio
from pathlib import Path
sys.path.append('/home/theod/Documents/ME555Capstone/object_detection/yolov5')
from models.common import DetectMultiBackend  # Ensure you have YOLOv5 in the same directory

# GPIO setup for LED control
LED = 17  # GPIO pin connected to the LED
h = lgpio.gpiochip_open(0)  # Open gpiochip (use '0' based on GPIO chip available)
lgpio.gpio_claim_output(h, LED)

# Load the YOLO model (using DetectMultiBackend from YOLOv5)
model_path = Path('/home/theod/Documents/ME555Capstone/object_detection/yolov5/runs/train/exp2/weights/best.pt')
device = 'cpu'  # Use 'cuda' if a GPU is available
model = DetectMultiBackend(model_path, device=device)
model.eval()

# Initialize webcam (port 4 as specified)
cap = cv2.VideoCapture(4)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # Shrink resolution to half of 640x480
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

def detect_and_label_parts(frame):
    global h  # Ensure 'h' is accessible in this function

    # Preprocess the frame for the model
    img = cv2.resize(frame, (640, 640))  # Resize frame to match model's expected input size
    img = img[:, :, ::-1]  # Convert BGR to RGB
    img = np.transpose(img, (2, 0, 1))  # Rearrange dimensions to (C, H, W)
    img = np.ascontiguousarray(img)  # Ensure memory layout is contiguous
    img = torch.from_numpy(img).float()  # Convert to PyTorch tensor
    img /= 255.0  # Normalize to [0, 1]
    img = img.unsqueeze(0)  # Add batch dimension

    # Run the frame through the model
    results = model(img)

    # Check the format of results and extract detections
    if isinstance(results, list):
        detections = results[0].cpu().numpy()  # If results is a list, extract first item
    else:
        detections = results[0].cpu().numpy()  # Fallback to first item in case of tensor output

    # Debug: Print the shape and content of detections
    print("Detections shape:", detections.shape)
    print("Detections content:", detections)

    object_detected = False

    # Process each detection in the batch
    for detection in detections[0]:  # Iterate through the second dimension
        if len(detection) >= 6:
            x1, y1, x2, y2, confidence, class_id = map(float, detection[:6])
            class_id = int(class_id)

            # Check if the confidence and class ID meet the conditions
            if confidence > 0.5 and class_id == 0:  # Adjust '0' to your target class ID if needed
                object_detected = True
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f"Object Detected ({confidence:.2f})", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Control the LED based on object detection
    if object_detected:
        lgpio.gpio_write(h, LED, 1)  # Turn LED on if the target object is detected
    else:
        lgpio.gpio_write(h, LED, 0)  # Turn LED off if no object is detected

print("Press 'q' to quit the program.")

try:
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        # Detect and label parts using PyTorch YOLO model
        detect_and_label_parts(frame)

        # Display the frame with annotations
        cv2.imshow("Part Detection", frame)

        # Exit loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Ensure the LED is off and clean up GPIO
    if h is not None:
        lgpio.gpio_write(h, LED, 0)
        lgpio.gpiochip_close(h)

    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()
