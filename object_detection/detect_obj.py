import sys
import torch
import cv2
import numpy as np
import lgpio

# Add YOLOv5 repository path to Python's sys.path
sys.path.append('/home/theod/Documents/ME555Capstone/object_detection/yolov5')

# GPIO setup for LED control
LED = 17  # GPIO pin connected to the LED
h = lgpio.gpiochip_open(0)  # Open gpiochip (use '0' based on GPIO chip available)
lgpio.gpio_claim_output(h, LED)

# Load the trained YOLOv5 model (best.pt)
model = torch.load('/home/theod/Documents/ME555Capstone/object_detection/yolov5/weights/best.pt', map_location='cpu')['model'].float()
model.eval()  # Set to evaluation mode

# Initialize webcam (index 0 is the default for most setups)
cap = cv2.VideoCapture(4)  # Use the webcam feed (index 4)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Define the class name (you can update this if needed)
# Assuming you trained for a single class or that your object is the first class
target_class_name = 'defective_part'  # Change this to your trained class name
target_class_id = 0  # If your object is the first class

# Function to detect and label parts using YOLOv5 and control the LED
def detect_and_label_parts(frame):
    global h  # Ensure 'h' is accessible in this function

    # Convert frame to RGB for YOLOv5 model (YOLOv5 expects RGB)
    img = frame[...,::-1]  # Convert BGR to RGB (OpenCV default is BGR)
    img = np.asarray(img)  # Ensure it's a NumPy array
    img = img / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = torch.from_numpy(img).float()  # Convert to tensor

    # Run inference on the image
    with torch.no_grad():
        results = model(img)  # Perform inference on the image

    # Get the results (bounding boxes, class ids, and confidence scores)
    boxes = results.xywh[0][:, :-1]  # Get coordinates of bounding boxes
    confidences = results.xywh[0][:, -1]  # Confidence scores for each detection
    labels = results.names  # Class labels from YOLOv5 model

    # Flag to detect if the target object is present
    object_detected = False

    for box, confidence, label in zip(boxes, confidences, labels):
        if confidence >= 0.5:  # Confidence threshold
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw bounding box
            cv2.putText(frame, f'{label} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Check if the detected object matches the target class
            if label == target_class_name or label == labels[target_class_id]:
                object_detected = True

    # Control the LED based on detection of the target object
    if object_detected:
        lgpio.gpio_write(h, LED, 1)  # Turn LED on if object detected
    else:
        lgpio.gpio_write(h, LED, 0)  # Turn LED off if no object detected

print("Press 'q' to quit the program.")

try:
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        # Detect and label parts using YOLOv5 model
        detect_and_label_parts(frame)

        # Display the frame with annotations
        cv2.imshow("Defect Detection", frame)

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
