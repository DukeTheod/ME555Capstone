import torch
import cv2
import numpy as np
import lgpio

# GPIO setup for LED control
LED = 17  # GPIO pin connected to the LED
h = lgpio.gpiochip_open(0)  # Open gpiochip (use '0' based on GPIO chip available)
lgpio.gpio_claim_output(h, LED)

# Load the YOLO model (using PyTorch)
model_path = '/home/theod/Documents/ME555Capstone/object_detection/best.pt'  # Replace with the actual path to your YOLO .pt model
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
model.eval()  # Set the model to evaluation mode

# Initialize webcam (index 0 is the default for most setups)
cap = cv2.VideoCapture(4)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Function to detect parts and control the LED
def detect_and_label_parts(frame):
    global h  # Ensure 'h' is accessible in this function

    # Convert the frame to a format that PyTorch can process
    results = model(frame)  # Run inference
    detections = results.xyxy[0].numpy()  # Get detections as NumPy array: [x1, y1, x2, y2, confidence, class_id]

    object_detected = False

    for detection in detections:
        x1, y1, x2, y2, confidence, class_id = map(float, detection[:6])  # Extract detection details
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
