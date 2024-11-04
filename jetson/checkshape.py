import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 model (pre-trained on general objects, customize if needed)
model = YOLO('yolov8n.pt')  # yolov8n.pt is the Nano version; adjust if using a custom shape model

# Initialize webcam
cap = cv2.VideoCapture(0)  # Replace with the conveyor camera if different

# Set resolution (optional)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def classify_shape(approx):
    """ Classify shape based on the number of vertices in the contour approximation """
    if len(approx) == 3:
        return "Triangle"
    elif len(approx) == 4:
        aspect_ratio = float(cv2.boundingRect(approx)[2]) / cv2.boundingRect(approx)[3]
        return "Square" if 0.95 <= aspect_ratio <= 1.05 else "Rectangle"
    elif len(approx) > 5:
        return "Circle"
    else:
        return "Unknown"

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from webcam")
        break

    # Run YOLOv8 to locate objects on the conveyor belt
    results = model(frame)

    # Process YOLOv8 detections
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get box coordinates
            cropped_frame = frame[y1:y2, x1:x2]  # Crop to bounding box

            # Convert to grayscale and apply thresholding for contour detection
            gray = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                # Approximate the contour to reduce the number of vertices
                epsilon = 0.04 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                # Classify shape
                shape = classify_shape(approx)
                label = f"Shape: {shape}"

                # Check if the shape is "right" or "wrong"
                if shape == "Circle":  # Replace "Circle" with the desired shape to mark as correct
                    status = "Right"
                    color = (0, 255, 0)  # Green for "Right"
                else:
                    status = "Wrong"
                    color = (0, 0, 255)  # Red for "Wrong"

                # Draw bounding box and labels
                cv2.drawContours(cropped_frame, [approx], -1, color, 2)
                cv2.putText(cropped_frame, f"{label} - {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Display the cropped frame with annotations
            frame[y1:y2, x1:x2] = cropped_frame

    # Show the entire frame with all detected objects
    cv2.imshow("Conveyor Belt Shape Detection", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
