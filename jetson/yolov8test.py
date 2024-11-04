import cv2
from ultralytics import YOLO

# Load YOLOv8 model (e.g., yolov8n for YOLOv8 Nano model, yolov8s for Small, etc.)
model = YOLO('yolov8n.pt')  # yolov8n.pt is the Nano version; you can use yolov8s.pt, yolov8m.pt, etc.

# Initialize webcam
cap = cv2.VideoCapture(0)  # 0 is typically the default webcam

# Set resolution (optional)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from webcam")
        break

    # Run YOLOv8 model on the frame
    results = model(frame)

    # Process results
    for result in results:
        boxes = result.boxes  # bounding boxes
        for box in boxes:
            # Extract box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get integer box coordinates
            confidence = box.conf[0]  # Confidence score
            class_id = int(box.cls[0])  # Class ID
            label = f"{model.names[class_id]}: {confidence:.2f}"

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow("YOLOv8 Object Detection", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
