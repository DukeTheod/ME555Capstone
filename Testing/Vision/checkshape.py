import cv2
import torch
import pyrealsense2 as rs
from ultralytics import YOLO

# Load the YOLOv5 model (replace 'yolov5s.pt' with your custom-trained model file)
model = YOLO("yolov5su.pt")  # Replace with your trained model path

# Initialize RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # Adjust resolution and FPS if needed
pipeline.start(config)

try:
    while True:
        # Capture frames from the RealSense camera
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Convert RealSense frame to numpy array
        frame = np.asanyarray(color_frame.get_data())

        # Run YOLO model on the frame
        results = model(frame)

        # Process and display the results
        for detection in results.xyxy[0]:  # Bounding box coordinates and confidence
            x1, y1, x2, y2, confidence, cls = map(int, detection)
            label = model.names[cls]

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Check if the detected object is a "defective part" (modify label accordingly)
            if label == "defective_part":  # Replace with the actual label used in training
                cv2.putText(frame, "DEFECT DETECTED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # Display the resulting frame
        cv2.imshow("Defect Detection", frame)

        # Break the loop with 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Stop the RealSense pipeline and close OpenCV window
    pipeline.stop()
    cv2.destroyAllWindows()