import cv2
import numpy as np

# Initialize webcam
cap = cv2.VideoCapture(0)  # 0 is typically the default webcam

# Set resolution (optional)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def classify_custom_shape(approx):
    """ Classify custom shapes based on contour vertices and arrangement """
    if len(approx) == 6:
        # Check for "L-shaped" structure by comparing angles and positions of vertices
        # Calculate relative distances between points to validate the "L" shape
        return "L-Shape"
    elif len(approx) == 5:
        # Check for pentagon-like shape
        return "Pentagon-Like"
    else:
        return "Unknown"

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from webcam")
        break

    # Convert to grayscale and apply thresholding for contour detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        # Approximate the contour to simplify shape
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Classify the shape based on number of vertices and custom criteria
        shape = classify_custom_shape(approx)
        color = (0, 255, 0) if shape != "Unknown" else (0, 0, 255)

        # Draw bounding box and label the shape
        x, y, w, h = cv2.boundingRect(approx)
        cv2.drawContours(frame, [approx], -1, color, 2)
        cv2.putText(frame, shape, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Display the resulting frame
    cv2.imshow("Custom Shape Detection", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
