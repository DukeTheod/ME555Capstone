import cv2
import numpy as np

# Initialize webcam (0 is usually the default camera index)
cap = cv2.VideoCapture(4)

# Define the color ranges for detecting red and green (HSV format)
lower_red1 = np.array([0, 150, 150])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 150, 150])
upper_red2 = np.array([180, 255, 255])
lower_green = np.array([40, 50, 50])
upper_green = np.array([80, 255, 255])

# Function to detect and label parts
def detect_and_label_parts(frame):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create masks for detecting red and green colors
    red_mask1 = cv2.inRange(hsv_frame, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv_frame, lower_red2, upper_red2)
    red_mask = red_mask1 | red_mask2  # Combine both red masks
    green_mask = cv2.inRange(hsv_frame, lower_green, upper_green)

    # Find contours for red parts
    contours_red, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours_red:
        area = cv2.contourArea(contour)
        if area > 500:  # Adjust the area threshold as needed
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, "False", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Find contours for green parts
    contours_green, _ = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours_green:
        area = cv2.contourArea(contour)
        if area > 500:  # Adjust the area threshold as needed
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "True", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

print("Press 'q' to quit the program.")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Detect and label parts based on color
    detect_and_label_parts(frame)

    # Display the frame with annotations
    cv2.imshow("Part Detection", frame)

    # Exit loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
