import cv2
import numpy as np
import lgpio
import time

# GPIO setup for LED control
LED = 17  # GPIO pin connected to the LED
h = lgpio.gpiochip_open(0)  # Open gpiochip (use '0' or '10' based on availability)
lgpio.gpio_claim_output(h, LED)

# Initialize webcam (index might change based on your setup)
cap = cv2.VideoCapture(4)

# Define the color ranges for detecting red (HSV format)
lower_red1 = np.array([0, 150, 150])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 150, 150])
upper_red2 = np.array([180, 255, 255])

# Function to detect and label parts and control the LED if red is detected
def detect_and_label_parts(frame):
    global h  # Ensure 'h' is accessible in this function
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create masks for detecting red color
    red_mask1 = cv2.inRange(hsv_frame, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv_frame, lower_red2, upper_red2)
    red_mask = red_mask1 | red_mask2  # Combine both red masks

    # Find contours for red parts
    contours_red, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    red_detected = False

    for contour in contours_red:
        area = cv2.contourArea(contour)
        if area > 500:  # Adjust the area threshold as needed
            red_detected = True
            x, y, w, h_rect = cv2.boundingRect(contour)  # Renamed variable to avoid conflict
            cv2.rectangle(frame, (x, y), (x + w, y + h_rect), (0, 0, 255), 2)
            cv2.putText(frame, "Red Detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Control the LED based on red detection
    if red_detected:
        lgpio.gpio_write(h, LED, 1)  # Turn LED on
    else:
        lgpio.gpio_write(h, LED, 0)  # Turn LED off

print("Press 'q' to quit the program.")

try:
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
finally:
    # Ensure the LED is off and clean up GPIO
    if h is not None:
        lgpio.gpio_write(h, LED, 0)
        lgpio.gpiochip_close(h)

    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()
