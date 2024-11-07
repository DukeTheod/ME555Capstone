import cv2
import numpy as np
import RPi.GPIO as GPIO
import time

# Setup for GPIO
LED_PIN = 17  # Choose the GPIO pin where the LED is connected
GPIO.setmode(GPIO.BCM)  # Use Broadcom pin numbering
GPIO.setup(LED_PIN, GPIO.OUT)  # Set the pin as an output

# Initialize webcam (0 is usually the default camera index)
cap = cv2.VideoCapture(0)

# Define the color ranges for detecting red (HSV format)
lower_red1 = np.array([0, 150, 150])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 150, 150])
upper_red2 = np.array([180, 255, 255])

# Function to detect and label parts, and light up the LED if red is detected
def detect_and_label_parts(frame):
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
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, "False", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Light up the LED if red is detected
    if red_detected:
        GPIO.output(LED_PIN, GPIO.HIGH)
    else:
        GPIO.output(LED_PIN, GPIO.LOW)

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
    # Clean up GPIO settings
    GPIO.output(LED_PIN, GPIO.LOW)
    GPIO.cleanup()
    cap.release()
    cv2.destroyAllWindows()