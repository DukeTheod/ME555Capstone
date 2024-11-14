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

# Define the color ranges for detecting green (HSV format)
lower_green = np.array([35, 100, 100])
upper_green = np.array([85, 255, 255])

# Define the region of interest (ROI) for the belt (adjust as needed)
roi_top = 100  # Starting Y coordinate of the ROI
roi_bottom = 300  # Ending Y coordinate of the ROI
roi_left = 50  # Starting X coordinate of the ROI
roi_right = 600  # Ending X coordinate of the ROI

# Function to detect and label parts and control the LED if green is detected
def detect_and_label_parts(frame):
    global h  # Ensure 'h' is accessible in this function

    # Crop the frame to the region of interest (ROI)
    roi_frame = frame[roi_top:roi_bottom, roi_left:roi_right]
    hsv_frame = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)

    # Create mask for detecting green color
    green_mask = cv2.inRange(hsv_frame, lower_green, upper_green)

    # Find contours for green parts
    contours_green, _ = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    green_detected = False

    for contour in contours_green:
        area = cv2.contourArea(contour)
        if area > 500:  # Adjust the area threshold as needed
            green_detected = True
            x, y, w, h_rect = cv2.boundingRect(contour)
            cv2.rectangle(roi_frame, (x, y), (x + w, y + h_rect), (0, 255, 0), 2)
            cv2.putText(roi_frame, "Green Detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Replace the processed ROI back into the original frame for display
    frame[roi_top:roi_bottom, roi_left:roi_right] = roi_frame

    # Control the LED based on green detection
    if green_detected:
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

        # Detect and label parts based on color within the ROI
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
