import cv2
import numpy as np

# Initialize webcam (0 is usually the default camera index)
cap = cv2.VideoCapture(0)

# Define the color range for detecting red (HSV format)
lower_red = np.array([0, 120, 70])
upper_red = np.array([10, 255, 255])
lower_red2 = np.array([170, 120, 70])
upper_red2 = np.array([180, 255, 255])

# Function to check if red color is present in the frame
def is_red_present(mask):
    # Count the number of red pixels
    red_pixels = cv2.countNonZero(mask)
    # Set threshold for detecting significant red area
    if red_pixels > 100:  # Adjust the threshold based on object size
        return True
    return False

print("Press 'q' to quit the program.")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Convert frame to HSV color space for color detection
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a mask for detecting red color
    red_mask1 = cv2.inRange(hsv_frame, lower_red, upper_red)
    red_mask2 = cv2.inRange(hsv_frame, lower_red2, upper_red2)
    red_mask = red_mask1 | red_mask2  # Combine both ranges

    # Check if red is present in the frame
    red_detected = is_red_present(red_mask)

    # Annotate the frame
    if red_detected:
        cv2.putText(frame, "Red Object Detected - Flagged as FALSE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        print("Warning: Red object detected on conveyor belt.")

    # Display the frame with annotation
    cv2.imshow("Conveyor Belt Monitoring", frame)

    # Exit loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
