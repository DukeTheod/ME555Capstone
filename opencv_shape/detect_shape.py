import cv2
import numpy as np

# Function to detect and label shapes
def detect_shapes(contours):
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        cv2.drawContours(frame, [approx], 0, (0, 255, 0), 2)
        x, y = approx.ravel()[0], approx.ravel()[1] - 10

        if len(approx) == 3:
            shape = "Triangle"
        elif len(approx) == 4:
            (x, y, w, h) = cv2.boundingRect(approx)
            shape = "Square" if 0.95 <= w/h <= 1.05 else "Rectangle"
        elif len(approx) == 5:
            shape = "Pentagon"
        elif len(approx) > 5:
            shape = "Circle"
        else:
            shape = "Unknown"

        cv2.putText(frame, shape, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

# Initialize camera capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    detect_shapes(contours)
    cv2.imshow('Shape Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()