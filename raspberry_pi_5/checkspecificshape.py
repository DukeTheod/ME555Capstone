import cv2
import numpy as np

cap = cv2.VideoCapture(4)  # 4 for rgb camera, and 2 is for depth camera

# Bring up camera view 
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from webcam")
        break


    cv2.imshow("Custom Shape Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
