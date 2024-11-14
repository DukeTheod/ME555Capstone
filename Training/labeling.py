import cv2
import os

# Paths
image_folder = 'C:\\Users\\theod\\Documents\\School\\Github\\ME555Capstone\\Training\\dataset\\val\\images'
output_label_folder = 'C:\\Users\\theod\\Documents\\School\\Github\\ME555Capstone\\Training\\dataset\\val\\labels'
os.makedirs(output_label_folder, exist_ok=True)

# Variables for drawing bounding boxes
drawing = False  # True if the mouse is pressed
ix, iy = -1, -1  # Initial coordinates
rectangles = []  # List to store bounding box coordinates

# Mouse callback function
def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, rectangles

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img_copy = img.copy()
            cv2.rectangle(img_copy, (ix, iy), (x, y), (0, 255, 0), 2)
            cv2.imshow('Image', img_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        rectangles.append((ix, iy, x, y))
        cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 2)
        cv2.imshow('Image', img)

# Process each image
for image_name in os.listdir(image_folder):
    if image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(image_folder, image_name)
        img = cv2.imread(image_path)
        img_height, img_width = img.shape[:2]

        # Set up the window and mouse callback
        cv2.imshow('Image', img)
        cv2.setMouseCallback('Image', draw_rectangle)

        rectangles = []  # Reset rectangles for each image
        print(f'Labeling {image_name} - Draw bounding boxes and close the window when done.')

        # Wait for the user to close the window
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Save labels automatically after window is closed
        if rectangles:
            label_file_path = os.path.join(output_label_folder, f"{os.path.splitext(image_name)[0]}.txt")
            with open(label_file_path, 'w') as label_file:
                for (x1, y1, x2, y2) in rectangles:
                    # Normalize coordinates
                    x_center = ((x1 + x2) / 2) / img_width
                    y_center = ((y1 + y2) / 2) / img_height
                    width = abs(x2 - x1) / img_width
                    height = abs(y2 - y1) / img_height

                    # Class ID is 0 for a single object type
                    label_file.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

            print(f"Labels saved for {image_name}")

print("Labeling complete for all images.")
