import cv2
import numpy as np
import os

# Load the original image
input_path = '/Users/theoy/Documents/GitHub/ME555Capstone/data/IMG_8505.jpg'
image = cv2.imread(input_path)

# Create an output directory to save augmented images
output_dir = 'augmented_dataset'
os.makedirs(output_dir, exist_ok=True)

# Define augmentation parameters
num_images = 100  # Number of augmented images to generate
rotation_angles = np.linspace(0, 360, num_images, endpoint=False)  # Rotate by different angles

# Generate augmented images
for i, angle in enumerate(rotation_angles):
    # Rotate the image
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, matrix, (w, h))

    # Random brightness adjustment
    brightness_factor = 0.5 + np.random.uniform()  # Brightness factor between 0.5 and 1.5
    bright_image = cv2.convertScaleAbs(rotated_image, alpha=brightness_factor, beta=0)

    # Add random noise
    noise = np.random.randint(0, 50, (h, w, 3), dtype='uint8')
    noisy_image = cv2.add(bright_image, noise)

    # Save the augmented image
    output_path = os.path.join(output_dir, f'part_augmented_{i}.jpg')
    cv2.imwrite(output_path, noisy_image)

print(f"Augmented images saved in {output_dir}")