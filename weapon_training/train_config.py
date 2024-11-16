import subprocess
import os

# Ensure you are in the YOLOv5 directory
yolov5_repo_path = '/home/theod/Documents/ME555Capstone/weapon_training/yolov5'

# Path to the data configuration file and pre-trained weights (adjust as needed)
data_yaml_path = '/home/theod/Documents/ME555Capstone/weapon_training/custom_data.yaml'  # Your data config path
weights_path = 'yolov5s.pt'  # Use 'yolov5n.pt' for Nano version or adjust as needed for smaller models

# Training parameters
img_size = 640  # Image size (adjust as needed for Raspberry Pi resources)
batch_size = 5  # Reduce batch size to fit in Raspberry Pi memory
epochs = 66  # Number of training epochs

# Command to train the YOLOv5 model
train_command = [
    'python3', 'train.py',
    '--img', str(img_size),
    '--batch', str(batch_size),
    '--epochs', str(epochs),
    '--data', data_yaml_path,
    '--weights', weights_path,
    '--device', 'cpu'  # Training on CPU as Raspberry Pi typically lacks CUDA support
]

# Run the training command
try:
    subprocess.run(train_command, check=True, cwd=yolov5_repo_path)
    print("Training completed successfully!")
except subprocess.CalledProcessError as e:
    print("An error occurred during training:", e)
