import subprocess

# Path to the YOLOv5 repository (make sure you clone it if you haven't)
yolov5_repo_path = 'C:\\Users\\theod\\Documents\\School\\Github\\ME555Capstone\\Training\\yolov5'

# Paths to configuration and weight files
data_yaml_path = 'C:\\Users\\theod\\Documents\\School\\Github\\ME555Capstone\\Training\\custom_data.yaml'  # Path to your data config file
weights_path = 'yolov5s.pt'  # You can use pre-trained weights (e.g., 'yolov5s.pt', 'yolov5m.pt')

# Training parameters
img_size = 640  # Image size
batch_size = 16  # Batch size
epochs = 50  # Number of epochs

# Command to train the YOLOv5 model with CUDA enabled
train_command = [
    'python', f'{yolov5_repo_path}\\train.py',
    '--img', str(img_size),
    '--batch', str(batch_size),
    '--epochs', str(epochs),
    '--data', data_yaml_path,
    '--weights', weights_path,
    '--device', 'cuda'  # Use '0' for the first GPU, 'cuda' for auto GPU detection, or '0,1' for multiple GPUs
]

# Run the training command
try:
    subprocess.run(train_command, check=True, cwd=yolov5_repo_path)
    print("Training completed successfully with CUDA enabled!")
except subprocess.CalledProcessError as e:
    print("An error occurred during training:", e)
