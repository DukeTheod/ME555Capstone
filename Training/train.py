import subprocess
import os

# Path to the YOLOv5 repository (Update this path according to your system)
yolov5_repo_path = 'C:\\Users\\theod\\Documents\\School\\Github\\ME555Capstone\\Training\\yolov5'

# Paths to configuration and weight files
data_yaml_path = 'C:\\Users\\theod\\Documents\\School\\Github\\ME555Capstone\\Training\\custom_data.yaml'  # Path to your data config file
weights_path = 'yolov5n.pt'  # Use the YOLOv5 Nano weights, or 'best.pt' after training

# Training parameters
img_size = 640  # Image size (adjust as needed)
batch_size = 16  # Batch size (adjust based on your system's GPU memory)
epochs = 50  # Number of epochs

# Command to train the YOLOv5 Nano model
train_command = [
    'python', f'{yolov5_repo_path}\\train.py',
    '--img', str(img_size),
    '--batch', str(batch_size),
    '--epochs', str(epochs),
    '--data', data_yaml_path,
    '--weights', weights_path,
    '--device', 'cuda:0'  # Use 'cuda' to leverage GPU; use 'cpu' if CUDA is unavailable
]

# Run the training command
try:
    subprocess.run(train_command, check=True, cwd=yolov5_repo_path)
    print("Training with YOLOv5 Nano completed successfully!")
    
    # Export the model to ONNX for Raspberry Pi usage
    export_command = [
        'python', f'{yolov5_repo_path}\\export.py',
        '--weights', 'runs\\train\\exp\\weights\\best.pt',  # Adjust path based on where the model is saved after training
        '--img-size', str(img_size),
        '--batch-size', '1',  # Set batch size to 1 for inference
        '--device', 'cpu',  # You can change to 'cuda' if using a GPU during export
        '--dynamic'  # This enables dynamic ONNX shapes, making the model more flexible
    ]
    
    # Run the export command to convert the model to ONNX
    subprocess.run(export_command, check=True, cwd=yolov5_repo_path)
    print("Model exported to ONNX successfully!")

except subprocess.CalledProcessError as e:
    print("An error occurred during training or exporting:", e)
