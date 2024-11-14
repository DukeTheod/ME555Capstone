#!/bin/bash

# Update system
echo "Updating system..."
sudo apt-get update && sudo apt-get upgrade -y

# Install Python 3 and pip
echo "Installing Python 3 and pip..."
sudo apt-get install -y python3 python3-pip python3-dev python3-venv

# Install build dependencies
echo "Installing build dependencies..."
sudo apt-get install -y build-essential libssl-dev libffi-dev python3-setuptools

# Install dependencies for OpenCV
echo "Installing OpenCV dependencies..."
sudo apt-get install -y libopencv-dev python3-opencv

# Install pip3 packages
echo "Installing Python dependencies..."
pip3 install --upgrade pip
pip3 install numpy opencv-python

# Install PyTorch (with CUDA support for Raspberry Pi)
echo "Installing PyTorch..."
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/armv7l

# Install other necessary dependencies
echo "Installing additional dependencies..."
pip3 install matplotlib seaborn tqdm

# Clone YOLOv5 repository
echo "Cloning YOLOv5 repository..."
git clone https://github.com/ultralytics/yolov5
cd yolov5

# Install YOLOv5 dependencies
echo "Installing YOLOv5 dependencies..."
pip3 install -r requirements.txt

# Optional: Install ONNX and ONNX Runtime for better performance
echo "Installing ONNX and ONNX Runtime (Optional)..."
pip3 install onnx onnxruntime

# Print successful installation message
echo "All dependencies installed successfully!"
