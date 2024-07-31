#!/bin/bash

# Install packages with conda
echo "Installing packages..."
conda install -y pytorch=12.1 torchvision -c pytorch -c nvidia
conda install -y tensorboard matplotlib
conda install gitpython -c conda-forge

# Install packages with pip
pip install typed-argument-parser opencv-python gym pyglet pymatreader timm blobfile torch_scatter

echo "All packages have been installed successfully."