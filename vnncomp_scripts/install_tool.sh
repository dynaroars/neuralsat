#!/bin/bash

TOOL_NAME=NeuralSAT
VERSION_STRING=v1

DIR=$(dirname $(dirname $(realpath $0)))
echo "Installing $TOOL_NAME at $DIR"

# check arguments
if [ "$1" != ${VERSION_STRING} ]; then
	echo "Expected first argument (version string) '$VERSION_STRING', got '$1'"
	exit 1
fi

# Install NVIDIA driver
wget https://us.download.nvidia.com/XFree86/Linux-x86_64/535.54.03/NVIDIA-Linux-x86_64-535.54.03.run

sudo nvidia-smi -pm 0
chmod +x ./NVIDIA-Linux-x86_64-535.54.03.run
sudo ./NVIDIA-Linux-x86_64-535.54.03.run --silent --dkms
# Remove old driver (if already installed) and reload the new one.
sudo rmmod nvidia_uvm; sudo rmmod nvidia_drm; sudo rmmod nvidia_modeset; sudo rmmod nvidia
sudo modprobe nvidia; sudo nvidia-smi -e 0; sudo nvidia-smi -r -i 0
sudo nvidia-smi -pm 1
# Make sure GPU shows up.
nvidia-smi

# Install dependencies
sudo apt update && sudo apt install -y python3 python3-pip && sudo apt install -y psmisc

pip3 install -r "$DIR/requirements.txt"
