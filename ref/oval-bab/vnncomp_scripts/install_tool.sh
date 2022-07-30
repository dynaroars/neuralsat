#!/bin/bash
# install_tool.sh script for VNNCOMP21 for oval

TOOL_NAME=oval
VERSION_STRING=v1

# check arguments
if [ "$1" != ${VERSION_STRING} ]; then
	echo "Expected first argument (version string) '$VERSION_STRING', got '$1'"
	exit 1
fi

echo "Installing $TOOL_NAME"
DIR=$(dirname $(dirname $(realpath $0)))

eval "$(conda shell.bash hook)" &&  # Allow for conda activate within bash script
conda create -y -n oval python=3.6 &&
conda activate oval &&
conda install -y pytorch torchvision cudatoolkit -c pytorch &&
pip install "$DIR" &&
conda deactivate