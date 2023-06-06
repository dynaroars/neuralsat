#! /bin/bash

set -e
set -x

# Install lapack, blas
yum install -y lapack-devel blas-devel

# Install CUDA 11.8:
echo "Installing CUDA 11.8"
yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64/cuda-rhel7.repo
yum install --setopt=obsoletes=0 -y \
    cuda-nvcc-11-8-11.8.89-1 \
    cuda-cudart-devel-11-8-11.8.89-1 \
    libcublas-devel-11-8-11.11.3.6-1 \
    libcurand-devel-11-8-10.3.0.86-1 \
    libcusolver-devel-11-8-11.4.1.48-1 \
    libcusparse-devel-11-8-11.7.5.86-1
ln -s cuda-11.8 /usr/local/cuda

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_HOME="/usr/local/cuda"

# test
nvcc -V
