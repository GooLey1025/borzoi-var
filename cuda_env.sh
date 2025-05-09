#!/bin/bash

# CUDA paths
export PATH=/usr/local/cuda-12.9/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64:$LD_LIBRARY_PATH

# cuDNN paths
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# Print confirmation
echo "CUDA and cuDNN environment variables have been set:"
echo "CUDA_HOME: /usr/local/cuda-12.9"
echo "PATH includes: /usr/local/cuda-12.9/bin"
echo "LD_LIBRARY_PATH includes: /usr/local/cuda-12.9/lib64 and /usr/lib/x86_64-linux-gnu" 