#!/bin/bash
rm -rf /root/code/cuda/NLM/build
cd /root/code/cuda/NLM
cmake -S . -B build -DCMAKE_C_COMPILER=/usr/bin/gcc-12 \
                    -DCMAKE_CXX_COMPILER=/usr/bin/g++-12 \
                    -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12
# -S . specifies the source directory
# -B build specifies the build directory (if it doesn't exist, it will be created)
# cuda12 requires gcc-12 as host compiler
cmake --build build -j$(nproc)
./build/opencv_example