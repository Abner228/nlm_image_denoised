# CUDA Non-Local Means Image Denoising

A small C++/CUDA project that implements grayscale image denoising with the
Non-Local Means (NLM) algorithm. OpenCV is used to load and save images, while
the denoising kernel runs on an NVIDIA GPU using CUDA.

## Features

- Reads an input image as grayscale.
- Computes patch similarities and denoises pixels with a CUDA kernel.
- Lets you configure the filtering strength, patch window, and search window.
- Saves the denoised image to a specified path or to `nlm_denoised.png` by
  default.

## Requirements

- An NVIDIA GPU and a compatible CUDA Toolkit
- CMake 3.10 or later
- A C++ compiler supported by your CUDA Toolkit
- OpenCV development libraries

## Build

From the project root, configure and build with CMake:

```bash
cmake -S . -B build
cmake --build build -j
```

If CMake cannot find CUDA or OpenCV, install their development packages or
provide their installation paths when configuring the project.

## Usage

```bash
./build/opencv_example <input_image> <h> <match_window> <search_window> [output_image]
```

For example:

```bash
./build/opencv_example noisy.png 10 3 21 denoised.png
```

Arguments:

- `input_image`: Path to the input image. It is loaded in grayscale.
- `h`: NLM filtering-strength parameter.
- `match_window`: Patch window size used to compare image regions.
- `search_window`: Search window size used to find similar regions.
- `output_image`: Optional output path. Defaults to `./nlm_denoised.png`.

The program also writes a copy of the input image as `./orig.png` in the
current working directory.

## Project layout

```text
include/nlm.cuh   NLM function declaration
src/main.cpp     Command-line interface and image I/O
src/nlm.cu       CUDA NLM kernel and host-side CUDA code
```
