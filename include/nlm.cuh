#pragma once
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

extern "C" void nlm_h(
    const cv::Mat& src, cv::Mat& dst, float h, int matchWindow, int searchWindow
);