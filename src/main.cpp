#include <opencv2/opencv.hpp>
#include <iostream>
#include <cuda_runtime.h>
#include <stdexcept>
#include "nlm.cuh"

int main(int argc, char** argv) {
    std::cout << "OpenCV version: " << CV_VERSION << std::endl;

    if (argc < 5) {
        std::cerr << "Usage: " << argv[0]
                  << " <input_image> <h> <match_window> <search_window> [output_image]"
                  << std::endl;
        return 1;
    }

    std::string path = argv[1];
    float h = 0.0f;
    int matchWindow = 0;
    int searchWindow = 0;

    try {
        h = std::stof(argv[2]);
        matchWindow = std::stoi(argv[3]);
        searchWindow = std::stoi(argv[4]);
    } catch (const std::exception& e) {
        std::cerr << "Invalid numeric arguments: " << e.what() << std::endl;
        return 1;
    }

    std::string outputPath = (argc >= 6) ? argv[5] : "./nlm_denoised.png";

    cv::Mat src = cv::imread(path, cv::IMREAD_GRAYSCALE);
    if (src.empty()) {
        std::cerr << "Failed to read image: " << path << std::endl;
        return 1;
    }

    src.convertTo(src, CV_32FC1);

    std::cout << "input image: " << path << std::endl;
    std::cout << "src rows: " << src.rows << std::endl;
    std::cout << "src cols: " << src.cols << std::endl;
    std::cout << "h: " << h << ", matchWindow: " << matchWindow
              << ", searchWindow: " << searchWindow << std::endl;

    cv::Mat dst;
    nlm_h(src, dst, h, matchWindow, searchWindow);

    cv::Mat out;
    dst.convertTo(out, CV_8UC1);
    cv::imwrite(outputPath, out);

    std::cout << "saved denoised image to " << outputPath << std::endl;
    return 0;
}