#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>

__global__ void nlm_d(const float* src,
                     float* dst,
                     int rows,
                     int cols,
                     float h,
                     int patchRadius,
                     int searchRadius) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= cols || y >= rows) {
        return;
    }

    float normalization = 0.0f;
    float weightedSum = 0.0f;

    for (int sy = -searchRadius; sy <= searchRadius; ++sy) {
        int y2 = y + sy;
        if (y2 < 0 || y2 >= rows) {
            continue;
        }

        for (int sx = -searchRadius; sx <= searchRadius; ++sx) {
            int x2 = x + sx;
            if (x2 < 0 || x2 >= cols) {
                continue;
            }

            float patchDistance = 0.0f;
            for (int py = -patchRadius; py <= patchRadius; ++py) {
                int yy = y + py;
                int yy2 = y2 + py;
                if (yy < 0 || yy >= rows || yy2 < 0 || yy2 >= rows) {
                    patchDistance += 1.0e6f;
                    continue;
                }

                for (int px = -patchRadius; px <= patchRadius; ++px) {
                    int xx = x + px;
                    int xx2 = x2 + px;
                    if (xx < 0 || xx >= cols || xx2 < 0 || xx2 >= cols) {
                        patchDistance += 1.0e6f;
                        continue;
                    }

                    float diff = src[yy * cols + xx] - src[yy2 * cols + xx2];
                    patchDistance += diff * diff;
                }
            }

            float weight = expf(-patchDistance / (h * h + 1e-6f));
            weightedSum += weight * src[y2 * cols + x2];
            normalization += weight;
        }
    }

    dst[y * cols + x] = (normalization > 0.0f) ? weightedSum / normalization : src[y * cols + x];
}

extern "C" void nlm_h(const cv::Mat& src, cv::Mat& dst, float h, int matchWindow, int searchWindow) {
    if (matchWindow <= 0 || searchWindow <= 0 || matchWindow > searchWindow) {
        std::cerr << "Invalid NLM window parameters." << std::endl;
        dst = src.clone();
        return;
    }
    
    if (src.empty()) {
        dst.release();
        return;
    }

    if (src.type() != CV_32FC1) {
        cv::Mat converted;
        src.convertTo(converted, CV_32FC1);
        nlm_h(converted, dst, h, matchWindow, searchWindow);
        return;
    }

    int rows = src.rows;
    int cols = src.cols;
    int patchRadius = matchWindow / 2;
    int searchRadius = searchWindow / 2;

    dst.create(rows, cols, CV_32FC1);

    float* d_src = nullptr;
    float* d_dst = nullptr;
    size_t bytes = static_cast<size_t>(rows) * cols * sizeof(float);

    cudaError_t err = cudaMalloc(&d_src, bytes);
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc d_src failed: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    err = cudaMalloc(&d_dst, bytes);
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc d_dst failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_src);
        return;
    }

    err = cudaMemcpy(d_src, src.data, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy H2D failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_src);
        cudaFree(d_dst);
        return;
    }

    dim3 threads(16, 16);
    dim3 blocks((cols + threads.x - 1) / threads.x, (rows + threads.y - 1) / threads.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    nlm_d<<<blocks, threads>>>(d_src, d_dst, rows, cols, h, patchRadius, searchRadius);
    cudaError_t kernelErr = cudaGetLastError();
    if (kernelErr != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(kernelErr) << std::endl;
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(d_src);
        cudaFree(d_dst);
        return;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "NLM kernel time: " << ms << " ms" << std::endl;

    err = cudaMemcpy(dst.data, d_dst, bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy D2H failed: " << cudaGetErrorString(err) << std::endl;
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_src);
    cudaFree(d_dst);
}
