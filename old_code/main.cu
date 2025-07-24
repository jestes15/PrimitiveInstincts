#include "cuda_mem_ctrl.hpp"
#include "kernels.h"

#include <unistd.h>

#include <cuda.h>
#include <ipp.h>
#include <cublas_v2.h>
#include <fstream>

#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>

constexpr int width = 1920;
constexpr int height = 1080;

constexpr int resize_width = 1984;
constexpr int resize_height = 1984;

constexpr IppiSize ipp_roi = IppiSize{width, height};

void quantizeDifferences(cublasHandle_t handle,
                         // cv::cuda::GpuMat ippGenerated,
                         cv::Mat ippGenerated,
                         float *basic_nvbt601,
                         float *basic_nvbt709,
                         float *basic_intel601,
                         float *basic_intel709,
                         float *fma_nvbt601,
                         float *fma_nvbt709,
                         float *fma_intel601,
                         float *fma_intel601_fp16,
                         float *fma_intel709,
                         float *approx,
                         std::ofstream &results)
{
    float *ipp_gen1, *ipp_gen2, *ipp_gen3, *ipp_gen4, *ipp_gen5, *ipp_gen6, *ipp_gen7, *ipp_gen8, *ipp_gen9, *ipp_gen10;
    auto size = width * height * 3 * sizeof(float);
    auto len = width * height * 3;
    float alpha = -1.0f;

    cudaMalloc((void **)&ipp_gen1, size);
    cudaMalloc((void **)&ipp_gen2, size);
    cudaMalloc((void **)&ipp_gen3, size);
    cudaMalloc((void **)&ipp_gen4, size);
    cudaMalloc((void **)&ipp_gen5, size);
    cudaMalloc((void **)&ipp_gen6, size);
    cudaMalloc((void **)&ipp_gen7, size);
    cudaMalloc((void **)&ipp_gen8, size);
    cudaMalloc((void **)&ipp_gen9, size);
    cudaMalloc((void **)&ipp_gen10, size);

    cudaMemcpy(ipp_gen1, ippGenerated.ptr(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(ipp_gen2, ippGenerated.ptr(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(ipp_gen3, ippGenerated.ptr(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(ipp_gen4, ippGenerated.ptr(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(ipp_gen5, ippGenerated.ptr(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(ipp_gen6, ippGenerated.ptr(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(ipp_gen7, ippGenerated.ptr(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(ipp_gen8, ippGenerated.ptr(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(ipp_gen9, ippGenerated.ptr(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(ipp_gen10, ippGenerated.ptr(), size, cudaMemcpyHostToDevice);

    // cudaMemcpy(ipp_gen1, ippGenerated.ptr(), size, cudaMemcpyDeviceToDevice);
    // cudaMemcpy(ipp_gen2, ippGenerated.ptr(), size, cudaMemcpyDeviceToDevice);
    // cudaMemcpy(ipp_gen3, ippGenerated.ptr(), size, cudaMemcpyDeviceToDevice);
    // cudaMemcpy(ipp_gen4, ippGenerated.ptr(), size, cudaMemcpyDeviceToDevice);
    // cudaMemcpy(ipp_gen5, ippGenerated.ptr(), size, cudaMemcpyDeviceToDevice);
    // cudaMemcpy(ipp_gen6, ippGenerated.ptr(), size, cudaMemcpyDeviceToDevice);
    // cudaMemcpy(ipp_gen7, ippGenerated.ptr(), size, cudaMemcpyDeviceToDevice);
    // cudaMemcpy(ipp_gen8, ippGenerated.ptr(), size, cudaMemcpyDeviceToDevice);
    // cudaMemcpy(ipp_gen9, ippGenerated.ptr(), size, cudaMemcpyDeviceToDevice);
    // cudaMemcpy(ipp_gen10, ippGenerated.ptr(), size, cudaMemcpyDeviceToDevice);

    float reference_nrm = 0;
    float basic_nvbt601_nrm = 0;
    float basic_nvbt709_nrm = 0;
    float basic_intel601_nrm = 0;
    float basic_intel709_nrm = 0;
    float fma_nvbt601_nrm = 0;
    float fma_nvbt709_nrm = 0;
    float fma_intel601_nrm = 0;
    float fma_intel601fp16_nrm = 0;
    float fma_intel709_nrm = 0;
    float approx_nrm = 0;

    cublasSnrm2(handle, len, ipp_gen1, 1, &reference_nrm);

    cublasSaxpy_v2(handle, len, &alpha, basic_nvbt601, 1, ipp_gen1, 1);
    cublasSaxpy_v2(handle, len, &alpha, basic_nvbt709, 1, ipp_gen2, 1);
    cublasSaxpy_v2(handle, len, &alpha, basic_intel601, 1, ipp_gen3, 1);
    cublasSaxpy_v2(handle, len, &alpha, basic_intel709, 1, ipp_gen4, 1);
    cublasSaxpy_v2(handle, len, &alpha, fma_nvbt601, 1, ipp_gen5, 1);
    cublasSaxpy_v2(handle, len, &alpha, fma_nvbt709, 1, ipp_gen6, 1);
    cublasSaxpy_v2(handle, len, &alpha, fma_intel601, 1, ipp_gen7, 1);
    cublasSaxpy_v2(handle, len, &alpha, fma_intel601_fp16, 1, ipp_gen8, 1);
    cublasSaxpy_v2(handle, len, &alpha, fma_intel709, 1, ipp_gen9, 1);
    cublasSaxpy_v2(handle, len, &alpha, approx, 1, ipp_gen10, 1);

    cublasSnrm2(handle, len, ipp_gen1, 1, &basic_nvbt601_nrm);
    cublasSnrm2(handle, len, ipp_gen2, 1, &basic_nvbt709_nrm);
    cublasSnrm2(handle, len, ipp_gen3, 1, &basic_intel601_nrm);
    cublasSnrm2(handle, len, ipp_gen4, 1, &basic_intel709_nrm);
    cublasSnrm2(handle, len, ipp_gen5, 1, &fma_nvbt601_nrm);
    cublasSnrm2(handle, len, ipp_gen6, 1, &fma_nvbt709_nrm);
    cublasSnrm2(handle, len, ipp_gen7, 1, &fma_intel601_nrm);
    cublasSnrm2(handle, len, ipp_gen8, 1, &fma_intel601fp16_nrm);
    cublasSnrm2(handle, len, ipp_gen9, 1, &fma_intel709_nrm);
    cublasSnrm2(handle, len, ipp_gen10, 1, &approx_nrm);

    cudaStream_t stream;
    cublasGetStream(handle, &stream);
    cudaStreamSynchronize(stream);

    // approx,basic_nvbt601,basic_nvbt709,basic_intel601,basic_intel709,fma_nvbt601,fma_nvbt709,fma_intel601,fma_intel709
    results << (approx_nrm / reference_nrm) << ","
            << (basic_nvbt601_nrm / reference_nrm) << ","
            << (basic_nvbt709_nrm / reference_nrm) << ","
            << (basic_intel601_nrm / reference_nrm) << ","
            << (basic_intel709_nrm / reference_nrm) << ","
            << (fma_nvbt601_nrm / reference_nrm) << ","
            << (fma_nvbt709_nrm / reference_nrm) << ","
            << (fma_intel601_nrm / reference_nrm) << ","
            << (fma_intel601fp16_nrm / reference_nrm) << ","
            << (fma_intel709_nrm / reference_nrm) << "\n";

    cudaFree(ipp_gen1);
    cudaFree(ipp_gen2);
    cudaFree(ipp_gen3);
    cudaFree(ipp_gen4);
    cudaFree(ipp_gen5);
    cudaFree(ipp_gen6);
    cudaFree(ipp_gen7);
    cudaFree(ipp_gen8);
    cudaFree(ipp_gen9);
    cudaFree(ipp_gen10);
}

void display_image(float *device_image, int width, int height) {
    float* host_image = nullptr;
    cudaMallocHost((void**)&host_image, sizeof(float) * width * height);
    cudaMemcpy(host_image, device_image, sizeof(float) * width * height, cudaMemcpyDeviceToHost);

    cv::Mat image(height, width, CV_32FC1, host_image);
    cv::namedWindow("Window", cv::WINDOW_NORMAL);
    cv::resizeWindow("Window", 800, 800);
    cv::imshow("Window", image);
    cv::waitKey(0);
    cv::destroyAllWindows();

    cudaFree(host_image);
}

int main()
{
    cv::utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_SILENT);

    auto context = npp_stream_ctx::create_npp_stream_ctx();
    cublasHandle_t handle;
    cublasCreate(&handle);

    cublasSetStream(handle, context.hStream);

    std::ofstream results("results.csv");

    results << "approx,basic_nvbt601,basic_nvbt709,basic_intel601,basic_intel709,fma_nvbt601,fma_nvbt709,fma_intel601,fma_intel601fp16,fma_intel709\n"
            << std::flush;

    for (int i = 0; i < 100; i++)
    {
        std::cout << '\r' << i + 1 << "/100\t[" << std::string(i, '.') << std::string(99 - i, ' ') << "]";

        std::stringstream ss;
        ss << "/home/bl4z3/develop/PrimitiveInstincts/test_sets/images_1920x1080/image_" << i << ".jpg";
        cv::Mat pimage = cv::imread(ss.str());
        cv::Mat image;
        cv::cvtColor(pimage, image, cv::COLOR_BGR2RGB);

        uint8_t *rgb_image = reinterpret_cast<uint8_t *>(ippMalloc(width * height * 3));
        uint8_t *cbycr_image = reinterpret_cast<uint8_t *>(ippMalloc(width * height * 2));
        uint8_t *device_source_1, *device_source_2, *device_source_3, *device_source_4, *device_source_5, *device_source_6, *device_source_7, *device_source_8, *device_source_9, *device_source_10;
        float *device_intermediate_1, *device_intermediate_2, *device_intermediate_3, *device_intermediate_4, *device_intermediate_5, *device_intermediate_6, *device_intermediate_7, *device_intermediate_8, *device_intermediate_9, *device_intermediate_10;
        float *device_destination_1, *device_destination_2, *device_destination_3, *device_destination_4, *device_destination_5, *device_destination_6, *device_destination_7, *device_destination_8, *device_destination_9, *device_destination_10;

        memcpy(rgb_image, image.ptr<void>(), width * height * 3);

        cudaMalloc((void **)&device_source_1, width * height * 2);
        cudaMalloc((void **)&device_source_2, width * height * 2);
        cudaMalloc((void **)&device_source_3, width * height * 2);
        cudaMalloc((void **)&device_source_4, width * height * 2);
        cudaMalloc((void **)&device_source_5, width * height * 2);
        cudaMalloc((void **)&device_source_6, width * height * 2);
        cudaMalloc((void **)&device_source_7, width * height * 2);
        cudaMalloc((void **)&device_source_8, width * height * 2);
        cudaMalloc((void **)&device_source_9, width * height * 2);
        cudaMalloc((void **)&device_source_10, width * height * 2);

        cudaMalloc((void **)&device_intermediate_1, sizeof(float) * width * height * 3);
        cudaMalloc((void **)&device_intermediate_2, sizeof(float) * width * height * 3);
        cudaMalloc((void **)&device_intermediate_3, sizeof(float) * width * height * 3);
        cudaMalloc((void **)&device_intermediate_4, sizeof(float) * width * height * 3);
        cudaMalloc((void **)&device_intermediate_5, sizeof(float) * width * height * 3);
        cudaMalloc((void **)&device_intermediate_6, sizeof(float) * width * height * 3);
        cudaMalloc((void **)&device_intermediate_7, sizeof(float) * width * height * 3);
        cudaMalloc((void **)&device_intermediate_8, sizeof(float) * width * height * 3);
        cudaMalloc((void **)&device_intermediate_9, sizeof(float) * width * height * 3);
        cudaMalloc((void **)&device_intermediate_10, sizeof(float) * width * height * 3);

        cudaMalloc((void **)&device_destination_1, sizeof(float) * resize_width * resize_height * 3);
        cudaMalloc((void **)&device_destination_2, sizeof(float) * resize_width * resize_height * 3);
        cudaMalloc((void **)&device_destination_3, sizeof(float) * resize_width * resize_height * 3);
        cudaMalloc((void **)&device_destination_4, sizeof(float) * resize_width * resize_height * 3);
        cudaMalloc((void **)&device_destination_5, sizeof(float) * resize_width * resize_height * 3);
        cudaMalloc((void **)&device_destination_6, sizeof(float) * resize_width * resize_height * 3);
        cudaMalloc((void **)&device_destination_7, sizeof(float) * resize_width * resize_height * 3);
        cudaMalloc((void **)&device_destination_8, sizeof(float) * resize_width * resize_height * 3);
        cudaMalloc((void **)&device_destination_9, sizeof(float) * resize_width * resize_height * 3);
        cudaMalloc((void **)&device_destination_10, sizeof(float) * resize_width * resize_height * 3);

        cudaMemset((void *)device_intermediate_1, 0, sizeof(float) * width * height * 3);
        cudaMemset((void *)device_intermediate_2, 0, sizeof(float) * width * height * 3);
        cudaMemset((void *)device_intermediate_3, 0, sizeof(float) * width * height * 3);
        cudaMemset((void *)device_intermediate_4, 0, sizeof(float) * width * height * 3);
        cudaMemset((void *)device_intermediate_5, 0, sizeof(float) * width * height * 3);
        cudaMemset((void *)device_intermediate_6, 0, sizeof(float) * width * height * 3);
        cudaMemset((void *)device_intermediate_7, 0, sizeof(float) * width * height * 3);
        cudaMemset((void *)device_intermediate_8, 0, sizeof(float) * width * height * 3);
        cudaMemset((void *)device_intermediate_9, 0, sizeof(float) * width * height * 3);
        cudaMemset((void *)device_intermediate_10, 0, sizeof(float) * width * height * 3);

        ippiRGBToCbYCr422_8u_C3C2R(rgb_image, width * 3, cbycr_image, width * 2, ipp_roi);

        cudaMemcpy(device_source_1, cbycr_image, width * height * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(device_source_2, cbycr_image, width * height * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(device_source_3, cbycr_image, width * height * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(device_source_4, cbycr_image, width * height * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(device_source_5, cbycr_image, width * height * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(device_source_6, cbycr_image, width * height * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(device_source_7, cbycr_image, width * height * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(device_source_8, cbycr_image, width * height * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(device_source_9, cbycr_image, width * height * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(device_source_10, cbycr_image, width * height * 2, cudaMemcpyHostToDevice);

        // cv::cuda::GpuMat ippImage = ipp_convertAndResize(cbycr_image);
        cv::Mat ippImage = ipp_convertAndResize(cbycr_image);
        convert_CbYCrToBGR24_f32_P3R(device_source_1, device_intermediate_1, width, height, device_destination_1, resize_width, resize_height, context, BASIC_IMPL_NVIDIA_BT601);
        convert_CbYCrToBGR24_f32_P3R(device_source_2, device_intermediate_2, width, height, device_destination_2, resize_width, resize_height, context, BASIC_IMPL_NVIDIA_BT709);
        convert_CbYCrToBGR24_f32_P3R(device_source_3, device_intermediate_3, width, height, device_destination_3, resize_width, resize_height, context, BASIC_IMPL_INTEL_BT601);
        convert_CbYCrToBGR24_f32_P3R(device_source_4, device_intermediate_4, width, height, device_destination_4, resize_width, resize_height, context, BASIC_IMPL_INTEL_BT709);
        convert_CbYCrToBGR24_f32_P3R(device_source_5, device_intermediate_5, width, height, device_destination_5, resize_width, resize_height, context, FMA_IMPL_NVIDIA_BT601);
        convert_CbYCrToBGR24_f32_P3R(device_source_6, device_intermediate_6, width, height, device_destination_6, resize_width, resize_height, context, FMA_IMPL_NVIDIA_BT709);
        convert_CbYCrToBGR24_f32_P3R(device_source_7, device_intermediate_7, width, height, device_destination_7, resize_width, resize_height, context, FMA_IMPL_INTEL_BT601);
        convert_CbYCrToBGR24_f32_P3R(device_source_8, device_intermediate_8, width, height, device_destination_8, resize_width, resize_height, context, FMA_IMPL_INTEL_BT601_FP16);
        convert_CbYCrToBGR24_f32_P3R(device_source_9, device_intermediate_9, width, height, device_destination_9, resize_width, resize_height, context, FMA_IMPL_INTEL_BT709);
        convert_CbYCrToBGR24_f32_P3R_approx(device_source_10, device_intermediate_10, width, height, device_destination_10, resize_width, resize_height, context);

        cudaStreamSynchronize(context.hStream);
        // display_image(device_destination_7, resize_width, resize_height);
        // display_image(device_destination_8, resize_width, resize_height);

        quantizeDifferences(handle,
                            ippImage,
                            device_destination_1,  // BASIC_IMPL_NVIDIA_BT601
                            device_destination_2,  // BASIC_IMPL_NVIDIA_BT709
                            device_destination_3,  // BASIC_IMPL_INTEL_BT601
                            device_destination_4,  // BASIC_IMPL_INTEL_BT709
                            device_destination_5,  // FMA_IMPL_NVIDIA_BT601
                            device_destination_6,  // FMA_IMPL_NVIDIA_BT709
                            device_destination_7,  // FMA_IMPL_INTEL_BT601
                            device_destination_8,  // FMA_IMPL_INTEL_BT601_FP16
                            device_destination_9,  // FMA_IMPL_INTEL_BT709
                            device_destination_10, // APPROX
                            results);

        cudaFree(device_source_1);
        cudaFree(device_source_2);
        cudaFree(device_source_3);
        cudaFree(device_source_4);
        cudaFree(device_source_5);
        cudaFree(device_source_6);
        cudaFree(device_source_7);
        cudaFree(device_source_8);
        cudaFree(device_source_9);
        cudaFree(device_intermediate_1);
        cudaFree(device_intermediate_2);
        cudaFree(device_intermediate_3);
        cudaFree(device_intermediate_4);
        cudaFree(device_intermediate_5);
        cudaFree(device_intermediate_6);
        cudaFree(device_intermediate_7);
        cudaFree(device_intermediate_8);
        cudaFree(device_intermediate_9);
        cudaFree(device_destination_1);
        cudaFree(device_destination_2);
        cudaFree(device_destination_3);
        cudaFree(device_destination_4);
        cudaFree(device_destination_5);
        cudaFree(device_destination_6);
        cudaFree(device_destination_7);
        cudaFree(device_destination_8);
        cudaFree(device_destination_9);

        ippFree(rgb_image);
        ippFree(cbycr_image);
    }

    std::cout << std::endl;
}
