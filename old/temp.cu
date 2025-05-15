#pragma warning(disable : 4101)

#include "ImageDefinitions.hpp"
#include <chrono>
#include <cstdint>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <mma.h>
#include <npp.h>
#include <opencv2/opencv.hpp>
#include <random>
#include <vector>

#define BLOCK_SIZE 32

#define TILE_WIDTH 32
#define TILE_HEIGHT 16

#define PX1_BLU_CH 0
#define PX1_GRN_CH 1
#define PX1_RED_CH 2
#define PX2_BLU_CH 3
#define PX2_GRN_CH 4
#define PX2_RED_CH 5

__global__ void cbycr422_to_bgr24(
    uint8_t *__restrict__ src, uint8_t *__restrict__ dst, int width, int height, int src_pitch, int dst_pitch)
{
    int x = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int64_t src_idx = y * src_pitch + 2 * x;
    int64_t dst_idx = y * dst_pitch + 3 * x;

    int32_t cb, y0, cr, y1;
    uint8_t *addr = src + src_idx;

    // PTX Assembly to speed up pulling data from global memory
    asm volatile(".reg .u8 t1;\n" // Declare temperary registers
                 ".reg .u8 t2;\n"
                 ".reg .u8 t3;\n"
                 ".reg .u8 t4;\n"
                 ".reg .s32 t5;\n"
                 ".reg .s32 t6;\n"
                 // Load each byte from global memory
                 // Instead of coalescing these accesses, I want to load each byte indivudually to lessen
                 // the impact of a warp stall. By the time I use the data that is being loaded in t1
                 // the global data will be loaded into t2-4. A stall is inevitable here, but I can
                 // at least delay when the stall occurs, and by delaying it i can shorten the stall.
                 "ld.global.nc.u8 t1, [%4];\n"
                 "ld.global.nc.u8 t2, [%4+1];\n"
                 "ld.global.nc.u8 t3, [%4+2];\n"
                 "ld.global.nc.u8 t4, [%4+3];\n"
                 // Convert the data in t1 from  uint8_t to int32_t
                 "cvt.s32.u8 t5, t1;\n"
                 // Add -128 to t5 to get the final modified chroma blue value
                 "add.s32 %0, t5, -128;\n"
                 // Convert the data in t1 from  uint8_t to int32_t
                 "cvt.s32.u8 t6, t2;\n"
                 // Add -16 to the luma value stored in register t6
                 "add.s32 t5, t6, -16;\n"
                 // Multiply the luma value by 298 and store it in y0
                 "mul.lo.s32 %1, t5, 298;\n"
                 // Convert the data in t1 from  uint8_t to int32_t
                 "cvt.s32.u8 t6, t3;\n"
                 // Add -128 to t5 to get the final modified chroma red value
                 "add.s32 %2, t6, -128;\n"
                 // Convert the data in t1 from  uint8_t to int32_t
                 "cvt.s32.u8 t5, t4;\n"
                 // Add -16 to the luma value stored in register t5
                 "add.s32 t6, t5, -16;\n"
                 // Multiply the luma value by 298 and store it in y1
                 "mul.lo.s32 %3, t6, 298;\n"
                 : "=r"(cb), "=r"(y0), "=r"(cr), "=r"(y1)
                 : "l"(addr));

    int32_t r0_ = (y0 + 409 * cr + 128) >> 8;
    int32_t g0_ = (y0 - 100 * cb - 208 * cr + 128) >> 8;
    int32_t b0_ = (y0 + 516 * cb + 128) >> 8;
    int32_t r1_ = (y1 + 409 * cr + 128) >> 8;
    int32_t g1_ = (y1 - 100 * cb - 208 * cr + 128) >> 8;
    int32_t b1_ = (y1 + 516 * cb + 128) >> 8;

    uint16_t data = (static_cast<uint8_t>(min(max(g0_, 0), 255)) << 8) | static_cast<uint8_t>(min(max(b0_, 0), 255));
    *(uint16_t *)&dst[dst_idx + 0] = data;
    data = (static_cast<uint8_t>(min(max(b1_, 0), 255)) << 8) | static_cast<uint8_t>(min(max(r0_, 0), 255));
    *(uint16_t *)&dst[dst_idx + 2] = data;
    data = (static_cast<uint8_t>(min(max(r1_, 0), 255)) << 8) | static_cast<uint8_t>(min(max(g1_, 0), 255));
    *(uint16_t *)&dst[dst_idx + 4] = data;
}

void convert_CbYCr_To_BGR24(uint8_t *__restrict__ src,
                            uint8_t *__restrict__ dst,
                            const uint32_t height,
                            const uint32_t width)
{
    dim3 block(TILE_WIDTH, TILE_HEIGHT);
    dim3 grid((width + (2 * TILE_WIDTH - 1)) / (2 * TILE_WIDTH), (height + TILE_HEIGHT - 1) / TILE_HEIGHT);
    cbycr422_to_bgr24<<<grid, block>>>(src,
                                       dst,
                                       width,
                                       height,
                                       width * ImageDefinitions::UYVY_BPP,
                                       width * ImageDefinitions::RGB_BPP);
}

int nppiRGB24ToCbYCr422(const uint8_t *__restrict__ device_src,
                        uint8_t *__restrict__ device_dst,
                        int width,
                        int height,
                        const NppiRect &ROI_Size_Loc = {0, 0, 0, 0})
{
    // Validate input parameters
    if (!device_src || !device_dst || width <= 0 || height <= 0 || (ROI_Size_Loc.x + ROI_Size_Loc.width) > width ||
        (ROI_Size_Loc.y + ROI_Size_Loc.height) > height)
        return -1; // Invalid parameters

    int srcStep = width * ImageDefinitions::RGB_BPP;  // RGB24: 3 bytes per pixel
    int dstStep = width * ImageDefinitions::UYVY_BPP; // CbYCr422: 2 bytes per pixel

    // Define ROI size
    NppiSize roiSize = {0, 0};
    uint32_t srcOffset = 0, dstOffset = 0;

    if (ROI_Size_Loc.width == 0 || ROI_Size_Loc.height == 0)
        roiSize = {width, height};
    else
    {
        roiSize.width = ROI_Size_Loc.width;
        roiSize.height = ROI_Size_Loc.height;

        srcOffset = ROI_Size_Loc.y * srcStep + ROI_Size_Loc.x * ImageDefinitions::RGB_BPP;
        dstOffset = ROI_Size_Loc.y * dstStep + ROI_Size_Loc.x * ImageDefinitions::UYVY_BPP;
    }

    // Perform RGB to YCbCr422 conversion on GPU
    nppiRGBToCbYCr422_8u_C3C2R(device_src + srcOffset, srcStep, device_dst + dstOffset, dstStep, roiSize);
    return 0; // Success
}

int nppiCbYCr422ToBGR24(const uint8_t *__restrict__ device_src,
                        uint8_t *__restrict__ device_dst,
                        int width,
                        int height,
                        const NppiRect &ROI_Size_Loc = {0, 0, 0, 0})
{
    // Validate input parameters
    if (!device_src || !device_dst || width <= 0 || height <= 0 || (ROI_Size_Loc.x + ROI_Size_Loc.width) > width ||
        (ROI_Size_Loc.y + ROI_Size_Loc.height) > height)
        return -1; // Invalid parameters

    int srcStep = width * ImageDefinitions::UYVY_BPP; // RGB24: 3 bytes per pixel
    int dstStep = width * ImageDefinitions::RGB_BPP;  // CbYCr422: 2 bytes per pixel

    // Define ROI size
    NppiSize roiSize = {0, 0};
    NppiSize fillROI = {0, 0};
    uint32_t srcOffset = 0, dstOffset = 0, fillOffset = 0;

    if (ROI_Size_Loc.width == 0 || ROI_Size_Loc.height == 0)
    {
        roiSize = {width, height};
    }
    else
    {
        roiSize.width = ROI_Size_Loc.width;
        roiSize.height = ROI_Size_Loc.height;

        fillROI.width = ROI_Size_Loc.width + ((ROI_Size_Loc.x == 0) ? 0 : 1) + ((ROI_Size_Loc.width == width) ? 0 : 1);
        fillROI.height =
            ROI_Size_Loc.height + ((ROI_Size_Loc.y == 0) ? 0 : 1) + ((ROI_Size_Loc.height == height) ? 0 : 1);
        fillOffset = (ROI_Size_Loc.y - ((ROI_Size_Loc.y == 0) ? 0 : 1)) * dstStep +
                     (ROI_Size_Loc.x - ((ROI_Size_Loc.x == 0) ? 0 : 1)) * ImageDefinitions::RGB_BPP;

        srcOffset = ROI_Size_Loc.y * srcStep + ROI_Size_Loc.x * ImageDefinitions::UYVY_BPP;
        dstOffset = ROI_Size_Loc.y * dstStep + ROI_Size_Loc.x * ImageDefinitions::RGB_BPP;
    }

    if (fillROI.width != 0 && fillROI.height != 0)
    {
        const Npp8u border_color[3] = {0, 0, 255};
        nppiSet_8u_C3R(border_color, device_dst + fillOffset, dstStep, fillROI);
    }

    // Perform YCbCr422 to BGR conversion on GPU
    nppiCbYCr422ToBGR_709HDTV_8u_C2C3R(device_src + srcOffset, srcStep, device_dst + dstOffset, dstStep, roiSize);
    return 0; // Success
}

__global__ void cbycr422_to_rgb_f32_clamped(uint8_t *__restrict__ src,
                                            float *__restrict__ dst_r,
                                            float *__restrict__ dst_g,
                                            float *__restrict__ dst_b,
                                            int width,
                                            int height,
                                            int src_pitch,
                                            int dst_pitch)
{
    int x = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int64_t src_idx = y * src_pitch + 2 * x;
    int64_t dst_idx = y * dst_pitch + x;

    int32_t cb, y0, cr, y1;
    uint8_t *addr = src + src_idx;

    // PTX Assembly to speed up pulling data from global memory
    asm volatile(".reg .u8 t1;\n" // Declare temperary registers
                 ".reg .u8 t2;\n"
                 ".reg .u8 t3;\n"
                 ".reg .u8 t4;\n"
                 ".reg .s32 t5;\n"
                 ".reg .s32 t6;\n"
                 // Load each byte from global memory
                 // Instead of coalescing these accesses, I want to load each byte indivudually to lessen
                 // the impact of a warp stall. By the time I use the data that is being loaded in t1
                 // the global data will be loaded into t2-4. A stall is inevitable here, but I can
                 // at least delay when the stall occurs, and by delaying it i can shorten the stall.
                 "ld.global.nc.u8 t1, [%4];\n"
                 "ld.global.nc.u8 t2, [%4+1];\n"
                 "ld.global.nc.u8 t3, [%4+2];\n"
                 "ld.global.nc.u8 t4, [%4+3];\n"
                 // Convert the data in t1 from  uint8_t to int32_t
                 "cvt.s32.u8 t5, t1;\n"
                 // Add -128 to t5 to get the final modified chroma blue value
                 "add.s32 %0, t5, -128;\n"
                 // Convert the data in t1 from  uint8_t to int32_t
                 "cvt.s32.u8 t6, t2;\n"
                 // Add -16 to the luma value stored in register t6
                 "add.s32 t5, t6, -16;\n"
                 // Multiply the luma value by 298 and store it in y0
                 "mul.lo.s32 %1, t5, 298;\n"
                 // Convert the data in t1 from  uint8_t to int32_t
                 "cvt.s32.u8 t6, t3;\n"
                 // Add -128 to t5 to get the final modified chroma red value
                 "add.s32 %2, t6, -128;\n"
                 // Convert the data in t1 from  uint8_t to int32_t
                 "cvt.s32.u8 t5, t4;\n"
                 // Add -16 to the luma value stored in register t5
                 "add.s32 t6, t5, -16;\n"
                 // Multiply the luma value by 298 and store it in y1
                 "mul.lo.s32 %3, t6, 298;\n"
                 : "=r"(cb), "=r"(y0), "=r"(cr), "=r"(y1)
                 : "l"(addr));

    float r0_ = static_cast<float>((y0 + 409 * cr + 128) >> 8);
    float g0_ = static_cast<float>((y0 - 100 * cb - 208 * cr + 128) >> 8);
    float b0_ = static_cast<float>((y0 + 516 * cb + 128) >> 8);
    float r1_ = static_cast<float>((y1 + 409 * cr + 128) >> 8);
    float g1_ = static_cast<float>((y1 - 100 * cb - 208 * cr + 128) >> 8);
    float b1_ = static_cast<float>((y1 + 516 * cb + 128) >> 8);

    float pair[2];
    // Calculate the reciprocal of 255, 1/255 with the intrinsic function __frcp
    // More information found here:
    // __frcp:
    // https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html?highlight=__frcp#_CPPv49__frcp_rzf
    float recip = __frcp_rz(255.0f);

    // Scale the integer value down to the [0.0, 1.0] (inclusive) scale
    // Information on these intrrinsic functions can be found here:
    // __fmul_rz:
    // https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html?highlight=__fmul_rz#_CPPv49__fmul_rzff
    // __saturatef:
    // https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html?highlight=__saturatef#_CPPv411__saturateff
    pair[0] = __saturatef(__fmul_rz(r0_, recip));
    pair[1] = __saturatef(__fmul_rz(r1_, recip));
    *(uint64_t *)&dst_r[dst_idx] = *(uint64_t *)&pair;

    pair[0] = __saturatef(__fmul_rz(g0_, recip));
    pair[1] = __saturatef(__fmul_rz(g1_, recip));
    *(uint64_t *)&dst_g[dst_idx] = *(uint64_t *)&pair;

    pair[0] = __saturatef(__fmul_rz(b0_, recip));
    pair[1] = __saturatef(__fmul_rz(b1_, recip));
    *(uint64_t *)&dst_b[dst_idx] = *(uint64_t *)&pair;
}

void CbYCr_To_RGB_f32_clamped(uint8_t *__restrict__ device_src,
                              float *__restrict__ device_red,
                              float *__restrict__ device_green,
                              float *__restrict__ device_blue,
                              int width,
                              int height)
{
    int src_pitch = width * ImageDefinitions::UYVY_BPP;
    int dst_pitch = width;

    dim3 block2(TILE_WIDTH, TILE_HEIGHT);
    dim3 grid2((width + (2 * TILE_WIDTH - 1)) / (2 * TILE_WIDTH), (height + TILE_HEIGHT - 1) / TILE_HEIGHT);

    cbycr422_to_rgb_f32_clamped<<<grid2, block2>>>(device_src,
                                                   device_red,
                                                   device_green,
                                                   device_blue,
                                                   width,
                                                   height,
                                                   src_pitch,
                                                   dst_pitch);
}

int main()
{
    std::cout << "##########################################################" << std::endl;
    std::cout << "# Image Processing Using NVIDIA Performance Primitives   #" << std::endl;
    std::cout << "#\tCreated by Joshua Estes                          #" << std::endl;
    std::cout << "##########################################################" << std::endl;

    std::random_device r;
    std::uniform_int_distribution<int> uniform_dist(0, 255);

    uint8_t *device_source_image = nullptr;
    uint8_t *device_intermediate_storage = nullptr;
    uint8_t *device_dst_image1 = nullptr;
    uint8_t *device_dst_image2 = nullptr;

    float *device_dst_red = nullptr;
    float *device_dst_blue = nullptr;
    float *device_dst_green = nullptr;

    float *host_dst_red = nullptr;
    float *host_dst_blue = nullptr;
    float *host_dst_green = nullptr;

    uint8_t *final_image1 = nullptr;
    uint8_t *final_image2 = nullptr;

    // Load the image
    cv::Mat image = cv::imread("images_1920x1080/image_72.jpg", cv::IMREAD_COLOR);

    int width = image.cols;
    int height = image.rows;

    cudaMalloc((void **)&device_source_image, width * height * ImageDefinitions::RGB_BPP);
    cudaMalloc((void **)&device_intermediate_storage, width * height * ImageDefinitions::UYVY_BPP);
    cudaMalloc((void **)&device_dst_image1, width * height * ImageDefinitions::RGB_BPP);
    cudaMalloc((void **)&device_dst_image2, width * height * ImageDefinitions::RGB_BPP);
    cudaMalloc((void **)&device_dst_red, width * height * sizeof(float));
    cudaMalloc((void **)&device_dst_blue, width * height * sizeof(float));
    cudaMalloc((void **)&device_dst_green, width * height * sizeof(float));

    cudaMallocHost((void **)&final_image1, width * height * ImageDefinitions::RGB_BPP);
    cudaMallocHost((void **)&final_image2, width * height * ImageDefinitions::RGB_BPP);
    cudaMallocHost((void **)&host_dst_red, width * height * sizeof(float));
    cudaMallocHost((void **)&host_dst_blue, width * height * sizeof(float));
    cudaMallocHost((void **)&host_dst_green, width * height * sizeof(float));

    // Check if the image was loaded successfully
    if (image.empty())
    {
        std::cout << "Could not open or find the image!" << std::endl;
        return -1;
    }

    // Copy the CbYCr422 data to device memory
    cudaMemcpy(device_source_image, image.data, width * height * ImageDefinitions::RGB_BPP, cudaMemcpyHostToDevice);

    nppiRGB24ToCbYCr422(device_source_image, device_intermediate_storage, width, height);
    convert_CbYCr_To_BGR24(device_intermediate_storage, device_dst_image1, height, width);
    nppiCbYCr422ToBGR24(device_intermediate_storage, device_dst_image2, width, height);
    CbYCr_To_RGB_f32_clamped(device_intermediate_storage,
                             device_dst_red,
                             device_dst_green,
                             device_dst_blue,
                             width,
                             height);

    // Copy the result back to host memory
    cudaMemcpy(final_image1, device_dst_image1, width * height * ImageDefinitions::RGB_BPP, cudaMemcpyDeviceToHost);
    cudaMemcpy(final_image2, device_dst_image2, width * height * ImageDefinitions::RGB_BPP, cudaMemcpyDeviceToHost);

    cudaMemcpy(host_dst_red, device_dst_red, width * height * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_dst_green, device_dst_green, width * height * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_dst_blue, device_dst_blue, width * height * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Outputing the RGB value of 5 pixels...\n");
    for (int i = 0; i < 5; i++)
    {
        printf("Original Pixel RGB: %d %d %d\n", image.data[i * 3], image.data[i * 3 + 1], image.data[i * 3 + 2]);
        printf(" EXP New Pixel BGR: %d %d %d\n", final_image1[i * 3], final_image1[i * 3 + 1], final_image1[i * 3 + 2]);
        printf("     New Pixel BGR: %d %d %d\n", final_image2[i * 3], final_image2[i * 3 + 1], final_image2[i * 3 + 2]);
        printf("     F32 Pixel BGR: %f %f %f\n\n", host_dst_blue[i], host_dst_green[i], host_dst_red[i]);
    }

    cudaFree(device_source_image);
    cudaFree(device_intermediate_storage);
    cudaFree(device_dst_image1);
    cudaFree(device_dst_image2);
    cudaFree(device_dst_red);
    cudaFree(device_dst_blue);
    cudaFree(device_dst_green);
    cudaFreeHost(final_image1);
    cudaFreeHost(final_image2);
    cudaFreeHost(host_dst_red);
    cudaFreeHost(host_dst_blue);
    cudaFreeHost(host_dst_green);

    cudaDeviceReset();

    return 0;
}
