#ifndef __KERNELS_H__
#define __KERNELS_H__

#include "image_sizes.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <ipp.h>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/opencv.hpp>

#include <functional>
#include <memory>
#include <type_traits>
#include <npp.h>
#include <stdint.h>
#include <stdio.h>

enum
{
    BASIC_IMPL_NVIDIA_BT601 = 0,
    BASIC_IMPL_NVIDIA_BT709,
    BASIC_IMPL_INTEL_BT601,
    BASIC_IMPL_INTEL_BT709,
    FMA_IMPL_NVIDIA_BT601,
    FMA_IMPL_NVIDIA_BT709,
    FMA_IMPL_INTEL_BT601,
    FMA_IMPL_INTEL_BT601_FP16,
    FMA_IMPL_INTEL_BT601_FP16_TO_FP32,
    FMA_IMPL_INTEL_BT709
};

#define TILE_WIDTH 32
#define TILE_HEIGHT 4

union conversion_union
{
    uint32_t double_word;
    struct
    {
        uint8_t cb;
        uint8_t y0;
        uint8_t cr;
        uint8_t y1;
    };
};

// cv::cuda::GpuMat ipp_convertAndResize(uint8_t *cbycr_image);
cv::Mat ipp_convertAndResize(uint8_t *cbycr_image);

int nppiCbYCr422ToBGR24(const uint8_t *__restrict__ device_src,
                        uint8_t *__restrict__ device_dst,
                        int width,
                        int height,
                        NppStreamContext context,
                        const NppiRect &ROI_Size_Loc = {0, 0, 0, 0});
void convert_CbYCrToBGR24_f32_P3R_approx(uint8_t *__restrict__ src,
                                         float *__restrict__ int_f32_HD_P3R,
                                         int32_t input_width,
                                         int32_t input_height,
                                         float *__restrict__ dst_f32_SQ_P3R,
                                         int sq_width,
                                         int sq_height,
                                         NppStreamContext context);

void convert_CbYCrToBGR24_f32_P3R(uint8_t *__restrict__ src,
                                  float *__restrict__ int_f32_HD_P3R,
                                  int32_t input_width,
                                  int32_t input_height,
                                  float *__restrict__ dst_f32_SQ_P3R,
                                  int sq_width,
                                  int sq_height,
                                  NppStreamContext context,
                                  uint8_t conv_type);

#endif // __KERNELS_H__
