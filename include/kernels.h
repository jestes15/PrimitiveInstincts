#ifndef __KERNELS_H__
#define __KERNELS_H__

#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <npp.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <stdint.h>
#include <stdio.h>

#include <functional>
#include <memory>

#include "device_structs.h"
#include "image_sizes.h"

constexpr int HD_WIDTH = 1920;
constexpr int HD_HEIGHT = 1080;
constexpr int SQ_WIDTH = 1984;
constexpr int SQ_HEIGHT = 1084;
constexpr int RGB_BPP = 3;
constexpr int UYVY_BPP = 2;

constexpr int src_size = HD_WIDTH * HD_HEIGHT * UYVY_BPP;
constexpr int dst_rgb_size = HD_WIDTH * HD_HEIGHT * RGB_BPP;
constexpr int int_f32_size = HD_WIDTH * HD_HEIGHT;
constexpr int dst_f32_size = SQ_WIDTH * SQ_HEIGHT;

int nppiRGB24ToCbYCr422(const uint8_t *__restrict__ device_src,
                        uint8_t *__restrict__ device_dst,
                        int width,
                        int height,
                        NppStreamContext context,
                        const NppiRect &ROI_Size_Loc = {0, 0, 0, 0});
int nppiCbYCr422ToBGR24(const uint8_t *__restrict__ device_src,
                        uint8_t *__restrict__ device_dst,
                        int width,
                        int height,
                        NppStreamContext context,
                        const NppiRect &ROI_Size_Loc = {0, 0, 0, 0});

void convert_CbYCr_To_BGR24(uint8_t *__restrict__ src,
                            float *__restrict__ dst_r,
                            float *__restrict__ dst_g,
                            float *__restrict__ dst_b,
                            const uint32_t height,
                            const uint32_t width);

void resize_BGR24_HD_to_1984x1984(std::shared_ptr<float> src_r,
                                  std::shared_ptr<float> src_g,
                                  std::shared_ptr<float> src_b,
                                  std::shared_ptr<float> dst_r,
                                  std::shared_ptr<float> dst_g,
                                  std::shared_ptr<float> dst_b,
                                  int src_width,
                                  int src_height,
                                  NppStreamContext context);

void convert_CbYCrTOBGR24_u8C3R_f32_P3R(thrust::device_vector<uint8_t> &cbycr422_src,
                                        thrust::host_vector<uint8_t> &h_bgr24,
                                        thrust::device_vector<uint8_t> &d_bgr24,
                                        thrust::device_vector<float> &int_f32_HD_P3R,
                                        thrust::device_vector<float> &dst_f32_SQ_P3R,
                                        NppStreamContext context,
                                        cudaStream_t cpAsyncStream);

void print_gpu_memory_u8(std::shared_ptr<uint8_t> src, uint32_t size);
void print_gpu_memory_f32(std::shared_ptr<float> src, uint32_t size);

#endif // __KERNELS_H__