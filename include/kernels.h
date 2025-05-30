#ifndef __KERNELS_H__
#define __KERNELS_H__

#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <npp.h>

#include <stdint.h>
#include <stdio.h>

#include <functional>
#include <memory>

#include "device_structs.h"
#include "image_sizes.h"

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

void print_gpu_memory_u8(std::shared_ptr<uint8_t> src, uint32_t size);
void print_gpu_memory_f32(std::shared_ptr<float> src, uint32_t size);

#endif // __KERNELS_H__