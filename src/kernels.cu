#include "kernels.h"

#define PX1_BLU_CH 0
#define PX1_GRN_CH 1
#define PX1_RED_CH 2
#define PX2_BLU_CH 3
#define PX2_GRN_CH 4
#define PX2_RED_CH 5

#define TILE_WIDTH 32
#define TILE_HEIGHT 16

__device__ uchar6 cbycr_to_bgr(uint32_t cbycr_byte)
{
    int32_t cb = (cbycr_byte & 0x000000FF) - 128;
    int32_t y0 = ((cbycr_byte & 0x0000FF00) >> 8) - 16;
    int32_t cr = ((cbycr_byte & 0x00FF0000) >> 16) - 128;
    int32_t y1 = ((cbycr_byte & 0xFF000000) >> 24) - 16;

    int32_t _r0 = (1164 * y0 + 1596 * cr) / 1000;
    int32_t _r1 = (1164 * y1 + 1596 * cr) / 1000;

    int32_t _g0 = (1164 * y0 - 813 * cr - 392 * cb) / 1000;
    int32_t _g1 = (1164 * y1 - 813 * cr - 392 * cb) / 1000;

    int32_t _b0 = (1164 * y0 + 2018 * cb) / 1000;
    int32_t _b1 = (1164 * y1 + 2018 * cb) / 1000;

    uchar6 data = {.r0 = static_cast<uint8_t>(min(max(_r0, 0), 255)),
                   .g0 = static_cast<uint8_t>(min(max(_g0, 0), 255)),
                   .b0 = static_cast<uint8_t>(min(max(_b0, 0), 255)),
                   .r1 = static_cast<uint8_t>(min(max(_r1, 0), 255)),
                   .g1 = static_cast<uint8_t>(min(max(_g1, 0), 255)),
                   .b1 = static_cast<uint8_t>(min(max(_b1, 0), 255))};

    return data;
}

__global__ void cbycr422_to_bgr24(const uint8_t *__restrict__ src,
                                  uint8_t *__restrict__ dst,
                                  const uint32_t width,
                                  const uint32_t height,
                                  const uint32_t src_stride,
                                  const uint32_t dst_stride,
                                  const uint32_t src_pitch,
                                  const uint32_t dst_pitch)
{
    // __shared__ alignas(32) uint8_t bgr_tile[TILE_WIDTH * TILE_WIDTH * image_sizes::RGB24_bytes_per_pixel * 2];

    uint32_t cbycr_byte = 0;
    uint32_t row = blockDim.y * blockIdx.y + threadIdx.y;
    uint32_t col = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t src_index = row * width + col;
    uint32_t dst_index = 6 * src_index;
    // uint16_t th_idx = (threadIdx.y * TILE_WIDTH * TILE_WIDTH) + (threadIdx.x * TILE_WIDTH);

    while (src_index < src_pitch && dst_index < dst_pitch)
    {
        cbycr_byte = ((uint32_t *)src)[src_index];
        uchar6 data = cbycr_to_bgr(cbycr_byte);

        dst[dst_index + PX1_BLU_CH] = data.b0;
        dst[dst_index + PX1_GRN_CH] = data.g0;
        dst[dst_index + PX1_RED_CH] = data.r0;
        dst[dst_index + PX2_BLU_CH] = data.b1;
        dst[dst_index + PX2_GRN_CH] = data.g1;
        dst[dst_index + PX2_RED_CH] = data.r1;

        src_index += src_stride;
        dst_index += dst_stride;
    }
}

void convert_CbYCr_To_BGR24(const uint8_t *__restrict__ src,
                            uint8_t *__restrict__ dst,
                            const uint32_t height,
                            const uint32_t width)
{
    dim3 block(32, 8);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    uint32_t src_stride = block.x * grid.x * grid.y;
    uint32_t dst_stride = block.x * grid.x * grid.y * 6;

    uint32_t src_step = width * height * image_sizes::CbYCr422_bytes_per_pixel;
    uint32_t dst_step = width * height * image_sizes::RGB24_bytes_per_pixel;

    cbycr422_to_bgr24<<<grid, block>>>(src, dst, width, height, src_stride, dst_stride, src_step, dst_step);
    cudaDeviceSynchronize();
}