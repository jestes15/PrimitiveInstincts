#include "kernels.h"

#define PX1_BLU_CH 0
#define PX1_GRN_CH 1
#define PX1_RED_CH 2
#define PX2_BLU_CH 3
#define PX2_GRN_CH 4
#define PX2_RED_CH 5

#define TILE_WIDTH 32
#define TILE_HEIGHT 16

__global__ void cbycr422_to_bgr24(
    uint8_t *__restrict__ src, uint8_t *__restrict__ dst, int width, int height, int src_pitch, int dst_pitch)
{
    int x = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int64_t src_idx = y * src_pitch + 2 * x;
    int64_t dst_idx = y * dst_pitch + 3 * x;

    int32_t cb, y0, cr, y1;
    uint8_t *addr = src + src_idx;

    asm volatile(".reg .u8 t1;\n"
                 ".reg .u8 t2;\n"
                 ".reg .u8 t3;\n"
                 ".reg .u8 t4;\n"
                 ".reg .s32 t5;\n"
                 ".reg .s32 t6;\n"
                 "");
}

void convert_CbYCr_To_BGR24(uint8_t *__restrict__ src,
                            uint8_t *__restrict__ dst,
                            const uint32_t height,
                            const uint32_t width)
{
    dim3 block(32, 8);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    uint32_t src_pitch = width * image_sizes::CbYCr422_bytes_per_pixel;
    uint32_t dst_pitch = width * image_sizes::RGB24_bytes_per_pixel;

    cbycr422_to_bgr24<<<grid, block>>>(src, dst, width, height, src_pitch, dst_pitch);
    cudaDeviceSynchronize();
}