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
                 // TODO: Make FMA
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
    cbycr422_to_bgr24<<<grid, block>>>(src, dst, width, height, width * 2, width * 3);
}

__global__ void cbycr422_to_bgr24_f32_clamped(uint8_t *__restrict__ src,
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

    // Calculate the reciprocal of 255, 1/255 with the intrinsic function __frcp
    // More information found here:
    // __frcp:
    // https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html?highlight=__frcp#_CPPv49__frcp_rzf
    float recip = 0.003922;

    int64_t src_idx = y * src_pitch + 2 * x;
    int64_t dst_idx = y * dst_pitch + x;
    int32_t cb, y0, cr, y1;
    uint8_t *addr = src + src_idx;
    float pair[2];
    float recip = 0.003922;

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

void convert_CbYCr_To_BGR24(uint8_t *__restrict__ src,
                            float *__restrict__ dst_r,
                            float *__restrict__ dst_g,
                            float *__restrict__ dst_b,
                            const uint32_t height,
                            const uint32_t width)
{
    uint32_t src_pitch = width * image_sizes::CbYCr422_bytes_per_pixel;
    uint32_t dst_pitch = width;

    dim3 block(TILE_WIDTH, TILE_HEIGHT);
    dim3 grid((width + (2 * TILE_WIDTH - 1)) / (2 * TILE_WIDTH), (height + TILE_HEIGHT - 1) / TILE_HEIGHT);

    cbycr422_to_bgr24_f32_clamped<<<grid, block>>>(src, dst_r, dst_g, dst_b, width, height, src_pitch, dst_pitch);
}

void resize_BGR24_HD_to_1984x1984(std::shared_ptr<float> src_r,
                                  std::shared_ptr<float> src_g,
                                  std::shared_ptr<float> src_b,
                                  std::shared_ptr<float> dst_r,
                                  std::shared_ptr<float> dst_g,
                                  std::shared_ptr<float> dst_b,
                                  int src_width,
                                  int src_height,
                                  NppStreamContext context)
{
    const Npp32f *src[3] = {src_r.get(), src_g.get(), src_b.get()};
    Npp32f *dst[3] = {dst_r.get(), dst_g.get(), dst_b.get()};
    NppiRect src_roi = {.x = 0, .y = 0, .width = src_width, .height = src_height};
    NppiRect dst_roi = {.x = 0, .y = 0, .width = 1984, .height = 1984};
    NppiSize src_size = {.width = src_width, .height = src_height};
    int32_t src_step = src_size.width * sizeof(float);
    int32_t dst_step = dst_roi.width * sizeof(float);
    double scale_factor = std::min(dst_roi.width / (src_width * 1.0), dst_roi.height / (src_height * 1.0));
    int new_width = std::ceil(scale_factor * src_width);
    int new_height = std::ceil(scale_factor * src_height);
    float padding_width = (dst_roi.width - new_width) / 2;
    float padding_height = (dst_roi.height - new_height) / 2;

    nppiResizeSqrPixel_32f_P3R_Ctx(src,
                                   src_size,
                                   src_step,
                                   src_roi,
                                   dst,
                                   dst_step,
                                   dst_roi,
                                   scale_factor,
                                   scale_factor,
                                   padding_width,
                                   padding_height,
                                   NPPI_INTER_LINEAR,
                                   context);
}

__global__ void print_kernel_u8(uint8_t *src, uint32_t size)
{
    for (int i = 0; i < size; i++)
        printf("src[%d] = %d\n", i, src[i]);
}

void print_gpu_memory_u8(std::shared_ptr<uint8_t> src, uint32_t size)
{
    printf("LOG: INFO -- %s entered\n", __PRETTY_FUNCTION__);
    dim3 block(1, 1, 1);
    dim3 grid(1, 1, 1);

    print_kernel_u8<<<grid, block>>>(src.get(), size);
}

__global__ void print_kernel_f32(float *src, uint32_t size)
{
    for (int i = 0; i < size; i++)
        printf("src[%d] = %f\n", i, src[i]);
}

void print_gpu_memory_f32(std::shared_ptr<float> src, uint32_t size)
{
    printf("LOG: INFO -- %s entered\n", __PRETTY_FUNCTION__);
    dim3 block(1, 1, 1);
    dim3 grid(1, 1, 1);

    print_kernel_f32<<<grid, block>>>(src.get(), size);
}