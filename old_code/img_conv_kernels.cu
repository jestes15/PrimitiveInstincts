#include <kernels.h>

#include <cudaProfiler.h>
#include <cuda_fp16.h>

struct half_constants
{
    half recip;
    half y_multiplicand;
    half y_addend;
    half cr_rpx_multiplicand;
    half cr_rpx_addend;
    half cr_gpx_multiplicand;
    half cr_gpx_addend;
    half cb_gpx_multiplicand;
    half cb_gpx_addend;
    half cb_bpx_multiplicand;
    half cb_bpx_addend;
};

__global__ void convertCbYCrToBGR24_f32_P3R_BASIC_IMPL_NVIDIA_BT601(uint8_t *__restrict__ src,
                                                                    float *__restrict__ dst_r,
                                                                    float *__restrict__ dst_g,
                                                                    float *__restrict__ dst_b,
                                                                    const int width,
                                                                    const int height,
                                                                    const int src_pitch,
                                                                    int dst_f32_pitch)
{

    conversion_union conv;

    int x = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    float cb, y0, cr, y1;
    float2 pixel;
    float2 red, green, blue;

    constexpr float recip = 0.003922f;
    int64_t src_u8_idx = y * src_pitch + 2 * x;
    int64_t dst_f32_idx = y * dst_f32_pitch + x;
    uint8_t *addr = src + src_u8_idx;

    conv.double_word = *(uint32_t *)addr;
    cb = static_cast<float>(conv.cb) - 128.0f;
    y0 = static_cast<float>(conv.y0);
    cr = static_cast<float>(conv.cr) - 128.0f;
    y1 = static_cast<float>(conv.y1);

    red.x = y0 + 1.596f * cr + 0.5f;
    red.y = y1 + 1.596f * cr + 0.5f;
    green.x = y0 - 0.813f * cr - 0.392f * cb + 0.5f;
    green.y = y1 - 0.813f * cr - 0.392f * cb + 0.5f;
    blue.x = y0 + 2.017f * cb + 0.5f;
    blue.y = y1 + 2.017f * cb + 0.5f;

    pixel.x = __saturatef(__fmul_rn(red.x, recip));
    pixel.y = __saturatef(__fmul_rn(red.y, recip));
    *(uint64_t *)&dst_r[dst_f32_idx] = *(uint64_t *)&pixel;

    pixel.x = __saturatef(__fmul_rn(green.x, recip));
    pixel.y = __saturatef(__fmul_rn(green.y, recip));
    *(uint64_t *)&dst_g[dst_f32_idx] = *(uint64_t *)&pixel;

    pixel.x = __saturatef(__fmul_rn(blue.x, recip));
    pixel.y = __saturatef(__fmul_rn(blue.y, recip));
    *(uint64_t *)&dst_g[dst_f32_idx] = *(uint64_t *)&pixel;
}

__global__ void convertCbYCrToBGR24_f32_P3R_BASIC_IMPL_NVIDIA_BT709(uint8_t *__restrict__ src,
                                                                    float *__restrict__ dst_r,
                                                                    float *__restrict__ dst_g,
                                                                    float *__restrict__ dst_b,
                                                                    const int width,
                                                                    const int height,
                                                                    const int src_pitch,
                                                                    int dst_f32_pitch)
{

    conversion_union conv;

    int x = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    float cb, y0, cr, y1;
    float2 pixel;
    float2 red, green, blue;

    constexpr float recip = 0.003922f;
    int64_t src_u8_idx = y * src_pitch + 2 * x;
    int64_t dst_f32_idx = y * dst_f32_pitch + x;
    uint8_t *addr = src + src_u8_idx;

    conv.double_word = *(uint32_t *)addr;
    cb = static_cast<float>(conv.cb) - 128.0f;
    y0 = static_cast<float>(conv.y0);
    cr = static_cast<float>(conv.cr) - 128.0f;
    y1 = static_cast<float>(conv.y1);

    red.x = y0 + 1.540f * cr + 0.5f;
    red.y = y1 + 1.540f * cr + 0.5f;
    green.x = y0 - 0.459f * cr - 0.183f * cb + 0.5f;
    green.y = y1 - 0.459f * cr - 0.183f * cb + 0.5f;
    blue.x = y0 + 1.816f * cb + 0.5f;
    blue.y = y1 + 1.816f * cb + 0.5f;

    pixel.x = __saturatef(__fmul_rn(red.x, recip));
    pixel.y = __saturatef(__fmul_rn(red.y, recip));
    *(uint64_t *)&dst_r[dst_f32_idx] = *(uint64_t *)&pixel;

    pixel.x = __saturatef(__fmul_rn(green.x, recip));
    pixel.y = __saturatef(__fmul_rn(green.y, recip));
    *(uint64_t *)&dst_g[dst_f32_idx] = *(uint64_t *)&pixel;

    pixel.x = __saturatef(__fmul_rn(blue.x, recip));
    pixel.y = __saturatef(__fmul_rn(blue.y, recip));
    *(uint64_t *)&dst_g[dst_f32_idx] = *(uint64_t *)&pixel;
}

__global__ void convertCbYCrToBGR24_f32_P3R_BASIC_IMPL_INTEL_BT601(uint8_t *__restrict__ src,
                                                                   float *__restrict__ dst_r,
                                                                   float *__restrict__ dst_g,
                                                                   float *__restrict__ dst_b,
                                                                   const int width,
                                                                   const int height,
                                                                   const int src_pitch,
                                                                   int dst_f32_pitch)
{

    conversion_union conv;

    int x = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    float cb, y0, cr, y1;
    float2 pixel;
    float2 red, green, blue;

    constexpr float recip = 0.003922f;
    int64_t src_u8_idx = y * src_pitch + 2 * x;
    int64_t dst_f32_idx = y * dst_f32_pitch + x;
    uint8_t *addr = src + src_u8_idx;

    conv.double_word = *(uint32_t *)addr;
    cb = static_cast<float>(conv.cb) - 128.0f;
    y0 = static_cast<float>(conv.y0) - 16.0f;
    cr = static_cast<float>(conv.cr) - 128.0f;
    y1 = static_cast<float>(conv.y1) - 16.0f;

    red.x = 1.164f * y0 + 1.596f * cr;
    red.y = 1.164f * y1 + 1.596f * cr;
    green.x = 1.164f * y0 - 0.813f * cr - 0.392f * cb;
    green.y = 1.164f * y1 - 0.813f * cr - 0.392f * cb;
    blue.x = 1.164f * y0 + 2.017f * cb;
    blue.y = 1.164f * y1 + 2.017f * cb;

    pixel.x = __saturatef(__fmul_rn(red.x, recip));
    pixel.y = __saturatef(__fmul_rn(red.y, recip));
    *(uint64_t *)&dst_r[dst_f32_idx] = *(uint64_t *)&pixel;

    pixel.x = __saturatef(__fmul_rn(green.x, recip));
    pixel.y = __saturatef(__fmul_rn(green.y, recip));
    *(uint64_t *)&dst_g[dst_f32_idx] = *(uint64_t *)&pixel;

    pixel.x = __saturatef(__fmul_rn(blue.x, recip));
    pixel.y = __saturatef(__fmul_rn(blue.y, recip));
    *(uint64_t *)&dst_g[dst_f32_idx] = *(uint64_t *)&pixel;
}

__global__ void convertCbYCrToBGR24_f32_P3R_BASIC_IMPL_INTEL_BT709(uint8_t *__restrict__ src,
                                                                   float *__restrict__ dst_r,
                                                                   float *__restrict__ dst_g,
                                                                   float *__restrict__ dst_b,
                                                                   const int width,
                                                                   const int height,
                                                                   const int src_pitch,
                                                                   int dst_f32_pitch)
{

    conversion_union conv;

    int x = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    float cb, y0, cr, y1;
    float2 pixel;
    float2 red, green, blue;

    constexpr float recip = 0.003922f;
    int64_t src_u8_idx = y * src_pitch + 2 * x;
    int64_t dst_f32_idx = y * dst_f32_pitch + x;
    uint8_t *addr = src + src_u8_idx;

    conv.double_word = *(uint32_t *)addr;
    cb = static_cast<float>(conv.cb) - 128.0f;
    y0 = static_cast<float>(conv.y0);
    cr = static_cast<float>(conv.cr) - 128.0f;
    y1 = static_cast<float>(conv.y1);

    red.x = y0 + 1.540f * cr;
    red.y = y1 + 1.540f * cr;
    green.x = y0 - 0.459f * cr - 0.183f * cb;
    green.y = y1 - 0.459f * cr - 0.183f * cb;
    blue.x = y0 + 1.816f * cb;
    blue.y = y1 + 1.816f * cb;

    pixel.x = __saturatef(__fmul_rn(red.x, recip));
    pixel.y = __saturatef(__fmul_rn(red.y, recip));
    *(uint64_t *)&dst_r[dst_f32_idx] = *(uint64_t *)&pixel;

    pixel.x = __saturatef(__fmul_rn(green.x, recip));
    pixel.y = __saturatef(__fmul_rn(green.y, recip));
    *(uint64_t *)&dst_g[dst_f32_idx] = *(uint64_t *)&pixel;

    pixel.x = __saturatef(__fmul_rn(blue.x, recip));
    pixel.y = __saturatef(__fmul_rn(blue.y, recip));
    *(uint64_t *)&dst_g[dst_f32_idx] = *(uint64_t *)&pixel;
}

__global__ void convertCbYCrToBGR24_f32_P3R_FMA_IMPL_NVIDIA_BT601(uint8_t *__restrict__ src,
                                                                  float *__restrict__ dst_r,
                                                                  float *__restrict__ dst_g,
                                                                  float *__restrict__ dst_b,
                                                                  const int width,
                                                                  const int height,
                                                                  const int src_pitch,
                                                                  int dst_f32_pitch)
{

    conversion_union conv;

    int x = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    float cb, y0, cr, y1;
    float cr_rpx, cr_gpx, cb_gpx, cb_bpx, cbcr_gpx;
    float2 pixel;
    float2 red, green, blue;

    constexpr float recip = 0.003922f;
    int64_t src_u8_idx = y * src_pitch + 2 * x;
    int64_t dst_f32_idx = y * dst_f32_pitch + x;
    uint8_t *addr = src + src_u8_idx;

    conv.double_word = *(uint32_t *)addr;
    cb = static_cast<float>(conv.cb);
    y0 = static_cast<float>(conv.y0);
    cr = static_cast<float>(conv.cr);
    y1 = static_cast<float>(conv.y1);

    // FMAF Operations
    cr_rpx = __fmaf_ieee_rn(1.596, cr, -220.788);
    cr_gpx = __fmaf_ieee_rn(-0.813, cr, 88.064);
    cb_gpx = __fmaf_ieee_rn(-0.392, cb, 50.676);
    cb_bpx = __fmaf_ieee_rn(2.017, cb, -274.176);

    // FADD Operations
    cbcr_gpx = __fadd_rn(cr_gpx, cb_gpx);
    red.x = __fadd_rn(y0, cr_rpx);
    red.y = __fadd_rn(y1, cr_rpx);
    green.x = __fadd_rn(y0, cbcr_gpx);
    green.y = __fadd_rn(y1, cbcr_gpx);
    blue.x = __fadd_rn(y0, cb_bpx);
    blue.y = __fadd_rn(y1, cb_bpx);

    pixel.x = __saturatef(__fmul_rn(red.x, recip));
    pixel.y = __saturatef(__fmul_rn(red.y, recip));
    *(uint64_t *)&dst_r[dst_f32_idx] = *(uint64_t *)&pixel;

    pixel.x = __saturatef(__fmul_rn(green.x, recip));
    pixel.y = __saturatef(__fmul_rn(green.y, recip));
    *(uint64_t *)&dst_g[dst_f32_idx] = *(uint64_t *)&pixel;

    pixel.x = __saturatef(__fmul_rn(blue.x, recip));
    pixel.y = __saturatef(__fmul_rn(blue.y, recip));
    *(uint64_t *)&dst_g[dst_f32_idx] = *(uint64_t *)&pixel;
}

__global__ void convertCbYCrToBGR24_f32_P3R_FMA_IMPL_NVIDIA_BT709(uint8_t *__restrict__ src,
                                                                  float *__restrict__ dst_r,
                                                                  float *__restrict__ dst_g,
                                                                  float *__restrict__ dst_b,
                                                                  const int width,
                                                                  const int height,
                                                                  const int src_pitch,
                                                                  int dst_f32_pitch)
{

    conversion_union conv;

    int x = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    float cb, y0, cr, y1;
    float cr_rpx, cr_gpx, cb_gpx, cb_bpx, cbcr_gpx;
    float2 pixel;
    float2 red, green, blue;

    constexpr float recip = 0.003922f;
    int64_t src_u8_idx = y * src_pitch + 2 * x;
    int64_t dst_f32_idx = y * dst_f32_pitch + x;
    uint8_t *addr = src + src_u8_idx;

    conv.double_word = *(uint32_t *)addr;
    cb = static_cast<float>(conv.cb);
    y0 = static_cast<float>(conv.y0);
    cr = static_cast<float>(conv.cr);
    y1 = static_cast<float>(conv.y1);

    // FMAF Operations
    cr_rpx = __fmaf_ieee_rn(1.540, cr, -196.62);
    cr_gpx = __fmaf_ieee_rn(-0.459, cr, 58.752);
    cb_gpx = __fmaf_ieee_rn(-0.183, cb, 23.924);
    cb_bpx = __fmaf_ieee_rn(1.816, cb, -231.948);

    // FADD Operations
    cbcr_gpx = __fadd_rn(cr_gpx, cb_gpx);
    red.x = __fadd_rn(y0, cr_rpx);
    red.y = __fadd_rn(y1, cr_rpx);
    green.x = __fadd_rn(y0, cbcr_gpx);
    green.y = __fadd_rn(y1, cbcr_gpx);
    blue.x = __fadd_rn(y0, cb_bpx);
    blue.y = __fadd_rn(y1, cb_bpx);

    pixel.x = __saturatef(__fmul_rn(red.x, recip));
    pixel.y = __saturatef(__fmul_rn(red.y, recip));
    *(uint64_t *)&dst_r[dst_f32_idx] = *(uint64_t *)&pixel;

    pixel.x = __saturatef(__fmul_rn(green.x, recip));
    pixel.y = __saturatef(__fmul_rn(green.y, recip));
    *(uint64_t *)&dst_g[dst_f32_idx] = *(uint64_t *)&pixel;

    pixel.x = __saturatef(__fmul_rn(blue.x, recip));
    pixel.y = __saturatef(__fmul_rn(blue.y, recip));
    *(uint64_t *)&dst_g[dst_f32_idx] = *(uint64_t *)&pixel;
}

__global__ void convertCbYCrToBGR24_f32_P3R_FMA_IMPL_INTEL_BT601(uint8_t *__restrict__ src,
                                                                 float *__restrict__ dst_r,
                                                                 float *__restrict__ dst_g,
                                                                 float *__restrict__ dst_b,
                                                                 const int width,
                                                                 const int height,
                                                                 const int src_pitch,
                                                                 int dst_f32_pitch)
{

    conversion_union conv;

    int x = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    float cb, y0, cr, y1;
    float scaled_y0, scaled_y1;
    float cr_rpx, cr_gpx, cb_gpx, cb_bpx, cbcr_gpx;
    float2 pixel;
    float2 red, green, blue;

    constexpr float recip = 0.003922f;
    int64_t src_u8_idx = y * src_pitch + 2 * x;
    int64_t dst_f32_idx = y * dst_f32_pitch + x;
    uint8_t *addr = src + src_u8_idx;

    conv.double_word = *(uint32_t *)addr;
    cb = static_cast<float>(conv.cb);
    y0 = static_cast<float>(conv.y0);
    cr = static_cast<float>(conv.cr);
    y1 = static_cast<float>(conv.y1);

    // FMAF Operations
    scaled_y0 = __fmaf_ieee_rn(1.164, y0, -18.624);
    scaled_y1 = __fmaf_ieee_rn(1.164, y1, -18.624);
    cr_rpx = __fmaf_ieee_rn(1.596, cr, -204.288);
    cr_gpx = __fmaf_ieee_rn(-0.813, cr, 104.064);
    cb_gpx = __fmaf_ieee_rn(-0.392, cb, 50.176);
    cb_bpx = __fmaf_ieee_rn(2.017, cb, -258.176);

    // FADD Operations
    cbcr_gpx = __fadd_rn(cr_gpx, cb_gpx);
    red.x = __fadd_rn(scaled_y0, cr_rpx);
    red.y = __fadd_rn(scaled_y1, cr_rpx);
    green.x = __fadd_rn(scaled_y0, cbcr_gpx);
    green.y = __fadd_rn(scaled_y1, cbcr_gpx);
    blue.x = __fadd_rn(scaled_y0, cb_bpx);
    blue.y = __fadd_rn(scaled_y1, cb_bpx);

    pixel.x = __saturatef(__fmul_rn(red.x, recip));
    pixel.y = __saturatef(__fmul_rn(red.y, recip));
    *(uint64_t *)&dst_r[dst_f32_idx] = *(uint64_t *)&pixel;

    pixel.x = __saturatef(__fmul_rn(green.x, recip));
    pixel.y = __saturatef(__fmul_rn(green.y, recip));
    *(uint64_t *)&dst_g[dst_f32_idx] = *(uint64_t *)&pixel;

    pixel.x = __saturatef(__fmul_rn(blue.x, recip));
    pixel.y = __saturatef(__fmul_rn(blue.y, recip));
    *(uint64_t *)&dst_g[dst_f32_idx] = *(uint64_t *)&pixel;
}

__global__ void convertCbYCrToBGR24_F16_P3R_FMA_IMPL_INTEL_BT601_FP16(uint8_t *__restrict__ src,
                                                                      half *__restrict__ dst_r,
                                                                      half *__restrict__ dst_g,
                                                                      half *__restrict__ dst_b,
                                                                      const int width,
                                                                      const int height,
                                                                      const int src_pitch,
                                                                      const int dst_f32_pitch,
                                                                      const half_constants constants)
{
    conversion_union conv;

    int x = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    half cb, y0, cr, y1;
    half scaled_y0, scaled_y1;
    half cr_rpx, cr_gpx, cb_gpx, cb_bpx, cbcr_gpx;
    half2 red, green, blue;
    half2 pixel;

    int64_t src_u8_idx = y * src_pitch + 2 * x;
    int64_t dst_f32_idx = y * dst_f32_pitch + x;
    uint8_t *addr = src + src_u8_idx;

    conv.double_word = *(uint32_t *)addr;
    cb = __float2half(conv.cb);
    y0 = __float2half(conv.y0);
    cr = __float2half(conv.cr);
    y1 = __float2half(conv.y1);

    // FMAF Operations
    scaled_y0 = __hfma(constants.y_multiplicand, y0, constants.y_addend);
    scaled_y1 = __hfma(constants.y_multiplicand, y1, constants.y_addend);
    cr_rpx = __hfma(constants.cr_rpx_multiplicand, cr, constants.cr_rpx_addend);
    cr_gpx = __hfma(constants.cr_gpx_multiplicand, cr, constants.cr_gpx_addend);
    cb_gpx = __hfma(constants.cb_gpx_multiplicand, cb, constants.cb_gpx_addend);
    cb_bpx = __hfma(constants.cb_bpx_multiplicand, cb, constants.cb_bpx_addend);

    // FADD Operations
    cbcr_gpx = __hadd(cr_gpx, cb_gpx);
    red.x = __hadd(scaled_y0, cr_rpx);
    red.y = __hadd(scaled_y1, cr_rpx);

    green.x = __hadd(scaled_y0, cbcr_gpx);
    green.y = __hadd(scaled_y1, cbcr_gpx);

    blue.x = __hadd(scaled_y0, cb_bpx);
    blue.y = __hadd(scaled_y1, cb_bpx);

    pixel.x = __hmul_sat(red.x, constants.recip);
    pixel.y = __hmul_sat(red.y, constants.recip);
    *(uint32_t *)&dst_r[dst_f32_idx] = *(uint32_t *)&pixel;

    pixel.x = __hmul_sat(green.x, constants.recip);
    pixel.y = __hmul_sat(green.y, constants.recip);
    *(uint32_t *)&dst_g[dst_f32_idx] = *(uint32_t *)&pixel;

    pixel.x = __hmul_sat(blue.x, constants.recip);
    pixel.y = __hmul_sat(blue.y, constants.recip);
    *(uint32_t *)&dst_g[dst_f32_idx] = *(uint32_t *)&pixel;
}

__global__ void convertCbYCrToBGR24_F32_P3R_FMA_IMPL_INTEL_BT601_FP16(uint8_t *__restrict__ src,
                                                                      float *__restrict__ dst_r,
                                                                      float *__restrict__ dst_g,
                                                                      float *__restrict__ dst_b,
                                                                      const int width,
                                                                      const int height,
                                                                      const int src_pitch,
                                                                      const int dst_f32_pitch,
                                                                      const half_constants constants)
{
    conversion_union conv;

    int x = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    half cb, y0, cr, y1;
    half scaled_y0, scaled_y1;
    half cr_rpx, cr_gpx, cb_gpx, cb_bpx, cbcr_gpx;
    half2 red, green, blue;
    float2 pixel;

    int64_t src_u8_idx = y * src_pitch + 2 * x;
    int64_t dst_f32_idx = y * dst_f32_pitch + x;
    uint8_t *addr = src + src_u8_idx;

    conv.double_word = *(uint32_t *)addr;
    cb = __float2half(conv.cb);
    y0 = __float2half(conv.y0);
    cr = __float2half(conv.cr);
    y1 = __float2half(conv.y1);

    // FMAF Operations
    scaled_y0 = __hfma(constants.y_multiplicand, y0, constants.y_addend);
    scaled_y1 = __hfma(constants.y_multiplicand, y1, constants.y_addend);
    cr_rpx = __hfma(constants.cr_rpx_multiplicand, cr, constants.cr_rpx_addend);
    cr_gpx = __hfma(constants.cr_gpx_multiplicand, cr, constants.cr_gpx_addend);
    cb_gpx = __hfma(constants.cb_gpx_multiplicand, cb, constants.cb_gpx_addend);
    cb_bpx = __hfma(constants.cb_bpx_multiplicand, cb, constants.cb_bpx_addend);

    // FADD Operations
    cbcr_gpx = __hadd(cr_gpx, cb_gpx);
    red.x = __hadd(scaled_y0, cr_rpx);
    red.y = __hadd(scaled_y1, cr_rpx);

    green.x = __hadd(scaled_y0, cbcr_gpx);
    green.y = __hadd(scaled_y1, cbcr_gpx);

    blue.x = __hadd(scaled_y0, cb_bpx);
    blue.y = __hadd(scaled_y1, cb_bpx);

    pixel.x = __half2float(__hmul_sat(red.x, constants.recip));
    pixel.y = __half2float(__hmul_sat(red.y, constants.recip));
    *(uint64_t *)&dst_r[dst_f32_idx] = *(uint64_t *)&pixel;

    pixel.x = __half2float(__hmul_sat(green.x, constants.recip));
    pixel.y = __half2float(__hmul_sat(green.y, constants.recip));
    *(uint32_t *)&dst_g[dst_f32_idx] = *(uint64_t *)&pixel;

    pixel.x = __half2float(__hmul_sat(blue.x, constants.recip));
    pixel.y = __half2float(__hmul_sat(blue.y, constants.recip));
    *(uint64_t *)&dst_g[dst_f32_idx] = *(uint64_t *)&pixel;
}

__global__ void convertCbYCrToBGR24_f32_P3R_FMA_IMPL_INTEL_BT709(uint8_t *__restrict__ src,
                                                                 float *__restrict__ dst_r,
                                                                 float *__restrict__ dst_g,
                                                                 float *__restrict__ dst_b,
                                                                 const int width,
                                                                 const int height,
                                                                 const int src_pitch,
                                                                 int dst_f32_pitch)
{

    conversion_union conv;

    int x = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    float cb, y0, cr, y1;
    float cr_rpx, cr_gpx, cb_gpx, cb_bpx, cbcr_gpx;
    float2 pixel;
    float2 red, green, blue;

    constexpr float recip = 0.003922f;
    int64_t src_u8_idx = y * src_pitch + 2 * x;
    int64_t dst_f32_idx = y * dst_f32_pitch + x;
    uint8_t *addr = src + src_u8_idx;

    conv.double_word = *(uint32_t *)addr;
    cb = static_cast<float>(conv.cb);
    y0 = static_cast<float>(conv.y0);
    cr = static_cast<float>(conv.cr);
    y1 = static_cast<float>(conv.y1);

    // FMAF Operations
    cr_rpx = __fmaf_ieee_rn(1.540, cr, -197.12);
    cr_gpx = __fmaf_ieee_rn(-0.459, cr, 58.752);
    cb_gpx = __fmaf_ieee_rn(-0.183, cb, 23.424);
    cb_bpx = __fmaf_ieee_rn(1.816, cb, -232.448);

    // FADD Operations
    cbcr_gpx = __fadd_rn(cr_gpx, cb_gpx);
    red.x = __fadd_rn(y0, cr_rpx);
    red.y = __fadd_rn(y1, cr_rpx);
    green.x = __fadd_rn(y0, cbcr_gpx);
    green.y = __fadd_rn(y1, cbcr_gpx);
    blue.x = __fadd_rn(y0, cb_bpx);
    blue.y = __fadd_rn(y1, cb_bpx);

    pixel.x = __saturatef(__fmul_rn(red.x, recip));
    pixel.y = __saturatef(__fmul_rn(red.y, recip));
    *(uint64_t *)&dst_r[dst_f32_idx] = *(uint64_t *)&pixel;

    pixel.x = __saturatef(__fmul_rn(green.x, recip));
    pixel.y = __saturatef(__fmul_rn(green.y, recip));
    *(uint64_t *)&dst_g[dst_f32_idx] = *(uint64_t *)&pixel;

    pixel.x = __saturatef(__fmul_rn(blue.x, recip));
    pixel.y = __saturatef(__fmul_rn(blue.y, recip));
    *(uint64_t *)&dst_g[dst_f32_idx] = *(uint64_t *)&pixel;
}

void convert_CbYCrToBGR24_f32_P3R(uint8_t *__restrict__ src,
                                  float *__restrict__ int_f32_HD_P3R,
                                  int32_t input_width,
                                  int32_t input_height,
                                  float *__restrict__ dst_f32_SQ_P3R,
                                  int sq_width,
                                  int sq_height,
                                  NppStreamContext context,
                                  uint8_t conv_type)
{
    half *contiguos_block = nullptr;
    half *contiguos_block_p2 = nullptr;
    half *f16_r = nullptr, *f16_g = nullptr, *f16_b = nullptr;
    half *f16_dst_r = nullptr, *f16_dst_g = nullptr, *f16_dst_b = nullptr;

    if (conv_type == FMA_IMPL_INTEL_BT601_FP16)
    {
        cudaMalloc((void **)&contiguos_block, sizeof(half) * input_width * input_height * 3);
        cudaMalloc((void **)&contiguos_block_p2, sizeof(half) * sq_width * sq_height * 3);

        f16_r = contiguos_block;
        f16_g = contiguos_block + (input_width * input_height);
        f16_b = contiguos_block + (2 * input_width * input_height);

        f16_dst_r = contiguos_block_p2;
        f16_dst_g = contiguos_block_p2 + (sq_width * sq_height);
        f16_dst_b = contiguos_block_p2 + (2 * sq_width * sq_height);
    }

    // Calculate pitch for input and output vectors
    int src_pitch = input_width * ImageDefinitions::UYVY_BPP;
    int int_f32_pitch = input_width;
    int dst_f32_pitch = sq_width;

    float *f32_r = int_f32_HD_P3R;
    float *f32_g = int_f32_HD_P3R + (input_width * input_height);
    float *f32_b = int_f32_HD_P3R + (2 * input_width * input_height);

    float *d_f32_r = dst_f32_SQ_P3R;
    float *d_f32_g = dst_f32_SQ_P3R + (dst_f32_pitch * dst_f32_pitch);
    float *d_f32_b = dst_f32_SQ_P3R + (2 * dst_f32_pitch * dst_f32_pitch);

    const Npp32f *resize_src[3] = {f32_r, f32_g, f32_b};
    Npp32f *dst[3] = {d_f32_r, d_f32_g, d_f32_b};
    NppiRect src_roi = {.x = 0, .y = 0, .width = input_width, .height = input_height};
    NppiRect dst_roi = {.x = 0, .y = 0, .width = sq_width, .height = sq_height};
    NppiSize src_size = {.width = input_width, .height = input_height};
    NppiSize dst_size = {.width = sq_width, .height = sq_height};
    int32_t src_step = src_size.width * sizeof(float);
    int32_t dst_step = dst_roi.width * sizeof(float);
    double scale_factor = std::min(dst_roi.width / (input_width * 1.0), dst_roi.height / (input_height * 1.0));
    int new_width = std::floor(scale_factor * input_width);
    int new_height = std::floor(scale_factor * input_height);
    float padding_width = (dst_roi.width - new_width) / 2;
    float padding_height = (dst_roi.height - new_height) / 2;

    half_constants constants{
        .recip = __float2half(0.003922f),
        .y_multiplicand = __float2half(1.164f),
        .y_addend = __float2half(-18.624f),
        .cr_rpx_multiplicand = __float2half(1.596f),
        .cr_rpx_addend = __float2half(-204.288f),
        .cr_gpx_multiplicand = __float2half(-0.813),
        .cr_gpx_addend = __float2half(104.064),
        .cb_gpx_multiplicand = __float2half(-0.392),
        .cb_gpx_addend = __float2half(50.176),
        .cb_bpx_multiplicand = __float2half(2.017),
        .cb_bpx_addend = __float2half(-258.176)};

    cuProfilerStart();

    // Calculate block and grid sizes
    dim3 block(TILE_WIDTH, TILE_HEIGHT);
    dim3 grid((input_width + (2 * TILE_WIDTH - 1)) / (2 * TILE_WIDTH), (input_height + TILE_HEIGHT - 1) / TILE_HEIGHT);

    switch (conv_type)
    {
    case BASIC_IMPL_NVIDIA_BT601:
        convertCbYCrToBGR24_f32_P3R_BASIC_IMPL_NVIDIA_BT601<<<grid, block, 0, context.hStream>>>(
            src, f32_r, f32_g, f32_b, input_width, input_height, src_pitch, int_f32_pitch);
        break;
    case BASIC_IMPL_NVIDIA_BT709:
        convertCbYCrToBGR24_f32_P3R_BASIC_IMPL_NVIDIA_BT709<<<grid, block, 0, context.hStream>>>(
            src, f32_r, f32_g, f32_b, input_width, input_height, src_pitch, int_f32_pitch);
        break;
    case BASIC_IMPL_INTEL_BT601:
        convertCbYCrToBGR24_f32_P3R_BASIC_IMPL_INTEL_BT601<<<grid, block, 0, context.hStream>>>(
            src, f32_r, f32_g, f32_b, input_width, input_height, src_pitch, int_f32_pitch);
        break;
    case BASIC_IMPL_INTEL_BT709:
        convertCbYCrToBGR24_f32_P3R_BASIC_IMPL_INTEL_BT709<<<grid, block, 0, context.hStream>>>(
            src, f32_r, f32_g, f32_b, input_width, input_height, src_pitch, int_f32_pitch);
        break;
    case FMA_IMPL_NVIDIA_BT601:
        convertCbYCrToBGR24_f32_P3R_FMA_IMPL_NVIDIA_BT601<<<grid, block, 0, context.hStream>>>(
            src, f32_r, f32_g, f32_b, input_width, input_height, src_pitch, int_f32_pitch);
        break;
    case FMA_IMPL_NVIDIA_BT709:
        convertCbYCrToBGR24_f32_P3R_FMA_IMPL_NVIDIA_BT709<<<grid, block, 0, context.hStream>>>(
            src, f32_r, f32_g, f32_b, input_width, input_height, src_pitch, int_f32_pitch);
        break;
    case FMA_IMPL_INTEL_BT601:
        convertCbYCrToBGR24_f32_P3R_FMA_IMPL_INTEL_BT601<<<grid, block, 0, context.hStream>>>(
            src, f32_r, f32_g, f32_b, input_width, input_height, src_pitch, int_f32_pitch);
        break;
    case FMA_IMPL_INTEL_BT601_FP16:
        convertCbYCrToBGR24_F16_P3R_FMA_IMPL_INTEL_BT601_FP16<<<grid, block, 0, context.hStream>>>(
            src, f16_r, f16_g, f16_b, input_width, input_height, src_pitch, int_f32_pitch, constants);
        break;
    case FMA_IMPL_INTEL_BT601_FP16_TO_FP32:
        convertCbYCrToBGR24_F32_P3R_FMA_IMPL_INTEL_BT601_FP16<<<grid, block, 0, context.hStream>>>(
            src, f32_r, f32_g, f32_b, input_width, input_height, src_pitch, int_f32_pitch, constants);
        break;
    case FMA_IMPL_INTEL_BT709:
        convertCbYCrToBGR24_f32_P3R_FMA_IMPL_INTEL_BT709<<<grid, block, 0, context.hStream>>>(
            src, f32_r, f32_g, f32_b, input_width, input_height, src_pitch, int_f32_pitch);
        break;
    }

    switch (conv_type)
    {
    case BASIC_IMPL_NVIDIA_BT601:
    case BASIC_IMPL_NVIDIA_BT709:
    case BASIC_IMPL_INTEL_BT601:
    case BASIC_IMPL_INTEL_BT709:
    case FMA_IMPL_NVIDIA_BT601:
    case FMA_IMPL_NVIDIA_BT709:
    case FMA_IMPL_INTEL_BT601:
    case FMA_IMPL_INTEL_BT709:
    case FMA_IMPL_INTEL_BT601_FP16_TO_FP32:
        nppiResizeSqrPixel_32f_P3R_Ctx(resize_src,
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

        cuProfilerStop();
        break;
    case FMA_IMPL_INTEL_BT601_FP16:
        NppiRect dstRect;

        nppiGetResizeRect(src_roi, &dstRect, scale_factor, scale_factor, padding_width, padding_height, NPPI_INTER_LINEAR);

        NppiImageDescriptor *imagesSrc,
            *dImagesSrc;
        NppiImageDescriptor *imagesDst, *dImagesDst;
        NppiResizeBatchROI_Advanced *roi, *dRoi;

        cudaMallocHost((void **)&imagesSrc, sizeof(NppiImageDescriptor) * 3);
        cudaMalloc((void **)&dImagesSrc, sizeof(NppiImageDescriptor) * 3);

        cudaMallocHost((void **)&imagesDst, sizeof(NppiImageDescriptor) * 3);
        cudaMalloc((void **)&dImagesDst, sizeof(NppiImageDescriptor) * 3);

        cudaMallocHost((void **)&roi, sizeof(NppiResizeBatchROI_Advanced) * 3);
        cudaMalloc((void **)&dRoi, sizeof(NppiResizeBatchROI_Advanced) * 3);

        imagesSrc[0].pData = (void *)f16_r;
        imagesSrc[0].nStep = input_width * sizeof(Npp16f);
        imagesSrc[0].oSize = src_size;
        imagesSrc[1].pData = (void *)f16_g;
        imagesSrc[1].nStep = input_width * sizeof(Npp16f);
        imagesSrc[1].oSize = src_size;
        imagesSrc[2].pData = (void *)f16_b;
        imagesSrc[2].nStep = input_width * sizeof(Npp16f);
        imagesSrc[2].oSize = src_size;

        imagesDst[0].pData = (void *)f16_dst_r;
        imagesDst[0].nStep = sq_width * sizeof(Npp16f);
        imagesDst[0].oSize = dst_size;
        imagesDst[1].pData = (void *)f16_dst_g;
        imagesDst[1].nStep = sq_width * sizeof(Npp16f);
        imagesDst[1].oSize = dst_size;
        imagesDst[2].pData = (void *)f16_dst_b;
        imagesDst[2].nStep = sq_width * sizeof(Npp16f);
        imagesDst[2].oSize = dst_size;

        roi[0].oSrcRectROI = src_roi;
        roi[0].oDstRectROI = dstRect;
        roi[1].oSrcRectROI = src_roi;
        roi[1].oDstRectROI = dstRect;
        roi[2].oSrcRectROI = src_roi;
        roi[2].oDstRectROI = dstRect;

        cudaMemcpy(dImagesSrc, imagesSrc, sizeof(NppiImageDescriptor) * 3, cudaMemcpyHostToDevice);
        cudaMemcpy(dImagesDst, imagesDst, sizeof(NppiImageDescriptor) * 3, cudaMemcpyHostToDevice);
        cudaMemcpy(dRoi, roi, sizeof(NppiResizeBatchROI_Advanced) * 3, cudaMemcpyHostToDevice);

        nppiResizeBatch_16f_C1R_Advanced_Ctx(sq_width, new_height, dImagesSrc, dImagesDst, dRoi, 3, NPPI_INTER_LINEAR, context);

        // nppiResize_16f_C1R_Ctx((Npp16f *)f16_r,
        //                        input_width * sizeof(Npp16f),
        //                        src_size,
        //                        src_roi,
        //                        (Npp16f *)f16_dst_r,
        //                        sq_width * sizeof(Npp16f),
        //                        dst_size,
        //                        dst_roi,
        //                        NPPI_INTER_LINEAR,
        //                        context);
        // nppiResize_16f_C1R_Ctx((Npp16f *)f16_g,
        //                        input_width * sizeof(Npp16f),
        //                        src_size,
        //                        src_roi,
        //                        (Npp16f *)f16_dst_g,
        //                        sq_width * sizeof(Npp16f),
        //                        dst_size,
        //                        dst_roi,
        //                        NPPI_INTER_LINEAR,
        //                        context);
        // nppiResize_16f_C1R_Ctx((Npp16f *)f16_b,
        //                        input_width * sizeof(Npp16f),
        //                        src_size,
        //                        src_roi,
        //                        (Npp16f *)f16_dst_b,
        //                        sq_width * sizeof(Npp16f),
        //                        dst_size,
        //                        dst_roi,
        //                        NPPI_INTER_LINEAR,
        //                        context);

        cuProfilerStop();

        nppiConvert_16f32f_C1R_Ctx((Npp16f *)f16_dst_r,
                                   sq_width * sizeof(Npp16f),
                                   dst[0],
                                   sq_width * sizeof(Npp32f),
                                   dst_size,
                                   context);
        nppiConvert_16f32f_C1R_Ctx((Npp16f *)f16_dst_g,
                                   sq_width * sizeof(Npp16f),
                                   dst[1],
                                   sq_width * sizeof(Npp32f),
                                   dst_size,
                                   context);
        nppiConvert_16f32f_C1R_Ctx((Npp16f *)f16_dst_b,
                                   sq_width * sizeof(Npp16f),
                                   dst[2],
                                   sq_width * sizeof(Npp32f),
                                   dst_size,
                                   context);
        break;
    }

    if (contiguos_block != nullptr)
        cudaFree(contiguos_block);

    if (contiguos_block_p2 != nullptr)
        cudaFree(contiguos_block_p2);
}
