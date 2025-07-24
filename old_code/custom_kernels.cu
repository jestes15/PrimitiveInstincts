#include "kernels.h"

__global__ void convertCbYCrToBGR24_f32_P3R_approx(uint8_t *__restrict__ src,
                                                   float *__restrict__ dst_r,
                                                   float *__restrict__ dst_g,
                                                   float *__restrict__ dst_b,
                                                   int width,
                                                   int height,
                                                   int src_pitch,
                                                   int dst_f32_pitch)
{

    conversion_union conv;

    int x = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int cb, y0, cr, y1;
    float pair[2];
    float r0_, g0_, b0_, r1_, g1_, b1_;

    constexpr float recip = 0.003922;
    int64_t src_u8_idx = y * src_pitch + 2 * x;
    int64_t dst_f32_idx = y * dst_f32_pitch + x;
    uint8_t *addr = src + src_u8_idx;

    conv.double_word = *(uint32_t *)addr;
    cb = static_cast<int>(conv.cb) - 128;
    y0 = 298 * (static_cast<int>(conv.y0) - 16);
    cr = static_cast<int>(conv.cr) - 128;
    y1 = 298 * (static_cast<int>(conv.y1) - 16);

    r0_ = static_cast<float>((y0 + 204 * cr + 128) >> 7);
    r1_ = static_cast<float>((y1 + 204 * cr + 128) >> 7);
    g0_ = static_cast<float>((y0 - 50 * cb - 104 * cr + 128) >> 7);
    g1_ = static_cast<float>((y1 - 50 * cb - 104 * cr + 128) >> 7);
    b0_ = static_cast<float>((y0 + 258 * cb + 128) >> 7);
    b1_ = static_cast<float>((y1 + 258 * cb + 128) >> 7);

    pair[0] = __saturatef(__fmul_rz(r0_, recip));
    pair[1] = __saturatef(__fmul_rz(r1_, recip));
    *(uint64_t *)&dst_r[dst_f32_idx] = *(uint64_t *)&pair;

    pair[0] = __saturatef(__fmul_rz(g0_, recip));
    pair[1] = __saturatef(__fmul_rz(g1_, recip));
    *(uint64_t *)&dst_g[dst_f32_idx] = *(uint64_t *)&pair;

    pair[0] = __saturatef(__fmul_rz(b0_, recip));
    pair[1] = __saturatef(__fmul_rz(b1_, recip));
    *(uint64_t *)&dst_g[dst_f32_idx] = *(uint64_t *)&pair;
}

void convert_CbYCrToBGR24_f32_P3R_approx(uint8_t *__restrict__ src,
                                         float *__restrict__ int_f32_HD_P3R,
                                         int32_t input_width,
                                         int32_t input_height,
                                         float *__restrict__ dst_f32_SQ_P3R,
                                         int sq_width,
                                         int sq_height,
                                         NppStreamContext context)
{
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
    int32_t src_step = src_size.width * sizeof(float);
    int32_t dst_step = dst_roi.width * sizeof(float);
    double scale_factor = std::min(dst_roi.width / (input_width * 1.0), dst_roi.height / (input_height * 1.0));
    int new_width = std::ceil(scale_factor * input_width);
    int new_height = std::ceil(scale_factor * input_height);
    float padding_width = (dst_roi.width - new_width) / 2;
    float padding_height = (dst_roi.height - new_height) / 2;

    // Calculate block and grid sizes
    dim3 block(TILE_WIDTH, TILE_HEIGHT);
    dim3 grid((input_width + (2 * TILE_WIDTH - 1)) / (2 * TILE_WIDTH), (input_height + TILE_HEIGHT - 1) / TILE_HEIGHT);

    convertCbYCrToBGR24_f32_P3R_approx<<<grid, block, 0, context.hStream>>>(
        src, f32_r, f32_g, f32_b, input_width, input_height, src_pitch, int_f32_pitch);
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
}
