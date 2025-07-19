// header-begin ------------------------------------------
// File       : img_conv.cu
//
// Author      : Joshua E
// Email       : estesjn2020@gmail.com
//
// Created on  : 7/18/2025
//
// header-end --------------------------------------------

#include "img_conv.hpp"

#define TILE_WIDTH 32
#define TILE_HEIGHT 8

img_conv::img_conv(int input_width, int input_height, int output_width, int output_height)
{
    sizes.input_width = input_width;
    sizes.input_height = input_height;
    sizes.output_width = output_width;
    sizes.output_height = output_height;

    context = create_npp_stream_ctx();
    cublasCreate(&cublas_handle);
    cublasSetStream(cublas_handle, context.hStream);

    src_size = {.width = sizes.input_width, .height = sizes.input_height};
    dst_size = {.width = sizes.output_width, .height = sizes.output_height};
    src_roi = {.x = 0, .y = 0, .width = sizes.input_width, .height = sizes.input_height};
    dst_roi = {.x = 0, .y = 0, .width = sizes.output_width, .height = sizes.output_height};

    src_pitch = sizes.input_width * ImageDefinitions::UYVY_BPP;
    pitch_f32 = sizes.input_width;
    pitch_f16 = sizes.input_width;

    scale_factor =
        std::min(sizes.output_width / (1.f * sizes.input_width), sizes.output_height / (1.f * sizes.input_height));
    new_width = std::floor(scale_factor * sizes.input_width);
    new_height = std::floor(scale_factor * sizes.input_height);
    padding_width = (dst_roi.width - new_width) / 2;
    padding_height = (dst_roi.height - new_height) / 2;

    nppiGetResizeRect(src_roi, &dst_batched_roi, scale_factor, scale_factor, padding_width, padding_height,
                      NPPI_INTER_LINEAR);

    this->constants.recip = __float2half(0.003922f);
    this->constants.y_multiplicand = __float2half(1.164f);
    this->constants.y_addend = __float2half(-18.624f);
    this->constants.cr_rpx_multiplicand = __float2half(1.596f);
    this->constants.cr_rpx_addend = __float2half(-204.288f);
    this->constants.cr_gpx_multiplicand = __float2half(-0.813);
    this->constants.cr_gpx_addend = __float2half(104.064);
    this->constants.cb_gpx_multiplicand = __float2half(-0.392);
    this->constants.cb_gpx_addend = __float2half(50.176);
    this->constants.cb_bpx_multiplicand = __float2half(2.017);
    this->constants.cb_bpx_addend = __float2half(-258.176);

    cudaMalloc(reinterpret_cast<void **>(&src_image),
               sizes.input_width * sizes.input_height * ImageDefinitions::RGB_BPP);
    cudaMalloc(reinterpret_cast<void **>(&int_f32),
               sizeof(float) * sizes.input_width * sizes.input_height * NUM_OF_CHANNELS);
    cudaMalloc(reinterpret_cast<void **>(&int_f16),
               sizeof(__half) * sizes.input_width * sizes.input_height * NUM_OF_CHANNELS);
    cudaMalloc(reinterpret_cast<void **>(&dst_f16),
               sizeof(__half) * sizes.output_width * sizes.output_height * NUM_OF_CHANNELS);
    cudaMalloc(reinterpret_cast<void **>(&dst_f32),
               sizeof(float) * sizes.output_width * sizes.output_height * NUM_OF_CHANNELS);
    cudaMalloc(reinterpret_cast<void **>(&reference),
               sizeof(float) * sizes.output_width * sizes.output_height * NUM_OF_CHANNELS);

    int_channels_f16[0] = (Npp16f *)int_f16;
    int_channels_f16[1] = (Npp16f *)int_f16 + (sizes.input_width * sizes.input_height);
    int_channels_f16[2] = (Npp16f *)int_f16 + (2 * sizes.input_width * sizes.input_height);

    int_channels_f32[0] = int_f32;
    int_channels_f32[1] = int_f32 + (sizes.input_width * sizes.input_height);
    int_channels_f32[2] = int_f32 + (2 * sizes.input_width * sizes.input_height);

    dst_channels_f16[0] = (Npp16f *)dst_f16;
    dst_channels_f16[1] = (Npp16f *)dst_f16 + (sizes.output_width * sizes.output_height);
    dst_channels_f16[2] = (Npp16f *)dst_f16 + (2 * sizes.output_width * sizes.output_height);

    dst_channels_f32[0] = dst_f32;
    dst_channels_f32[1] = dst_f32 + (sizes.output_width * sizes.output_height);
    dst_channels_f32[2] = dst_f32 + (2 * sizes.output_width * sizes.output_height);

    cudaMallocHost(reinterpret_cast<void **>(&imagesSrc), sizeof(NppiImageDescriptor) * NUM_OF_CHANNELS);
    cudaMallocHost(reinterpret_cast<void **>(&imagesDst), sizeof(NppiImageDescriptor) * NUM_OF_CHANNELS);
    cudaMallocHost(reinterpret_cast<void **>(&roi), sizeof(NppiResizeBatchROI_Advanced) * NUM_OF_CHANNELS);
    cudaMalloc(reinterpret_cast<void **>(&dImagesSrc), sizeof(NppiImageDescriptor) * NUM_OF_CHANNELS);
    cudaMalloc(reinterpret_cast<void **>(&dImagesDst), sizeof(NppiImageDescriptor) * NUM_OF_CHANNELS);
    cudaMalloc(reinterpret_cast<void **>(&dRoi), sizeof(NppiResizeBatchROI_Advanced) * NUM_OF_CHANNELS);

    imagesSrc[0].pData = (void *)int_channels_f16[0];
    imagesSrc[0].nStep = sizes.input_width * sizeof(Npp16f);
    imagesSrc[0].oSize = src_size;
    imagesSrc[1].pData = (void *)int_channels_f16[1];
    imagesSrc[1].nStep = sizes.input_width * sizeof(Npp16f);
    imagesSrc[1].oSize = src_size;
    imagesSrc[2].pData = (void *)int_channels_f16[2];
    imagesSrc[2].nStep = sizes.input_width * sizeof(Npp16f);
    imagesSrc[2].oSize = src_size;

    imagesDst[0].pData = (void *)dst_channels_f16[0];
    imagesDst[0].nStep = sizes.output_width * sizeof(Npp16f);
    imagesDst[0].oSize = dst_size;
    imagesDst[1].pData = (void *)dst_channels_f16[1];
    imagesDst[1].nStep = sizes.output_width * sizeof(Npp16f);
    imagesDst[1].oSize = dst_size;
    imagesDst[2].pData = (void *)dst_channels_f16[2];
    imagesDst[2].nStep = sizes.output_width * sizeof(Npp16f);
    imagesDst[2].oSize = dst_size;

    roi[0].oSrcRectROI = src_roi;
    roi[0].oDstRectROI = dst_batched_roi;
    roi[1].oSrcRectROI = src_roi;
    roi[1].oDstRectROI = dst_batched_roi;
    roi[2].oSrcRectROI = src_roi;
    roi[2].oDstRectROI = dst_batched_roi;

    cudaMemcpy(dImagesSrc, imagesSrc, sizeof(NppiImageDescriptor) * NUM_OF_CHANNELS, cudaMemcpyHostToDevice);
    cudaMemcpy(dImagesDst, imagesDst, sizeof(NppiImageDescriptor) * NUM_OF_CHANNELS, cudaMemcpyHostToDevice);
    cudaMemcpy(dRoi, roi, sizeof(NppiResizeBatchROI_Advanced) * NUM_OF_CHANNELS, cudaMemcpyHostToDevice);
}

img_conv::~img_conv()
{
    cudaFree(src_image);
    cudaFree(int_f32);
    cudaFree(int_f16);
    cudaFree(dst_f16);
    cudaFree(dst_f32);
    cudaFree(reference);
    cudaFree(imagesSrc);
    cudaFree(imagesDst);
    cudaFree(roi);
    cudaFree(dImagesSrc);
    cudaFree(dImagesDst);
    cudaFree(dRoi);
}

NppStreamContext img_conv::create_npp_stream_ctx()
{
    int device;
    cudaGetDevice(&device);

    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties, 0);

    int major = 0;
    int minor = 0;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0);

    uint32_t flags = 0;
    cudaStreamGetFlags(stream, &flags);

    // Set up NPP Stream Context
    NppStreamContext context = {.hStream = stream,
                                .nCudaDeviceId = device,
                                .nMultiProcessorCount = properties.multiProcessorCount,
                                .nMaxThreadsPerMultiProcessor = properties.maxThreadsPerMultiProcessor,
                                .nMaxThreadsPerBlock = properties.maxThreadsPerBlock,
                                .nSharedMemPerBlock = properties.sharedMemPerBlock,
                                .nCudaDevAttrComputeCapabilityMajor = major,
                                .nCudaDevAttrComputeCapabilityMinor = minor,
                                .nStreamFlags = flags};

    return context;
}

__global__ void convertCbYCrToBGR24_F32_P3R_BASIC_IMPL_NVIDIA_BT601(uint8_t *__restrict__ src,
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

__global__ void convertCbYCrToBGR24_F32_P3R_BASIC_IMPL_NVIDIA_BT709(uint8_t *__restrict__ src,
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

__global__ void convertCbYCrToBGR24_F32_P3R_BASIC_IMPL_INTEL_BT601(uint8_t *__restrict__ src,
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

__global__ void convertCbYCrToBGR24_F32_P3R_BASIC_IMPL_INTEL_BT709(uint8_t *__restrict__ src,
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

__global__ void convertCbYCrToBGR24_F32_P3R_FMA_IMPL_NVIDIA_BT601(uint8_t *__restrict__ src,
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

__global__ void convertCbYCrToBGR24_F32_P3R_FMA_IMPL_NVIDIA_BT709(uint8_t *__restrict__ src,
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

__global__ void convertCbYCrToBGR24_F32_P3R_FMA_IMPL_INTEL_BT601(uint8_t *__restrict__ src,
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
                                                                      const int dst_f16_pitch,
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
    int64_t dst_f16_idx = y * dst_f16_pitch + x;
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
    *(uint32_t *)&dst_r[dst_f16_idx] = *(uint32_t *)&pixel;

    pixel.x = __hmul_sat(green.x, constants.recip);
    pixel.y = __hmul_sat(green.y, constants.recip);
    *(uint32_t *)&dst_g[dst_f16_idx] = *(uint32_t *)&pixel;

    pixel.x = __hmul_sat(blue.x, constants.recip);
    pixel.y = __hmul_sat(blue.y, constants.recip);
    *(uint32_t *)&dst_g[dst_f16_idx] = *(uint32_t *)&pixel;
}

#pragma nv_diag_suppress static_var_with_dynamic_init

__global__ void convertCbYCrToBGR24_F16_P3R_FMA_IMPL_INTEL_BT601_FP16_exp(uint8_t *__restrict__ src,
                                                                          half *__restrict__ dst_r,
                                                                          half *__restrict__ dst_g,
                                                                          half *__restrict__ dst_b,
                                                                          const int width,
                                                                          const int height,
                                                                          const int src_pitch,
                                                                          const int dst_f16_pitch,
                                                                          const half_constants constants)
{
    // 1024 Bytes
    __shared__ uint32_t input_image_data[TILE_WIDTH * TILE_HEIGHT];
    __shared__ cuda::barrier<cuda::thread_scope_block> barrier;
    conversion_union conv;

    if (threadIdx.x == 0 && threadIdx.y == 0)
        init(&barrier, blockDim.x * blockDim.y);

    __syncthreads();

    if (threadIdx.x == 0 && threadIdx.y == 0)
    {
        int64_t src_u8_idx = (blockIdx.y * TILE_HEIGHT) * src_pitch + 128 * (blockIdx.x * TILE_WIDTH);
        cuda::memcpy_async(&input_image_data[0], src + src_u8_idx, 128 * sizeof(uint32_t) * TILE_WIDTH * TILE_HEIGHT,
                           barrier);
    }

    int x = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    half cb, y0, cr, y1;
    half scaled_y0, scaled_y1;
    half cr_rpx, cr_gpx, cb_gpx, cb_bpx, cbcr_gpx;
    half2 red, green, blue;
    half2 pixel;

    int64_t dst_f16_idx = y * dst_f16_pitch + x;
    half *red_ptr = dst_r + dst_f16_idx;
    half *grn_ptr = dst_g + dst_f16_idx;
    half *blu_ptr = dst_b + dst_f16_idx;

    barrier.arrive_and_wait();
    conv.double_word = input_image_data[threadIdx.y * TILE_HEIGHT + threadIdx.x];
    cb = __float2half(conv.cb);
    y0 = __float2half(conv.y0);
    cr = __float2half(conv.cr);
    y1 = __float2half(conv.y1);

    scaled_y0 = __hfma(constants.y_multiplicand, y0, constants.y_addend);
    scaled_y1 = __hfma(constants.y_multiplicand, y1, constants.y_addend);
    cr_rpx = __hfma(constants.cr_rpx_multiplicand, cr, constants.cr_rpx_addend);
    cr_gpx = __hfma(constants.cr_gpx_multiplicand, cr, constants.cr_gpx_addend);
    cb_gpx = __hfma(constants.cb_gpx_multiplicand, cb, constants.cb_gpx_addend);
    red.x = __hadd(scaled_y0, cr_rpx);
    red.y = __hadd(scaled_y1, cr_rpx);
    cb_bpx = __hfma(constants.cb_bpx_multiplicand, cb, constants.cb_bpx_addend);
    cbcr_gpx = __hadd(cr_gpx, cb_gpx);
    green.x = __hadd(scaled_y0, cbcr_gpx);
    green.y = __hadd(scaled_y1, cbcr_gpx);
    blue.x = __hadd(scaled_y0, cb_bpx);
    blue.y = __hadd(scaled_y1, cb_bpx);

    pixel.x = __hmul_sat(red.x, constants.recip);
    pixel.y = __hmul_sat(red.y, constants.recip);
    *(uint32_t *)red_ptr = *(uint32_t *)&pixel;

    pixel.x = __hmul_sat(green.x, constants.recip);
    pixel.y = __hmul_sat(green.y, constants.recip);
    *(uint32_t *)grn_ptr = *(uint32_t *)&pixel;

    pixel.x = __hmul_sat(blue.x, constants.recip);
    pixel.y = __hmul_sat(blue.y, constants.recip);
    *(uint32_t *)blu_ptr = *(uint32_t *)&pixel;
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

__global__ void convertCbYCrToBGR24_F32_P3R_FMA_IMPL_INTEL_BT709(uint8_t *__restrict__ src,
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

__global__ void convertCbYCrToBGR24_F32_P3R_APPROX(uint8_t *__restrict__ src,
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

void img_conv::convert_CbYCrToBGR(uint8_t conv_type)
{
    dim3 block(TILE_WIDTH, TILE_HEIGHT);
    dim3 grid((sizes.input_width + (2 * TILE_WIDTH - 1)) / (2 * TILE_WIDTH),
              (sizes.input_height + TILE_HEIGHT - 1) / TILE_HEIGHT);

    cuProfilerStart();

    switch (conv_type)
    {
    case BASIC_IMPL_NVIDIA_BT601:
        convertCbYCrToBGR24_F32_P3R_BASIC_IMPL_NVIDIA_BT601<<<grid, block, 0, context.hStream>>>(
            src_image, int_channels_f32[0], int_channels_f32[1], int_channels_f32[2], sizes.input_width,
            sizes.input_height, src_pitch, pitch_f32);
        break;
    case BASIC_IMPL_NVIDIA_BT709:
        convertCbYCrToBGR24_F32_P3R_BASIC_IMPL_NVIDIA_BT709<<<grid, block, 0, context.hStream>>>(
            src_image, int_channels_f32[0], int_channels_f32[1], int_channels_f32[2], sizes.input_width,
            sizes.input_height, src_pitch, pitch_f32);
        break;
    case BASIC_IMPL_INTEL_BT601:
        convertCbYCrToBGR24_F32_P3R_BASIC_IMPL_INTEL_BT601<<<grid, block, 0, context.hStream>>>(
            src_image, int_channels_f32[0], int_channels_f32[1], int_channels_f32[2], sizes.input_width,
            sizes.input_height, src_pitch, pitch_f32);
        break;
    case BASIC_IMPL_INTEL_BT709:
        convertCbYCrToBGR24_F32_P3R_BASIC_IMPL_INTEL_BT709<<<grid, block, 0, context.hStream>>>(
            src_image, int_channels_f32[0], int_channels_f32[1], int_channels_f32[2], sizes.input_width,
            sizes.input_height, src_pitch, pitch_f32);
        break;
    case FMA_IMPL_NVIDIA_BT601:
        convertCbYCrToBGR24_F32_P3R_FMA_IMPL_NVIDIA_BT601<<<grid, block, 0, context.hStream>>>(
            src_image, int_channels_f32[0], int_channels_f32[1], int_channels_f32[2], sizes.input_width,
            sizes.input_height, src_pitch, pitch_f32);
        break;
    case FMA_IMPL_NVIDIA_BT709:
        convertCbYCrToBGR24_F32_P3R_FMA_IMPL_NVIDIA_BT709<<<grid, block, 0, context.hStream>>>(
            src_image, int_channels_f32[0], int_channels_f32[1], int_channels_f32[2], sizes.input_width,
            sizes.input_height, src_pitch, pitch_f32);
        break;
    case FMA_IMPL_INTEL_BT601:
        convertCbYCrToBGR24_F32_P3R_FMA_IMPL_INTEL_BT601<<<grid, block, 0, context.hStream>>>(
            src_image, int_channels_f32[0], int_channels_f32[1], int_channels_f32[2], sizes.input_width,
            sizes.input_height, src_pitch, pitch_f32);
        break;
    case FMA_IMPL_INTEL_BT601_FP16:
        convertCbYCrToBGR24_F16_P3R_FMA_IMPL_INTEL_BT601_FP16_exp<<<grid, block, 0, context.hStream>>>(
            src_image, reinterpret_cast<half *>(int_channels_f16[0]), reinterpret_cast<half *>(int_channels_f16[1]),
            reinterpret_cast<half *>(int_channels_f16[2]), sizes.input_width, sizes.input_height, src_pitch, pitch_f16,
            constants);
        break;
    case FMA_IMPL_INTEL_BT601_FP16_TO_FP32:
        convertCbYCrToBGR24_F32_P3R_FMA_IMPL_INTEL_BT601_FP16<<<grid, block, 0, context.hStream>>>(
            src_image, int_channels_f32[0], int_channels_f32[1], int_channels_f32[2], sizes.input_width,
            sizes.input_height, src_pitch, pitch_f32, constants);
        break;
    case FMA_IMPL_INTEL_BT709:
        convertCbYCrToBGR24_F32_P3R_FMA_IMPL_INTEL_BT709<<<grid, block, 0, context.hStream>>>(
            src_image, int_channels_f32[0], int_channels_f32[1], int_channels_f32[2], sizes.input_width,
            sizes.input_height, src_pitch, pitch_f32);
        break;
    case APPROX_IMPL:
        convertCbYCrToBGR24_F32_P3R_APPROX<<<grid, block, 0, context.hStream>>>(
            src_image, int_channels_f32[0], int_channels_f32[1], int_channels_f32[2], sizes.input_width,
            sizes.input_height, src_pitch, pitch_f32);
        break;
    }

    cuProfilerStop();

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
    case APPROX_IMPL:
        nppiResizeSqrPixel_32f_P3R_Ctx(int_channels_f32, src_size, sizes.input_width * sizeof(float), src_roi,
                                       dst_channels_f32, sizes.output_width * sizeof(float), dst_roi, scale_factor,
                                       scale_factor, padding_width, padding_height, NPPI_INTER_LINEAR, context);
        break;
    case FMA_IMPL_INTEL_BT601_FP16:
        nppiResizeBatch_16f_C1R_Advanced_Ctx(sizes.output_width, sizes.output_height, dImagesSrc, dImagesDst, dRoi, 3,
                                             NPPI_INTER_LINEAR, context);
        nppiConvert_16f32f_C1R_Ctx(dst_channels_f16[0], sizes.output_width * sizeof(Npp16f), dst_channels_f32[0],
                                   sizes.output_width * sizeof(Npp32f), dst_size, context);
        nppiConvert_16f32f_C1R_Ctx(dst_channels_f16[1], sizes.output_width * sizeof(Npp16f), dst_channels_f32[1],
                                   sizes.output_width * sizeof(Npp32f), dst_size, context);
        nppiConvert_16f32f_C1R_Ctx(dst_channels_f16[2], sizes.output_width * sizeof(Npp16f), dst_channels_f32[2],
                                   sizes.output_width * sizeof(Npp32f), dst_size, context);
        break;
    }
}

std::tuple<float, float, float> img_conv::compute_rel_err()
{
    int len = sizes.output_height * sizes.output_width * NUM_OF_CHANNELS;
    float reference_nrm = 0;
    float dut_nrm = 0;

    float alpha = -1.0f;

    cublasSnrm2(cublas_handle, len, reference, 1, &reference_nrm);
    cublasSaxpy(cublas_handle, len, &alpha, dst_f32, 1, reference, 1);
    cublasSnrm2(cublas_handle, len, reference, 1, &dut_nrm);

    return std::make_tuple(reference_nrm, dut_nrm, dut_nrm / reference_nrm);
}

int img_conv::upload_data(std::uint8_t *image)
{
    if (image == nullptr)
        return EXIT_FAILURE;

    return cudaMemcpy(src_image, image, sizes.input_height * sizes.input_width * ImageDefinitions::UYVY_BPP,
                      cudaMemcpyHostToDevice);
}

int img_conv::upload_reference(float *image)
{
    if (image == nullptr)
        return EXIT_FAILURE;

    return cudaMemcpy(reference, image, sizeof(float) * sizes.output_height * sizes.output_width * NUM_OF_CHANNELS,
                      cudaMemcpyHostToDevice);
}

int img_conv::copy_dst_to_reference()
{
    return cudaMemcpy(reference, dst_f32, sizeof(float) * sizes.output_height * sizes.output_width * NUM_OF_CHANNELS,
                      cudaMemcpyDeviceToDevice);
}

std::uint8_t *img_conv::get_u8_ptr()
{
    return src_image;
}

float *img_conv::get_f32_ptr()
{
    return dst_f32;
}

half *img_conv::get_f16_ptr()
{
    return dst_f16;
}

// footer-begin ------------------------------------------
// default.CUDA-C++
// File       : img_conv.cu
// footer-end --------------------------------------------