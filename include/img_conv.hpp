#include "image_sizes.h"

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <functional>
#include <ipp.h>
#include <memory>
#include <npp.h>

#include <opencv2/opencv.hpp>
#include <stdint.h>
#include <stdio.h>
#include <type_traits>

#define NUM_OF_CHANNELS 3

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
    FMA_IMPL_INTEL_BT709,
    APPROX_IMPL
};

const std::map<int, std::string> enum_str_map = {
    {BASIC_IMPL_NVIDIA_BT601, "BASIC_IMPL_NVIDIA_BT601"},
    {BASIC_IMPL_NVIDIA_BT709, "BASIC_IMPL_NVIDIA_BT709"},
    {BASIC_IMPL_INTEL_BT601, "BASIC_IMPL_INTEL_BT601"},
    {BASIC_IMPL_INTEL_BT709, "BASIC_IMPL_INTEL_BT709"},
    {FMA_IMPL_NVIDIA_BT601, "FMA_IMPL_NVIDIA_BT601"},
    {FMA_IMPL_NVIDIA_BT709, "FMA_IMPL_NVIDIA_BT709"},
    {FMA_IMPL_INTEL_BT601, "FMA_IMPL_INTEL_BT601"},
    {FMA_IMPL_INTEL_BT601_FP16, "FMA_IMPL_INTEL_BT601_FP16"},
    {FMA_IMPL_INTEL_BT601_FP16_TO_FP32, "FMA_IMPL_INTEL_BT601_FP16_TO_FP32"},
    {FMA_IMPL_INTEL_BT709, "FMA_IMPL_INTEL_BT709"},
    {APPROX_IMPL, "APPROX_IMPL"}};

struct sizes_t
{
    int input_width;
    int input_height;
    int output_width;
    int output_height;
};

union conversion_union {
    uint32_t double_word;
    struct
    {
        std::uint8_t cb;
        std::uint8_t y0;
        std::uint8_t cr;
        std::uint8_t y1;
    };
};

struct half_constants
{
    __half recip;
    __half y_multiplicand;
    __half y_addend;
    __half cr_rpx_multiplicand;
    __half cr_rpx_addend;
    __half cr_gpx_multiplicand;
    __half cr_gpx_addend;
    __half cb_gpx_multiplicand;
    __half cb_gpx_addend;
    __half cb_bpx_multiplicand;
    __half cb_bpx_addend;
};

class img_conv
{
  private:
    static const int TILE_WIDTH = 32;
    static const int TILE_HEIGHT = 4;

    NppStreamContext context;
    cublasHandle_t cublas_handle;
    sizes_t sizes;

    // Device Pointers
    std::uint8_t *src_image = nullptr;
    float *int_f32 = nullptr;
    __half *int_f16 = nullptr;
    __half *dst_f16 = nullptr;
    float *dst_f32 = nullptr;
    float *reference = nullptr;

    // Host Pointers
    std::uint8_t *cbycr_image = nullptr;
    cv::Mat final_image;

    int src_pitch, pitch_f32, pitch_f16;
    float scale_factor;
    float padding_width, padding_height;

    Npp16f *int_channels_f16[NUM_OF_CHANNELS], *dst_channels_f16[NUM_OF_CHANNELS];
    Npp32f *int_channels_f32[NUM_OF_CHANNELS], *dst_channels_f32[NUM_OF_CHANNELS];

    NppiSize src_size, dst_size;
    NppiRect src_roi, dst_roi, dst_batched_roi;

    NppiRect dstRect_16f;
    NppiImageDescriptor *imagesSrc, *dImagesSrc;
    NppiImageDescriptor *imagesDst, *dImagesDst;
    NppiResizeBatchROI_Advanced *roi, *dRoi;
    half_constants constants;
    NppStreamContext create_npp_stream_ctx();

    cv::Mat normalize_cv_mat(const cv::Mat &input, bool aNormalize);
    cv::Mat resize_with_aspect_ratio(const cv::Mat &input);

  public:
    img_conv(int input_width, int input_height, int output_width, int output_height);
    ~img_conv();
    
    void convert_CbYCrToBGR(std::uint8_t conv_type);
    void create_reference_image();
    void zero_data();
    std::tuple<float, float, float> compute_rel_err();
    int upload_data(std::uint8_t *image);
    int upload_reference(float *image);
    int upload_reference();
    void set_cbycr_image(std::uint8_t *cbycr_image);
    std::uint8_t *get_u8_ptr();
    float *get_f32_ptr();
    half *get_f16_ptr();
};
