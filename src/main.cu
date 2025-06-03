#include "cuda_mem_ctrl.hpp"
#include "kernels.h"

#include <opencv2/opencv.hpp>

int main()
{
    cuInit(0);

    cv::Mat image = cv::imread("test_sets/images_1920x1080/image_72.jpg");
    cudaStream_t stream;
    thrust::host_vector<uint8_t> hOriginal(dst_rgb_size);
    thrust::device_vector<uint8_t> dOriginal(dst_rgb_size);
    thrust::device_vector<uint8_t> dCbYCr(src_size);
    thrust::host_vector<uint8_t> hFinBGR(dst_rgb_size);
    thrust::device_vector<uint8_t> dFinBGR(dst_rgb_size);
    thrust::device_vector<float> int_f32_HD_P3R(HD_WIDTH * HD_HEIGHT);
    thrust::device_vector<float> dst_f32_SQ_P3R(SQ_WIDTH * SQ_HEIGHT);

    std::copy(image.data, image.data + (HD_WIDTH * HD_HEIGHT * 3), hOriginal.begin());
    dOriginal = hOriginal;

    auto context = npp_stream_ctx::create_npp_stream_ctx();
    cudaStreamCreate(&stream);

    nppiRGB24ToCbYCr422(thrust::raw_pointer_cast(dOriginal.data()), thrust::raw_pointer_cast(dCbYCr.data()), HD_WIDTH,
                        HD_HEIGHT, context);
    convert_CbYCrTOBGR24_u8C3R_f32_P3R(dCbYCr, hFinBGR, dFinBGR, int_f32_HD_P3R, dst_f32_SQ_P3R, context, stream);

    cv::Mat image_hd(HD_HEIGHT, HD_WIDTH, CV_8UC3, hFinBGR.data());

    cv::imshow("Image", image_hd);

    cv::waitKey(0);
    cv::destroyAllWindows();
}