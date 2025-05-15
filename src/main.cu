#include "cuda_mem_ctrl.h"
#include "kernels.h"

#include <opencv2/opencv.hpp>

int main()
{
    cuInit(0);

    cv::Mat image = cv::imread("test_sets/images_1920x1080/image_72.jpg");

    uint64_t src_size = image.rows * image.cols * image.elemSize();
    uint64_t int_size = image.rows * image.cols * 2;
    uint64_t dst_size = image.rows * image.cols * sizeof(float);

    auto dSrc = cuda_mem_ctrl::cuda_dev_malloc<uint8_t>(src_size);
    auto dInt = cuda_mem_ctrl::cuda_dev_malloc<uint8_t>(int_size);

    auto dDst_f32_r = cuda_mem_ctrl::cuda_dev_malloc<float>(dst_size);
    auto dDst_f32_g = cuda_mem_ctrl::cuda_dev_malloc<float>(dst_size);
    auto dDst_f32_b = cuda_mem_ctrl::cuda_dev_malloc<float>(dst_size);

    auto hDst_f32_r = cuda_mem_ctrl::cuda_host_malloc<float>(dst_size);
    auto hDst_f32_g = cuda_mem_ctrl::cuda_host_malloc<float>(dst_size);
    auto hDst_f32_b = cuda_mem_ctrl::cuda_host_malloc<float>(dst_size);

    cuda_mem_ctrl::cuda_cpy(image.data, dSrc.get(), src_size, cudaMemcpyHostToDevice);
    auto context = npp_stream_ctx::create_npp_stream_ctx();

    nppiRGB24ToCbYCr422(dSrc.get(), dInt.get(), image.cols, image.rows, context);
    convert_CbYCr_To_BGR24(dInt.get(), dDst_f32_r.get(), dDst_f32_g.get(), dDst_f32_b.get(), image.rows, image.cols);

    cuda_mem_ctrl::cuda_cpy(dDst_f32_r.get(), hDst_f32_r.get(), dst_size, cudaMemcpyDeviceToHost);
    cuda_mem_ctrl::cuda_cpy(dDst_f32_g.get(), hDst_f32_g.get(), dst_size, cudaMemcpyDeviceToHost);
    cuda_mem_ctrl::cuda_cpy(dDst_f32_b.get(), hDst_f32_b.get(), dst_size, cudaMemcpyDeviceToHost);

    cv::Mat image_red(image.rows, image.cols, CV_32FC1, hDst_f32_r.get());

    cv::imshow("Image", image);
    cv::imshow("F32 Red Channel", image_red);

    for (int i = 0; i < 5; i++)
        printf("F32 RGB: %f %f %f\n", hDst_f32_r.get()[i], hDst_f32_g.get()[i], hDst_f32_b.get()[i]);

    cv::waitKey(0);
    cv::destroyAllWindows();
}