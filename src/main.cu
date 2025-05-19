#include "cuda_mem_ctrl.hpp"
#include "kernels.h"

#include <opencv2/opencv.hpp>

int main()
{
    cuInit(0);

    cv::Mat image = cv::imread("test_sets/images_1920x1080/image_72.jpg");
    uint32_t hd_width = image.cols;
    uint32_t hd_height = image.rows;
    int square_width = 1984;
    int square_height = 1984;

    uint64_t src_size = hd_height * hd_width * image_sizes::RGB24_bytes_per_pixel;
    uint64_t int_size = hd_height * hd_width * 2;
    uint64_t hd_dst_size = hd_height * hd_width * sizeof(float);
    uint64_t f_dst_size = 1984 * 1984 * sizeof(float);

    auto host_rgb_source_image = cuda_mem_ctrl::cuda_host_malloc<uint8_t>(src_size);
    auto device_rgb_source_image = cuda_mem_ctrl::cuda_dev_malloc<uint8_t>(src_size);
    auto device_cbycr_image = cuda_mem_ctrl::cuda_dev_malloc<uint8_t>(int_size);

    auto device_rgb_f32_red_HD = cuda_mem_ctrl::cuda_dev_malloc<float>(hd_dst_size);
    auto device_rgb_f32_grn_HD = cuda_mem_ctrl::cuda_dev_malloc<float>(hd_dst_size);
    auto device_rgb_f32_blu_HD = cuda_mem_ctrl::cuda_dev_malloc<float>(hd_dst_size);
    auto host_rgb_f32_red_HD = cuda_mem_ctrl::cuda_host_malloc<float>(hd_dst_size);
    auto host_rgb_f32_grn_HD = cuda_mem_ctrl::cuda_host_malloc<float>(hd_dst_size);
    auto host_rgb_f32_blu_HD = cuda_mem_ctrl::cuda_host_malloc<float>(hd_dst_size);

    auto device_rgb_f32_red_SQ = cuda_mem_ctrl::cuda_dev_malloc<float>(f_dst_size);
    auto device_rgb_f32_grn_SQ = cuda_mem_ctrl::cuda_dev_malloc<float>(f_dst_size);
    auto device_rgb_f32_blu_SQ = cuda_mem_ctrl::cuda_dev_malloc<float>(f_dst_size);
    auto host_rgb_f32_red_SQ = cuda_mem_ctrl::cuda_host_malloc<float>(f_dst_size);
    auto host_rgb_f32_grn_SQ = cuda_mem_ctrl::cuda_host_malloc<float>(f_dst_size);
    auto host_rgb_f32_blu_SQ = cuda_mem_ctrl::cuda_host_malloc<float>(f_dst_size);

    std::copy(image.data, image.data + (hd_width * hd_height * 3), host_rgb_source_image.get());
    cuda_mem_ctrl::cuda_cpy(host_rgb_source_image, device_rgb_source_image, src_size, cudaMemcpyHostToDevice);
    auto context = npp_stream_ctx::create_npp_stream_ctx();

    nppiRGB24ToCbYCr422(device_rgb_source_image.get(), device_cbycr_image.get(), hd_width, hd_height, context);
    convert_CbYCr_To_BGR24(device_cbycr_image.get(), device_rgb_f32_red_HD.get(), device_rgb_f32_grn_HD.get(),
                           device_rgb_f32_blu_HD.get(), hd_height, hd_width);
    resize_BGR24_HD_to_1984x1984(device_rgb_f32_red_HD, device_rgb_f32_grn_HD, device_rgb_f32_blu_HD,
                                 device_rgb_f32_red_SQ, device_rgb_f32_grn_SQ, device_rgb_f32_blu_SQ, hd_width,
                                 hd_height, context);

    cuda_mem_ctrl::cuda_cpy(device_rgb_f32_red_HD, host_rgb_f32_red_HD, f_dst_size, cudaMemcpyDeviceToHost);
    cuda_mem_ctrl::cuda_cpy(device_rgb_f32_blu_HD, host_rgb_f32_blu_HD, f_dst_size, cudaMemcpyDeviceToHost);
    cuda_mem_ctrl::cuda_cpy(device_rgb_f32_grn_HD, host_rgb_f32_grn_HD, f_dst_size, cudaMemcpyDeviceToHost);

    cuda_mem_ctrl::cuda_cpy(device_rgb_f32_red_SQ, host_rgb_f32_red_SQ, f_dst_size, cudaMemcpyDeviceToHost);
    cuda_mem_ctrl::cuda_cpy(device_rgb_f32_blu_SQ, host_rgb_f32_blu_SQ, f_dst_size, cudaMemcpyDeviceToHost);
    cuda_mem_ctrl::cuda_cpy(device_rgb_f32_grn_SQ, host_rgb_f32_grn_SQ, f_dst_size, cudaMemcpyDeviceToHost);

    cv::Mat image_red_hd(hd_height, hd_width, CV_32FC1, host_rgb_f32_red_HD.get());
    cv::Mat image_grn_hd(hd_height, hd_width, CV_32FC1, host_rgb_f32_blu_HD.get());
    cv::Mat image_blu_hd(hd_height, hd_width, CV_32FC1, host_rgb_f32_grn_HD.get());

    cv::Mat image_red(square_height, square_width, CV_32FC1, host_rgb_f32_red_SQ.get());
    cv::Mat image_grn(square_height, square_width, CV_32FC1, host_rgb_f32_blu_SQ.get());
    cv::Mat image_blu(square_height, square_width, CV_32FC1, host_rgb_f32_grn_SQ.get());

    cv::imshow("Image", image);
    cv::imshow("F32 Red Channel", image_red);
    cv::imshow("F32 Blue Channel", image_blu);
    cv::imshow("F32 Green Channel", image_grn);

    cv::imshow("HD F32 Red Channel", image_red_hd);
    cv::imshow("HD F32 Blue Channel", image_blu_hd);
    cv::imshow("HD F32 Green Channel", image_grn_hd);

    cv::imwrite("test_sets/results/f32_red.png", image_red);
    cv::imwrite("test_sets/results/f32_grn.png", image_grn);
    cv::imwrite("test_sets/results/f32_blu.png", image_blu);

    cv::waitKey(0);
    cv::destroyAllWindows();
}