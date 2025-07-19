#include "img_conv.hpp"

#include <ranges>

int main()
{
    int width = 1920, height = 1080;
    std::uint8_t *cbycr_image = nullptr;
    std::uint8_t *rgb_image = nullptr;
    uint64_t run = 0;

    img_conv image_class_hd(width, height, 1984, 1984);
    cbycr_image = (std::uint8_t *)ippMalloc_L(width * height * 2);
    rgb_image = (std::uint8_t *)ippMalloc_L(width * height * 3);

    for (int i = 0; i < 100; i++)
    {
        std::stringstream ss;
        ss << "test_sets/images_1920x1080/image_" << i << ".jpg";
        cv::Mat rgb_image_mat;
        cv::Mat image = cv::imread(ss.str());
        cv::cvtColor(image, rgb_image_mat, cv::COLOR_BGR2RGB);
        memcpy(rgb_image, rgb_image_mat.ptr<void>(), width * height * 3);
        ippiRGBToCbYCr422_8u_C3C2R(rgb_image, width * 3, cbycr_image, width * 2, {width, height});
        image_class_hd.upload_data(cbycr_image);

        for (auto &kernel : enum_str_map)
        {
            printf("KERN RUN: %lu\n", run++);
            image_class_hd.convert_CbYCrToBGR(kernel.first);
        }
    }

    ippFree(cbycr_image);
    ippFree(rgb_image);

    // cbycr_image = nullptr;
    // rgb_image = nullptr;

    // width = 3840;
    // height = 2160;
    // img_conv image_class_4k(width, height, 1984, 1984);
    // cbycr_image = (std::uint8_t *)ippMalloc_L(width * height * 2);
    // rgb_image = (std::uint8_t *)ippMalloc_L(width * height * 3);
    // run = 0;

    // for (int i = 0; i < 100; i++)
    // {
    //     std::stringstream ss;
    //     ss << "test_sets/images_3840x2160/image_" << i << ".jpg";
    //     cv::Mat rgb_image_mat;
    //     cv::Mat image = cv::imread(ss.str());
    //     cv::cvtColor(image, rgb_image_mat, cv::COLOR_BGR2RGB);
    //     memcpy(rgb_image, rgb_image_mat.ptr<void>(), width * height * 3);
    //     ippiRGBToCbYCr422_8u_C3C2R(rgb_image, width * 3, cbycr_image, width * 2, {width, height});
    //     image_class_4k.upload_data(cbycr_image);

    //     for (auto &kernel : enum_str_map)
    //     {
    //         printf("KERN RUN: %lu\n", run++);
    //         image_class_4k.convert_CbYCrToBGR(kernel.first);
    //     }
    // }

    // ippFree(cbycr_image);
    // ippFree(rgb_image);
}