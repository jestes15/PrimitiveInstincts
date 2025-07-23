#include "img_conv.hpp"

#include <filesystem>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>

void display_image(float *device_image, int width, int height)
{
    float *host_image = nullptr;
    cudaMallocHost((void **)&host_image, sizeof(float) * width * height);
    cudaMemcpy(host_image, device_image, sizeof(float) * width * height, cudaMemcpyDeviceToHost);

    cv::Mat image(height, width, CV_32FC1, host_image);
    cv::namedWindow("Window", cv::WINDOW_NORMAL);
    cv::resizeWindow("Window", 800, 800);
    cv::imshow("Window", image);
    cv::waitKey(0);
    cv::destroyAllWindows();

    cudaFree(host_image);
}

int main(int argc, char *argv[])
{
    if (argc != 4)
    {
        printf("Too few arguemnts...\n");
        printf("Usage:\n");
        printf("\timg_conv_test /path/to/images output_width output_height");
        exit(EXIT_FAILURE);
    }

    int output_width = std::atoi(argv[2]);
    int output_height = std::atoi(argv[3]);

    cv::utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_SILENT);
    std::set<std::string> image_extensions = {".jpg", ".jpeg", ".png"};
    std::vector<float> accum(enum_str_map.size());
    std::vector<std::string> image_filepaths;
    uint8_t *rgb_image_ptr = nullptr;
    uint8_t *cbycr_image_ptr = nullptr;

    bool uniform_sizes = true;
    int width = -1, height = -1;

    for (const auto &file : std::filesystem::recursive_directory_iterator(argv[1]))
    {
        if (!file.is_regular_file())
            continue;

        if (image_extensions.count(file.path().extension()) == 0)
            continue;

        image_filepaths.push_back(file.path());

        cv::Mat img = cv::imread(file.path());

        if (width == -1)
            width = img.cols;

        if (height == -1)
            height = img.rows;

        if (width != img.cols && height != img.rows)
            uniform_sizes = false;
    }

    if (!uniform_sizes)
    {
        std::cout << "All images must be uniform in size\n";
        exit(EXIT_FAILURE);
    }

    IppiSize ipp_roi = {
        .width = width,
        .height = height};

    rgb_image_ptr = reinterpret_cast<uint8_t *>(ippMalloc(width * height * ImageDefinitions::RGB_BPP));
    cbycr_image_ptr = reinterpret_cast<uint8_t *>(ippMalloc(width * height * ImageDefinitions::UYVY_BPP));
    img_conv testing_class(width, height, output_width, output_height);

    int progress = 0;

    for (const auto &file : image_filepaths)
    {
        std::cout << "[" << progress++ << "/" << image_filepaths.size() << "]\r";

        cv::Mat rgb_image;
        cv::Mat bgr_image = cv::imread(file);
        cv::cvtColor(bgr_image, rgb_image, cv::COLOR_BGR2RGB);

        memcpy(rgb_image_ptr, rgb_image.ptr<void>(), width * height * ImageDefinitions::RGB_BPP);
        ippiRGBToCbYCr422_8u_C3C2R(rgb_image_ptr, width * ImageDefinitions::RGB_BPP, cbycr_image_ptr, width * ImageDefinitions::UYVY_BPP, ipp_roi);

        testing_class.set_cbycr_image(cbycr_image_ptr);
        testing_class.upload_data(cbycr_image_ptr);
        testing_class.create_reference_image();

        for (auto &i : enum_str_map)
        {
            testing_class.upload_reference();
            testing_class.convert_CbYCrToBGR(i.first);
            auto [ref_nrm, dut_nrm, diff] = testing_class.compute_rel_err();
            testing_class.zero_data();

            std::cout << file << ":\t" << ref_nrm << " " << dut_nrm << " " << diff << "\n";

            accum[i.first] += diff;
        }

        break;
    }

    for (auto &i : enum_str_map)
    {
        std::cout << "Result for " << i.second << ": " << accum[i.first] / 1 << "\n";
        // std::cout << "Result for " << i.second << ": " << accum[i.first] / image_filepaths.size() << "\n";
    }

    ippFree(rgb_image_ptr);
    ippFree(cbycr_image_ptr);

    std::cout << std::endl;
}
