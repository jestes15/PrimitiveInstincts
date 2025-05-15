#include <iostream>

#include <opencv2/opencv.hpp>


int main()
{
    cudaSetDevice(0);

    std::cout << "##########################################################" << std::endl;
    std::cout << "# Image Processing Using NVIDIA Performance Primitives   #" << std::endl;
    std::cout << "#\tCreated by Joshua Estes                          #" << std::endl;
    std::cout << "##########################################################" << std::endl;

    uint8_t *device_source_image = nullptr;
    uint8_t *device_intermediate_storage = nullptr;
    uint8_t *device_dst_image = nullptr;
    uint8_t *host_int = nullptr;
    uint8_t *final_image = nullptr;

    // Load the image
    cv::Mat image =

    int width = image.cols;
    int height = image.rows;

    cudaMalloc((void **)&device_source_image, width * height * image_sizes::RGB24_bytes_per_pixel);
    cudaMalloc((void **)&device_intermediate_storage, width * height * image_sizes::CbYCr422_bytes_per_pixel);
    cudaMalloc((void **)&device_dst_image, width * height * image_sizes::RGB24_bytes_per_pixel);

    cudaMallocHost((void **)&final_image, width * height * image_sizes::RGB24_bytes_per_pixel);
    cudaMallocHost((void **)&host_int, width * height * image_sizes::CbYCr422_bytes_per_pixel);

    // Check if the image was loaded successfully
    if (image.empty())
    {
        std::cout << "Could not open or find the image!" << std::endl;
        return -1;
    }

    for (int i = 0; i < 6; i++)
        printf("data[%d] - > %d\n", i, image.data[i]);

    // Copy the CbYCr422 data to device memory
    cudaMemcpy(device_source_image,
               image.data,
               width * height * image_sizes::RGB24_bytes_per_pixel,
               cudaMemcpyHostToDevice);

    NppiSize roiSize = {.width = width, .height = height};
    printf("nppi func ret: %d\n",
           nppiRGBToCbYCr422_8u_C3C2R(device_source_image,
                                          width * image_sizes::RGB24_bytes_per_pixel,
                                          device_intermediate_storage,
                                          width * image_sizes::CbYCr422_bytes_per_pixel,
                                          roiSize));

    // nppiRGB24ToCbYCr422(device_source_image, device_intermediate_storage, width, height);

    cudaMemcpy(host_int,
               device_intermediate_storage,
               width * height * image_sizes::CbYCr422_bytes_per_pixel,
               cudaMemcpyDeviceToHost);

    for (int i = 0; i < 6; i++)
        printf("src[%d] - > %d\n", i, host_int[i]);

    convert_CbYCr_To_BGR24(device_intermediate_storage, device_dst_image, height, width);

    // Copy the result back to host memory
    cudaMemcpy(final_image,
               device_dst_image,
               width * height * image_sizes::RGB24_bytes_per_pixel,
               cudaMemcpyDeviceToHost);

    // Create a Mat for the RGB image
    // cv::Mat rgbImage(height, width, CV_8UC3, final_image);
    // cv::imshow("images_1920x1080/image_23.jpg pre-process", image);
    // cv::imshow("images_1920x1080/image_23.jpg post-process", rgbImage);

    cudaFree(device_source_image);
    cudaFree(device_intermediate_storage);
    cudaFree(device_dst_image);
    cudaFreeHost(final_image);
    cudaFreeHost(host_int);

    cudaDeviceReset();

    // cv::waitKey(0);          // Wait for a key press
    // cv::destroyAllWindows(); // Close all OpenCV windows
    return 0;
}