#include "kernels.h"

constexpr int width = 1920;
constexpr int height = 1080;
constexpr int resize_width = 1984;
constexpr int resize_height = 1984;

cv::Mat resizeKeepAspectRatioPadTopBottom(const cv::Mat &aInputMat,
                                          std::size_t aNewHeight,
                                          std::size_t aNewWidth,
                                          const cv::Scalar &aBgColor)
{
    // Resize first
    float tRatio = std::min(aNewWidth / (aInputMat.cols * 1.0), aNewHeight / (aInputMat.rows * 1.0));
    int tUnpadW = std::ceil(tRatio * aInputMat.cols);
    int tUnpadH = std::ceil(tRatio * aInputMat.rows);
    cv::Mat tResized(tUnpadH, tUnpadW, CV_8UC3);
    cv::resize(aInputMat, tResized, tResized.size(), 0, 0, cv::INTER_LINEAR);
    // Then padding
    // Determine pad height and width
    float tPadWidth = (aNewHeight - tUnpadW) / 2.0;
    float tPadHeight = (aNewHeight - tUnpadH) / 2.0;
    cv::Mat tResizedAndPadded(aNewHeight, aNewWidth, CV_8UC3, aBgColor);
    // Determine where to place padding
    int tTopPad = std::lround(tPadHeight - 0.1);
    int tBottomPad = std::lround(tPadHeight + 0.1);
    int tLeftPad = std::lround(tPadWidth - 0.1);
    int tRightPad = std::lround(tPadWidth + 0.1);
    int tBorderType = cv::BORDER_CONSTANT;
    cv::copyMakeBorder(
        tResized, tResizedAndPadded, tTopPad, tBottomPad, tLeftPad, tRightPad, tBorderType, aBgColor);

    return tResizedAndPadded;
}

cv::Mat blobFromGpuMat(const cv::Mat &aInputMat, bool aNormalize)
{
    cv::Mat tGpuDst(1, aInputMat.rows * aInputMat.cols, CV_8UC3);
    cv::Mat tFloat;

    size_t tWidth = aInputMat.cols * aInputMat.rows;
    std::vector<cv::Mat> tInputChannels{
        cv::Mat(aInputMat.rows, aInputMat.cols, CV_8U, &(tGpuDst.ptr()[0])),
        cv::Mat(aInputMat.rows, aInputMat.cols, CV_8U, &(tGpuDst.ptr()[tWidth])),
        cv::Mat(aInputMat.rows, aInputMat.cols, CV_8U, &(tGpuDst.ptr()[tWidth * 2]))};
    cv::split(aInputMat, tInputChannels);

    if (aNormalize)
    {
        tGpuDst.convertTo(tFloat, CV_32FC3, 1.0f / 255.0f);
    }
    else
    {
        tGpuDst.convertTo(tFloat, CV_32FC3);
    }

    return tFloat;
}

cv::Mat ipp_convertAndResize(uint8_t *cbycr_image)
{
    uint8_t *rgb = reinterpret_cast<uint8_t *>(ippMalloc(width * height * 3));
    ippiCbYCr422ToRGB_8u_C2C3R(cbycr_image, width * 2, rgb, width * 3, IppiSize{width, height});
    cv::Mat tDeviceImageMat(height, width, CV_8UC3, rgb);
    // cv::Mat cpuImg(height, width, CV_8UC3, rgb);
    cv::Mat tResizedInputImage =
        resizeKeepAspectRatioPadTopBottom(tDeviceImageMat, resize_height, resize_width, cv::Scalar(0, 0, 0));
    cv::Mat blob = blobFromGpuMat(tResizedInputImage, true);

    return blob;
}
