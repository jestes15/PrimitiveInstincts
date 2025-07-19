// #include "kernels.h"

// int nppiRGB24ToCbYCr422(const uint8_t *__restrict__ device_src,
//                         uint8_t *__restrict__ device_dst,
//                         int width,
//                         int height,
//                         NppStreamContext context,
//                         const NppiRect &ROI_Size_Loc)
// {
//     // Validate input parameters
//     if (!device_src || !device_dst || width <= 0 || height <= 0 || (ROI_Size_Loc.x + ROI_Size_Loc.width) > width ||
//         (ROI_Size_Loc.y + ROI_Size_Loc.height) > height)
//         return -1; // Invalid parameters

//     int srcStep = width * image_sizes::RGB24_bytes_per_pixel;    // RGB24: 3 bytes per pixel
//     int dstStep = width * image_sizes::CbYCr422_bytes_per_pixel; // CbYCr422: 2 bytes per pixel

//     // Define ROI size
//     NppiSize roiSize = {0, 0};
//     uint32_t srcOffset = 0, dstOffset = 0;

//     if (ROI_Size_Loc.width == 0 || ROI_Size_Loc.height == 0)
//         roiSize = {.width = width, .height = height};
//     else
//     {
//         roiSize.width = ROI_Size_Loc.width;
//         roiSize.height = ROI_Size_Loc.height;

//         srcOffset = ROI_Size_Loc.y * srcStep + ROI_Size_Loc.x * image_sizes::RGB24_bytes_per_pixel;
//         dstOffset = ROI_Size_Loc.y * dstStep + ROI_Size_Loc.x * image_sizes::CbYCr422_bytes_per_pixel;
//     }

//     // Perform RGB to YCbCr422 conversion on GPU
//     nppiRGBToCbYCr422_8u_C3C2R_Ctx(device_src + srcOffset, srcStep, device_dst + dstOffset, dstStep, roiSize, context);
//     return 0; // Success
// }

// int nppiCbYCr422ToBGR24(const uint8_t *__restrict__ device_src,
//                         uint8_t *__restrict__ device_dst,
//                         int width,
//                         int height,
//                         NppStreamContext context,
//                         const NppiRect &ROI_Size_Loc)
// {
//     // Validate input parameters
//     if (!device_src || !device_dst || width <= 0 || height <= 0 || (ROI_Size_Loc.x + ROI_Size_Loc.width) > width ||
//         (ROI_Size_Loc.y + ROI_Size_Loc.height) > height)
//         return -1; // Invalid parameters

//     int srcStep = width * image_sizes::CbYCr422_bytes_per_pixel; // RGB24: 3 bytes per pixel
//     int dstStep = width * image_sizes::RGB24_bytes_per_pixel;    // CbYCr422: 2 bytes per pixel

//     // Define ROI size
//     NppiSize roiSize = {0, 0};
//     NppiSize fillROI = {0, 0};
//     uint32_t srcOffset = 0, dstOffset = 0, fillOffset = 0;

//     if (ROI_Size_Loc.width == 0 || ROI_Size_Loc.height == 0)
//     {
//         roiSize = {width, height};
//     }
//     else
//     {
//         roiSize.width = ROI_Size_Loc.width;
//         roiSize.height = ROI_Size_Loc.height;

//         fillROI.width = ROI_Size_Loc.width + ((ROI_Size_Loc.x == 0) ? 0 : 1) + ((ROI_Size_Loc.width == width) ? 0 : 1);
//         fillROI.height =
//             ROI_Size_Loc.height + ((ROI_Size_Loc.y == 0) ? 0 : 1) + ((ROI_Size_Loc.height == height) ? 0 : 1);
//         fillOffset = (ROI_Size_Loc.y - ((ROI_Size_Loc.y == 0) ? 0 : 1)) * dstStep +
//                      (ROI_Size_Loc.x - ((ROI_Size_Loc.x == 0) ? 0 : 1)) * image_sizes::RGB24_bytes_per_pixel;

//         srcOffset = ROI_Size_Loc.y * srcStep + ROI_Size_Loc.x * image_sizes::CbYCr422_bytes_per_pixel;
//         dstOffset = ROI_Size_Loc.y * dstStep + ROI_Size_Loc.x * image_sizes::RGB24_bytes_per_pixel;
//     }

//     if (fillROI.width != 0 && fillROI.height != 0)
//     {
//         const Npp8u border_color[3] = {0, 0, 255};
//         nppiSet_8u_C3R_Ctx(border_color, device_dst + fillOffset, dstStep, fillROI, context);
//     }

//     // Perform YCbCr422 to BGR conversion on GPU
//     nppiCbYCr422ToBGR_709HDTV_8u_C2C3R_Ctx(device_src + srcOffset, srcStep, device_dst + dstOffset, dstStep, roiSize,
//                                            context);
//     return 0; // Success
// }
