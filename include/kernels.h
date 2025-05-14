#ifndef FD0C52F4_8FD3_4586_A7A9_E2963FBEA290
#define FD0C52F4_8FD3_4586_A7A9_E2963FBEA290

#include <cuda.h>
#include <cuda_runtime.h>
#include <generated_cuda_runtime_api_meta.h>
#include <mma.h>
#include <npp.h>

#include <stdint.h>

#include "device_structs.h"
#include "image_sizes.h"

int nppiRGB24ToCbYCr422(const uint8_t *__restrict__ device_src,
                        uint8_t *__restrict__ device_dst,
                        int width,
                        int height,
                        const NppiRect &ROI_Size_Loc);
int nppiCbYCr422ToBGR24(const uint8_t *__restrict__ device_src,
                        uint8_t *__restrict__ device_dst,
                        int width,
                        int height,
                        const NppiRect &ROI_Size_Loc);

void convert_CbYCr_To_BGR24(const uint8_t *__restrict__ src,
                            uint8_t *__restrict__ dst,
                            const uint32_t height,
                            const uint32_t width);

#endif /* FD0C52F4_8FD3_4586_A7A9_E2963FBEA290 */
