// #ifndef A64DC06E_2611_4CC3_BAE3_BAD6CC3597AF
// #define A64DC06E_2611_4CC3_BAE3_BAD6CC3597AF

// // CUDA copy kind
// #include <cuda.h>
// #include <npp.h>

// #include <stdint.h>
// #include <stdio.h>

// #include <functional>
// #include <memory>

// namespace npp_stream_ctx
// {
// NppStreamContext create_npp_stream_ctx()
// {
//     cudaStream_t stream;
//     cudaStreamCreate(&stream);

//     cudaDeviceProp properties;
//     cudaGetDeviceProperties_v2(&properties, 0);

//     int major = 0;
//     int minor = 0;
//     cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0);
//     cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0);

//     uint32_t flags = 0;
//     cudaStreamGetFlags(stream, &flags);

//     // Set up NPP Stream Context
//     NppStreamContext context = {.hStream = 0,
//                                 .nCudaDeviceId = cudaGetDevice(0),
//                                 .nMultiProcessorCount = properties.multiProcessorCount,
//                                 .nMaxThreadsPerMultiProcessor = properties.maxThreadsPerMultiProcessor,
//                                 .nMaxThreadsPerBlock = properties.maxThreadsPerBlock,
//                                 .nSharedMemPerBlock = properties.sharedMemPerBlock,
//                                 .nCudaDevAttrComputeCapabilityMajor = major,
//                                 .nCudaDevAttrComputeCapabilityMinor = minor,
//                                 .nStreamFlags = flags};

//     return context;
// }
// } // namespace npp_stream_ctx

// class cuda_mem_ctrl
// {
//   public:
//     template <typename T> static std::shared_ptr<T> cuda_host_malloc(uint64_t num_of_bytes)
//     {
//         printf("LOG: INFO -- %s entered\n", __PRETTY_FUNCTION__);
//         int obj_bytes = sizeof(T) * num_of_bytes;
//         T *ptr = nullptr;
//         if (cudaMallocHost(reinterpret_cast<void **>(&ptr), obj_bytes) != cudaSuccess)
//         {
//             printf("LOG: ERROR in %s\n", __PRETTY_FUNCTION__);
//             return std::shared_ptr<T>(nullptr);
//         }

//         return std::shared_ptr<T>(ptr, cuda_mem_ctrl::__cuda_free_host<T>);
//     }
//     template <typename T> static std::shared_ptr<T> cuda_dev_malloc(uint64_t num_of_bytes)
//     {
//         printf("LOG: INFO -- %s entered\n", __PRETTY_FUNCTION__);
//         int obj_bytes = sizeof(T) * num_of_bytes;
//         T *ptr = nullptr;
//         if (cudaMalloc(reinterpret_cast<void **>(&ptr), obj_bytes) != cudaSuccess)
//         {
//             printf("LOG: ERROR in %s\n", __PRETTY_FUNCTION__);
//             return std::shared_ptr<T>(nullptr);
//         }

//         return std::shared_ptr<T>(ptr, cuda_mem_ctrl::__cuda_free_dev<T>);
//     }
//     template <typename T>
//     static int cuda_cpy(std::shared_ptr<T> shared_src,
//                         std::shared_ptr<T> shared_dst,
//                         uint64_t size,
//                         cudaMemcpyKind mv_type)
//     {
//         printf("LOG: INFO -- %s entered\n", __PRETTY_FUNCTION__);
//         int attr = 0;

//         T *src = shared_src.get();
//         T *dst = shared_dst.get();

//         switch (mv_type)
//         {
//         case cudaMemcpyHostToHost:
//             if (src == nullptr || dst == nullptr)
//                 return cudaErrorInvalidHostPointer;

//             return cudaMemcpy(dst, src, size, mv_type);
//             break;
//         case cudaMemcpyHostToDevice:
//             if (src == nullptr)
//                 return cudaErrorInvalidHostPointer;
//             if (dst == nullptr)
//                 return cudaErrorInvalidDevicePointer;

//             return cudaMemcpy(dst, src, size, mv_type);
//             break;
//         case cudaMemcpyDeviceToHost:
//             if (src == nullptr)
//                 return cudaErrorInvalidDevicePointer;
//             if (dst == nullptr)
//                 return cudaErrorInvalidHostPointer;

//             return cudaMemcpy(dst, src, size, mv_type);
//             break;
//         case cudaMemcpyDeviceToDevice:
//             if (src == nullptr)
//                 return cudaErrorInvalidDevicePointer;
//             if (dst == nullptr)
//                 return cudaErrorInvalidDevicePointer;

//             return cudaMemcpy(dst, src, size, mv_type);
//             break;
//         case cudaMemcpyDefault:
//             cuDeviceGetAttribute(&attr, CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING, 0);

//             if (attr == 0)
//                 return cudaErrorInvalidMemcpyDirection;
//             if (src == nullptr || dst == nullptr)
//                 return cudaErrorInvalidDevicePointer;

//             return cudaMemcpy(dst, src, size, mv_type);
//             break;

//         default:
//             return cudaErrorInvalidMemcpyDirection;
//         }
//     }
//     template <typename T> static void __cuda_free_host(T *pointer)
//     {
//         printf("LOG: INFO -- %s entered\n", __PRETTY_FUNCTION__);
//         cudaFreeHost(pointer);
//     }
//     template <typename T> static void __cuda_free_dev(T *pointer)
//     {
//         printf("LOG: INFO -- %s entered\n", __PRETTY_FUNCTION__);
//         cudaFree(pointer);
//     }
// };

// #endif /* A64DC06E_2611_4CC3_BAE3_BAD6CC3597AF */
