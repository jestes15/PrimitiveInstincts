#ifndef A64DC06E_2611_4CC3_BAE3_BAD6CC3597AF
#define A64DC06E_2611_4CC3_BAE3_BAD6CC3597AF

// CUDA copy kind
#include <cuda.h>

#include <stdint.h>

#include <memory>

class cuda_mem_ctrl
{
  public:
    template <typename T> static std::unique_ptr<T> cuda_host_malloc(uint64_t num_of_bytes)
    {
        printf("LOG: INFO -- %s entered\n", __PRETTY_FUNCTION__);
        int obj_bytes = sizeof(T) * num_of_bytes;
        T *ptr = nullptr;
        if (cudaMallocHost(reinterpret_cast<void **>(&ptr), obj_bytes) != cudaSuccess)
        {
            printf("LOG: ERROR in %s\n", __PRETTY_FUNCTION__);
            return std::unique_ptr<T>(nullptr);
        }

        std::unique_ptr uPtr(ptr, cuda_mem_ctrl::__cuda_free_host<T>);

        return uPtr;
    }
    template <typename T> static std::unique_ptr<T> cuda_dev_malloc(uint64_t num_of_bytes)
    {
        printf("LOG: INFO -- %s entered\n", __PRETTY_FUNCTION__);
        int obj_bytes = sizeof(T) * num_of_bytes;
        T *ptr = nullptr;
        if (cudaMalloc(reinterpret_cast<void **>(&ptr), obj_bytes) != cudaSuccess)
        {
            printf("LOG: ERROR in %s\n", __PRETTY_FUNCTION__);
            return std::unique_ptr<T>(nullptr);
        }

        std::unique_ptr uPtr(ptr, cuda_mem_ctrl::__cuda_free_dev<T>);

        return uPtr;
    }
    template <typename T> static int cuda_cpy(T *src, T *dst, uint64_t size, cudaMemcpyKind mv_type)
    {
        printf("LOG: INFO -- %s entered\n", __PRETTY_FUNCTION__);
        switch (mv_type)
        {
        case cudaMemcpyHostToHost:
            if (src == nullptr || dst == nullptr)
                return cudaErrorInvalidHostPointer;

            return cudaMemcpy(dst, src, size, mv_type);
            break;
        case cudaMemcpyHostToDevice:
            if (src == nullptr)
                return cudaErrorInvalidHostPointer;
            if (dst == nullptr)
                return cudaErrorInvalidDevicePointer;

            return cudaMemcpy(dst, src, size, mv_type);
            break;
        case cudaMemcpyDeviceToHost:
            if (src == nullptr)
                return cudaErrorInvalidDevicePointer;
            if (dst == nullptr)
                return cudaErrorInvalidHostPointer;

            return cudaMemcpy(dst, src, size, mv_type);
            break;
        case cudaMemcpyDeviceToDevice:
            if (src == nullptr)
                return cudaErrorInvalidDevicePointer;
            if (dst == nullptr)
                return cudaErrorInvalidDevicePointer;

            return cudaMemcpy(dst, src, size, mv_type);
            break;
        case cudaMemcpyDefault:
            int attr = 0;
            cuDeviceGetAttribute(&attr, CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING, 0);

            if (attr == 0)
                return cudaErrorInvalidMemcpyDirection;
            if (src == nullptr || dst == nullptr)
                return cudaErrorInvalidDevicePointer;

            return cudaMemcpy(dst, src, size, mv_type);
            break;

        default:
            return cudaErrorInvalidMemcpyDirection;
        }
    }

  private:
    template <typename T> static void __cuda_free_host(T *pointer)
    {
        printf("LOG: INFO -- %s entered\n", __PRETTY_FUNCTION__);
        cudaFreeHost(pointer);
    }
    template <typename T> static void __cuda_free_dev(T *pointer)
    {
        printf("LOG: INFO -- %s entered\n", __PRETTY_FUNCTION__);
        cudaFree(pointer);
    }
};

#endif /* A64DC06E_2611_4CC3_BAE3_BAD6CC3597AF */
