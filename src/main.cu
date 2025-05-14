#include "cuda_mem_ctrl.h"
#include "kernels.h"

int main()
{
    cuInit(0);

    uint8_t *src = nullptr, *dst = nullptr;
    cuda_mem_ctrl::cuda_cpy(src, dst, 0, cudaMemcpyDefault);
}