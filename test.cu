#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

int main()
{
    thrust::host_vector<float> src(1920 * 1080 * 3);

    printf("size: %ld\n", src.size());
    printf("capacity: %ld\n", src.capacity());
    printf("Type: %s\n", typeid(src.begin()).name());
    printf("Size of type: %ld\n", sizeof(src.begin()));
    printf("Type: %s\n", typeid(src.front()).name());
    printf("Size of type: %ld\n", sizeof(src.front()));
}