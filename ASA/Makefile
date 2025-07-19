CUDA_LINKER_FLAGS = -lcuda -lnppc -lnppial -lnppicc -lnppidei -lnppif -lnppig -lnppim -lcublas -lcudart
CUDA_COMPILER = nvcc

CXX_FLAGS = -Wno-deprecated-gpu-targets -g -O3 -std=c++17 --generate-line-info -arch=compute_90a -code=sm_90a -Xcudafe "--diag_suppress=611"

OPENCV_LINKER_FLAGS = -L/home/aubjne001/opencv_install/lib64 -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_core
OPENCV_INCLUDE_DIR = -I/home/aubjne001/opencv_install/include/opencv4

INTEL_IPP_LINKER_FLAGS = -L/apps/x86-64/apps/intel_2023.1.0/ipp/latest/lib/intel64 -lippcore -lipps -lippi -lippcc -lippdc
INTEL_IPP_INCLUDE_DIR = -I/apps/x86-64/apps/intel_2023.1.0/ipp/latest/include

# Command
all:
	$(CUDA_COMPILER) $(CXX_FLAGS) -o main src/img_conv.cu src/main.cu -I include $(CUDA_LINKER_FLAGS) $(OPENCV_INCLUDE_DIR) $(OPENCV_LINKER_FLAGS) $(INTEL_IPP_INCLUDE_DIR) $(INTEL_IPP_LINKER_FLAGS)