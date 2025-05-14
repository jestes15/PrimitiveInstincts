CUDA_LINKER_FLAGS = -lcusparse -lcusolver -lcublas -lcupti -lnppc -lnppial -lnppicc -lnppidei -lnppif -lnppig -lnppim
CUDA_COMPILER = nvcc

CXX_FLAGS = -Wno-deprecated-gpu-targets -g -O0 -std=c++17 --generate-line-info -arch=sm_89 -Xcudafe "--diag_suppress=611"

OPENCV_LINKER_FLAGS = -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_core 
OPENCV_INCLUDE_DIR = -I/usr/include/opencv4

all:
	$(CUDA_COMPILER) $(CXX_FLAGS) -o main img_proc_with_npp.cu $(CUDA_LINKER_FLAGS) $(OPENCV_INCLUDE_DIR) $(OPENCV_LINKER_FLAGS)