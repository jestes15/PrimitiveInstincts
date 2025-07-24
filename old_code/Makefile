SRC_DIR = .
OBJ_DIR = obj
INC_DIR = .
EXE = main
CUDA_ARCH_FLAGS = -arch=sm_89

CUDA_COMPILER = nvcc

OPENCV_INCLUDE_DIR = /usr/local/include/opencv4
IPP_INCLUDE_DIR = /opt/intel/ipp/include
CUDA_INCLUDE_DIR = /usr/local/cuda/include

CUDA_LD_FLAGS = -lcuda -lcupti -lnppc -lnppial -lnppicc -lnppidei -lnppif -lnppig -lnppim -lcublas -lcudart -lnvrtc
IPP_LD_FLAGS = -L/opt/intel/ipp/lib/intel64 -lippcore -lipps -lippi -lippcc -lippdc
OPENCV_LD_FLAGS = -L/usr/local/lib/opencv4 -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_core -lopencv_cudaarithm -lopencv_cudawarping -lopencv_cudaimgproc

CXX_FLAGS = -Wno-deprecated-gpu-targets -g -O0 $(CUDA_ARCH_FLAGS) -std=c++20 --generate-line-info -Xcudafe "--diag_suppress=611" -I $(INC_DIR) -isystem $(OPENCV_INCLUDE_DIR) -I $(IPP_INCLUDE_DIR) -isystem $(CUDA_INCLUDE_DIR)
LD_FLAGS = -Wno-deprecated-gpu-targets $(CUDA_LD_FLAGS) $(OPENCV_LD_FLAGS) $(IPP_LD_FLAGS)

SRC_FILES := $(wildcard $(SRC_DIR)/*.cu)
OBJ_FILES := $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/%.o,$(SRC_FILES))

build: $(OBJ_FILES) $(EXE)

# Rule to build object files from src
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	@mkdir -p $(@D)
	$(CUDA_COMPILER) $(CXX_FLAGS) -c -o $@ $<

# Rule to build object files from tests (CTRL_DIR)
$(CTRL_OBJ_DIR)/%.o: $(CTRL_DIR)/%.cu $(OBJ_FILES)
	@mkdir -p $(@D)
	$(CUDA_COMPILER) $(CXX_FLAGS) -c -o $@ $<

# Rule to link test executables, linking with all object files from src
$(EXE): $(OBJ_FILES)
	@mkdir -p $(@D)
	$(CUDA_COMPILER) $^ -o $@ $(LD_FLAGS)

.PHONY: clean
clean:
	rm -f $(OBJ_DIR)/*.o *.png main test *.ptx