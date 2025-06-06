SRC_DIR = src
OBJ_DIR = obj
INC_DIR = include
EXE = main

CUDA_COMPILER = nvcc

OPENCV_INCLUDE_DIR = /usr/include/opencv4

CUDA_LD_FLAGS = -lcuda -lcupti -lnppc -lnppial -lnppicc -lnppidei -lnppif -lnppig -lnppim
OPENCV_LD_FLAGS = -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_core 

CXX_FLAGS = -Wno-deprecated-gpu-targets -g -O2 -std=c++20 --generate-line-info -arch=compute_89 -code=sm_89 -Xcudafe "--diag_suppress=611" -I $(INC_DIR) -isystem $(OPENCV_INCLUDE_DIR)
LD_FLAGS = -Wno-deprecated-gpu-targets $(CUDA_LD_FLAGS) $(OPENCV_LD_FLAGS)

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
	$(CUDA_COMPILER) $(LD_FLAGS) $^ -o $@

.PHONY: clean
clean:
	rm -f $(OBJ_DIR)/*.o