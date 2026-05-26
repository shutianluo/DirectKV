Torch_DIR=$(python -c 'import torch;print(torch.utils.cmake_prefix_path)')/Torch
CUDA_HOST_COMPILER_PATH=$(which g++-11)
CXX_COMPILER_PATH=$(which g++-13)

mkdir -p build
cmake -B build -S . -DTorch_DIR=$Torch_DIR -DModel=$1 -DTP=$2 -DCMAKE_CUDA_HOST_COMPILER=${CUDA_HOST_COMPILER_PATH} -DCMAKE_CXX_COMPILER=${CXX_COMPILER_PATH}
cmake --build build
