cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(PagedAttentionCPU C CXX)

# Meta information
set(TARGET_NAME_PREFIX pacpu)
set(ISPC_SRC_NAME pacpu)
# set(ISPC_TARGETS "avx2")
# set(ISPC_TARGETS "avx512spr-x16")
set(ISPC_FLAGS "-O3" "--opt=fast-math")
set(TARGET_SOURCES "pacpu.cpp" "pacpu.ispc")

set(ISPC_ARCH "x86-64")
set(ISPC_ARCH_BIT "64")

# Set a small number here, we don't actually compile any CUDA kernels
set(CMAKE_CUDA_ARCHITECTURES 70)

enable_language(ISPC)
set(GEN_TORCH_LIBRARY TRUE)

function(target_add_common_options target model_name tp_degree)
  set_property(TARGET ${target} PROPERTY CXX_STANDARD 17)
  set_property(TARGET ${target} PROPERTY POSITION_INDEPENDENT_CODE ON)
  set_property(TARGET ${target} PROPERTY ISPC_INSTRUCTION_SETS "${ISPC_TARGETS}")
  target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:ISPC>:${ISPC_FLAGS}>)
  target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:ISPC>:--arch=${ISPC_ARCH}>)

  set(arch_flag "-m${ISPC_ARCH_BIT}")
  target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:C,CXX>:-Ofast -march=native>)
  target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:C,CXX>:-fopenmp>)
  target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:C,CXX>:${arch_flag}>)
  string(TOUPPER ${model_name} MODEL_NAME)
  string(TOUPPER ${tp_degree} TP_DEGREE)
  target_compile_definitions(${target} PRIVATE "${MODEL_NAME}" TP_DEGREE=${TP_DEGREE})

  find_package(OpenMP REQUIRED)
target_link_libraries(${target} PRIVATE OpenMP::OpenMP_CXX)
endfunction()

function(gen_torch_lib model_name tp_degree)
  set(TARGET_NAME "${TARGET_NAME_PREFIX}-${model_name}-tp${tp_degree}")

  add_library(${TARGET_NAME} SHARED)
  target_sources(${TARGET_NAME} PRIVATE ${TARGET_SOURCES})
  target_add_common_options(${TARGET_NAME} ${model_name} ${tp_degree})
  target_compile_options(${TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:C,CXX>:${TORCH_CXX_FLAGS}>)
  target_link_libraries(${TARGET_NAME} PRIVATE "${TORCH_LIBRARIES}")
endfunction()

if (GEN_TORCH_LIBRARY)
  find_package(Torch REQUIRED)
  # set(CMAKE_CXX_FLAGS "-Ofast -march=native ${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

  # Create library, Model and TP should be pass as arguments
  gen_torch_lib(${Model} ${TP})
endif()
