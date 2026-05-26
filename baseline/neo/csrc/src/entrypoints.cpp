#include <torch/extension.h>

#include "block_swapping.h"
#include "small_kernels.h"
#include "attention.h"
#include "linear.h"

PYBIND11_MODULE(swiftllm_c, m) {
  m.def("swap_blocks", &swap_blocks);
  
  m.def("fused_add_rmsnorm_inplace", &fused_add_rmsnorm_inplace);
  m.def("silu_and_mul_inplace", &silu_and_mul_inplace);
  m.def("rotary_embedding_inplace", &rotary_embedding_inplace);
  m.def("store_kvcache", &store_kvcache);
  m.def("embedding", &embedding);

  // m.def("paged_attention", &paged_attention);

  m.def("linear", &linear);
}
