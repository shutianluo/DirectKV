#include <cuda_fp16.h>
#include <cublas_v2.h>
#include "linear.h"

bool has_handle = false;
cublasHandle_t handle;

void cublas_init_handle(){
	cublasStatus_t status = cublasCreate(&handle);
	if (status != CUBLAS_STATUS_SUCCESS) {
		std::cerr << "cublas_init_handle failed: " << status << std::endl;
		throw std::runtime_error("cublas_init_handle failed");
	}
}

/* Row major 
 *   A (m x k) einsum(ik, jk -> ij) B (n x k) = C (m x n) 
 * Equivalent to column major 
 *   B (n x k) @ A^T (k x m) = C^T (m x n)
 */
void array_linear(
	int m,
	int n,
	int k,
	const half* Aarray,
	const half* Barray,
	half* Carray
) {
  const float alpha = 1.0;
  const float beta = 0.0;
	cublasStatus_t status = cublasSgemmEx(
		handle,
		CUBLAS_OP_T,
		CUBLAS_OP_N,
		n,
		m,
		k,
		&alpha,
		Barray,
		CUDA_R_16F,
		k,
		Aarray,
		CUDA_R_16F,
    k,
		&beta,
		Carray,
		CUDA_R_16F,
		n
	);
	if (status != CUBLAS_STATUS_SUCCESS) {
		std::cerr << "cublasGemmEx failed: " << status << std::endl;
		throw std::runtime_error("cublasGemmEx failed");
	}
}

torch::Tensor linear(
  torch::Tensor a,
  torch::Tensor w
) {
  int m = a.size(0);
  int k = a.size(1);
  int n = w.size(0);
	auto r = torch::empty({m, n}, a.options());

  const half* a_data = (half*)a.data_ptr();
	const half* w_data = (half*)w.data_ptr();
	half* r_data = (half*)r.data_ptr();

	if (!has_handle) {
		cublas_init_handle();
		has_handle = true;
	}

	array_linear(
		m,
		n,
		k,
		a_data,
		w_data,
		r_data
	);

	return r;
}

void linear_inplace(
	torch::Tensor a,
	torch::Tensor w,
	torch::Tensor r
) {
	int m = a.size(0);
	int k = a.size(1);
	int n = w.size(0);

	const half* a_data = (half*)a.data_ptr();
	const half* w_data = (half*)w.data_ptr();
	half* r_data = (half*)r.data_ptr();

	if (!has_handle) {
		cublas_init_handle();
		has_handle = true;
	}

	array_linear(
		m,
		n,
		k,
		a_data,
		w_data,
		r_data
	);
}