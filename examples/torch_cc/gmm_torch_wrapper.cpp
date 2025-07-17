#include <torch/extension.h>

#include <vector>

#include "examples/thunder_kittens/tk_gmm_utils.h"

namespace examples::torch_cc {

// Convert torch::Tensor to std::vector<float>
std::vector<float> tensor_to_vector(const torch::Tensor& tensor) {
  // Ensure tensor is contiguous and on CPU
  auto cpu_tensor = tensor.cpu().contiguous();
  auto data_ptr = cpu_tensor.data_ptr<float>();
  auto size = cpu_tensor.numel();
  return std::vector<float>(data_ptr, data_ptr + size);
}

// Convert std::vector<float> to torch::Tensor
torch::Tensor vector_to_tensor(const std::vector<float>& vec,
                               const std::vector<int64_t>& shape) {
  auto options = torch::TensorOptions().dtype(torch::kFloat32);
  auto tensor = torch::from_blob(const_cast<float*>(vec.data()), shape,
                                 options)
                    .clone();  // Clone to ensure tensor owns the data
  return tensor;
}

// PyTorch wrapper for the GMM function
torch::Tensor gmm_torch(const torch::Tensor& A, const torch::Tensor& B,
                        float alpha, float beta, const torch::Tensor& C) {
  // Validate input dimensions
  TORCH_CHECK(A.dim() == 2, "Matrix A must be 2-dimensional");
  TORCH_CHECK(B.dim() == 2, "Matrix B must be 2-dimensional");
  TORCH_CHECK(C.dim() == 2, "Matrix C must be 2-dimensional");

  int64_t K = A.size(0);  // rows of A
  int64_t L = A.size(1);  // cols of A, rows of B
  int64_t M = B.size(1);  // cols of B

  TORCH_CHECK(B.size(0) == L,
              "Matrix dimensions don't match for multiplication");
  TORCH_CHECK(C.size(0) == K && C.size(1) == M,
              "Output matrix C has wrong dimensions");

  // Ensure all tensors are float32 and contiguous
  auto A_contig = A.to(torch::kFloat32).contiguous();
  auto B_contig = B.to(torch::kFloat32).contiguous();
  auto C_contig = C.to(torch::kFloat32).contiguous();

  // Convert to std::vector for the GMM function
  auto A_vec = tensor_to_vector(A_contig);
  auto B_vec = tensor_to_vector(B_contig);
  auto C_vec = tensor_to_vector(C_contig);

  // Call the original GMM function
  examples::thunder_kittens::gmm(A_vec, B_vec, alpha, beta, static_cast<int>(K),
                                 static_cast<int>(L), static_cast<int>(M),
                                 C_vec);

  // Convert result back to tensor
  return vector_to_tensor(C_vec, {K, M});
}

// Alternative interface that creates output tensor
torch::Tensor gmm_torch_create(const torch::Tensor& A, const torch::Tensor& B,
                               float alpha, float beta) {
  // Validate input dimensions
  TORCH_CHECK(A.dim() == 2, "Matrix A must be 2-dimensional");
  TORCH_CHECK(B.dim() == 2, "Matrix B must be 2-dimensional");

  int64_t K = A.size(0);  // rows of A
  int64_t L = A.size(1);  // cols of A, rows of B
  int64_t M = B.size(1);  // cols of B

  TORCH_CHECK(B.size(0) == L,
              "Matrix dimensions don't match for multiplication");

  // Create output tensor filled with zeros
  auto C = torch::zeros({K, M}, torch::TensorOptions().dtype(torch::kFloat32));

  return gmm_torch(A, B, alpha, beta, C);
}

}  // namespace examples::torch_cc

// Python module binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gmm", &examples::torch_cc::gmm_torch,
        "General Matrix Multiply: C = alpha * A @ B + beta * C");
  m.def("gmm_create", &examples::torch_cc::gmm_torch_create,
        "General Matrix Multiply with output creation: returns alpha * A @ B");
}
