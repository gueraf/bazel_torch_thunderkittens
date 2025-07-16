namespace examples::thunder_kittens {

float* to_device(const std::vector<float>& vec) {
  float* d_ptr;
  cudaMalloc(&d_ptr, vec.size() * sizeof(float));
  cudaMemcpy(d_ptr, vec.data(), vec.size() * sizeof(float),
             cudaMemcpyHostToDevice);
  return d_ptr;
}

void gmm_gpu(const float* A, const float* B, float alpha, float beta, int K,
             int L, int M, float* C) {
  // TODO
}

}  // namespace examples::thunder_kittens