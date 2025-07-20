// Simple example demonstrating how to use ZLUDA
#include <cuda_runtime.h>
#include <dlfcn.h>

#include <filesystem>
#include <iostream>
#include <memory>
#include <string>

void check_loaded_libraries() {
  std::cout << "\n=== Checking Loaded Libraries ===" << std::endl;

  // Check if we can get info about the loaded CUDA library
  void* handle = dlopen(nullptr, RTLD_LAZY);
  if (!handle) {
    std::cout << "Failed to open handle to current process" << std::endl;
    return;
  }

  // RAII wrapper for dlclose
  auto handle_guard =
      std::unique_ptr<void, decltype(&dlclose)>(handle, dlclose);

  // Look for CUDA runtime symbols to see which library provides them
  void* symbol = dlsym(handle, "cudaGetDeviceCount");
  if (!symbol) {
    std::cout << "cudaGetDeviceCount symbol not found" << std::endl;
    return;
  }

  Dl_info info;
  if (!dladdr(symbol, &info)) {
    std::cout << "Failed to get symbol information" << std::endl;
    return;
  }

  std::cout << "cudaGetDeviceCount symbol loaded from: " << info.dli_fname
            << std::endl;

  // Resolve symlinks to get the absolute path using std::filesystem
  std::filesystem::path lib_path(info.dli_fname);
  std::error_code ec;
  std::filesystem::path real_path =
      std::filesystem::weakly_canonical(lib_path, ec);

  const char* resolved_fname = nullptr;
  std::string real_path_str;

  if (!ec) {
    real_path_str = real_path.string();
    resolved_fname = real_path_str.c_str();
    std::cout << "Resolved absolute path: " << resolved_fname << std::endl;
  } else {
    std::cout << "Failed to resolve path: " << ec.message() << std::endl;
    resolved_fname = info.dli_fname;  // fallback to original
  }

  // Check if the resolved path contains "zluda"
  std::string path_str(resolved_fname);
  if (path_str.find("zluda") != std::string::npos ||
      path_str.find("libcuda.so") != std::string::npos) {
    std::cout << "✅ ZLUDA library detected!" << std::endl;
  } else {
    std::cout << "⚠️  System CUDA library detected" << std::endl;
  }
}

void check_cuda_runtime_version() {
  std::cout << "\n=== CUDA Runtime Version ===" << std::endl;

  int runtimeVersion = 0;
  cudaError_t error = cudaRuntimeGetVersion(&runtimeVersion);

  if (error == cudaSuccess) {
    int major = runtimeVersion / 1000;
    int minor = (runtimeVersion % 1000) / 10;
    std::cout << "CUDA Runtime Version: " << major << "." << minor << std::endl;
  } else {
    std::cout << "Failed to get CUDA runtime version: "
              << cudaGetErrorString(error) << std::endl;
  }
}

void check_driver_version() {
  std::cout << "\n=== CUDA Driver Version ===" << std::endl;

  int driverVersion = 0;
  cudaError_t error = cudaDriverGetVersion(&driverVersion);

  if (error == cudaSuccess) {
    int major = driverVersion / 1000;
    int minor = (driverVersion % 1000) / 10;
    std::cout << "CUDA Driver Version: " << major << "." << minor << std::endl;
  } else {
    std::cout << "Failed to get CUDA driver version: "
              << cudaGetErrorString(error) << std::endl;
  }
}

int main() {
  std::cout << "ZLUDA Example - Testing CUDA Runtime" << std::endl;
  std::cout << "====================================" << std::endl;

  // Check which libraries are loaded
  check_loaded_libraries();

  // Check CUDA versions
  check_cuda_runtime_version();
  check_driver_version();

  std::cout << "\n=== Device Information ===" << std::endl;

  // Get CUDA device count
  int deviceCount = 0;
  cudaError_t error = cudaGetDeviceCount(&deviceCount);

  if (error != cudaSuccess) {
    std::cout << "CUDA Error: " << cudaGetErrorString(error) << std::endl;
    return 1;
  }

  std::cout << "Number of CUDA devices: " << deviceCount << std::endl;

  if (deviceCount > 0) {
    cudaDeviceProp prop;
    for (int i = 0; i < deviceCount; i++) {
      error = cudaGetDeviceProperties(&prop, i);
      if (error == cudaSuccess) {
        std::cout << "\nDevice " << i << ":" << std::endl;
        std::cout << "  Name: " << prop.name << std::endl;
        std::cout << "  Compute Capability: " << prop.major << "." << prop.minor
                  << std::endl;
        std::cout << "  Total Global Memory: "
                  << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
        std::cout << "  Multiprocessors: " << prop.multiProcessorCount
                  << std::endl;
        std::cout << "  Max Threads Per Block: " << prop.maxThreadsPerBlock
                  << std::endl;
      } else {
        std::cout << "Failed to get properties for device " << i << ": "
                  << cudaGetErrorString(error) << std::endl;
      }
    }
  } else {
    std::cout << "No CUDA devices found. This might indicate:" << std::endl;
    std::cout << "  - No AMD GPU available" << std::endl;
    std::cout << "  - ZLUDA not properly configured" << std::endl;
    std::cout << "  - ROCm drivers not installed" << std::endl;
  }

  std::cout << "\n====================================" << std::endl;
  std::cout << "ZLUDA verification complete." << std::endl;

  return 0;
}
