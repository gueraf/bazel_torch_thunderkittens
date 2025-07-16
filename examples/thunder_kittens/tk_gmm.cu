#include "kittens.cuh"

namespace examples::thunder_kittens {

float *to_device(const std::vector<float> &vec) {
  float *d_ptr;
  cudaMalloc(&d_ptr, vec.size() * sizeof(float));
  cudaMemcpy(d_ptr, vec.data(), vec.size() * sizeof(float),
             cudaMemcpyHostToDevice);
  return d_ptr;
}

void gmm_gpu(const float *A, const float *B, float alpha, float beta, int K,
             int L, int M, float *C) {
  // TODO
}

template <int M_BLOCK, int N_BLOCK>
struct matmul_layout {
  using base_tile = kittens::st_bf<64, 64>;
  using global_layout = kittens::gl<kittens::bf16, 1, 1, -1, -1, base_tile>;
  struct globals {
    global_layout A, B, C;
  };
  struct input_block {
    base_tile a[M_BLOCK], b[N_BLOCK];
  };
  struct finish_block {
    base_tile c[M_BLOCK][N_BLOCK];
  };
  struct common_state {
    int2 coord;
  };
  struct consumer_state {
    kittens::rt_fl<16, N_BLOCK * base_tile::cols> accum;
  };
};

template <int _M_BLOCK = 2, int _N_BLOCK = 4, int _SUPER_M = 12>
struct matmul_template {
  static constexpr int M_BLOCK = _M_BLOCK, N_BLOCK = _N_BLOCK,
                       SUPER_M = _SUPER_M;
  using layout = matmul_layout<M_BLOCK, N_BLOCK>;
  using wide_tile = kittens::st_bf<64, 64 * N_BLOCK>;
  static constexpr int NUM_CONSUMER_WARPS = M_BLOCK * 4, INPUT_PIPE_STAGES = 4,
                       PRODUCER_BARRIER_ARRIVALS = 1;
  // Helper functions
  template <bool PERISISTENT_GRID = true>
  __host__ static inline dim3 grid(int M, int N, int K) {
    return dim3(
        PERISISTENT_GRID
            ? 132
            : M * N / (M_BLOCK * N_BLOCK * layout::base_tile::num_elements));
  }
  // ThunderKittens template functions
  //   __device__ static inline void common_setup(
  //       kittens::common_setup_args<layout> args) {
  //     int Rblocks = args.globals.C.rows() / (M_BLOCK * 64),
  //         Cblocks = args.globals.C.cols() / (N_BLOCK * 64);
  //     int super_rows = (Rblocks / SUPER_M) * SUPER_M,
  //         final_rows = Rblocks - super_rows, super_repeat = SUPER_M *
  //         Cblocks;
  //     int task_id = args.task_iter * gridDim.x + blockIdx.x;
  //     if (task_id < super_rows * Cblocks)
  //       args.common.coord = {
  //           SUPER_M * (task_id / super_repeat) + task_id % SUPER_M,
  //           (task_id % super_repeat) / SUPER_M};
  //     else if (task_id < Rblocks * Cblocks) {
  //       int remainder_id = task_id - super_rows * Cblocks;
  //       args.common.coord = {super_rows + (remainder_id % final_rows),
  //                            remainder_id / final_rows};
  //     } else {  // Id is too high, no more work to do
  //       args.num_iters = -1;
  //       return;
  //     }
  //     args.num_iters = args.globals.A.cols() / 64;
  //     int id = kittens::warpgroup::groupid() == NUM_CONSUMER_WARPS / 4
  //                  ? 0
  //                  : kittens::warpgroup::groupid();  // producer sets as 0
  //     args.common.coord = {args.common.coord.x * M_BLOCK + id,
  //                          args.common.coord.y * N_BLOCK};
  //   }
  //   struct producer {
  //     __device__ static void setup(producer_setup_args<layout> args) {
  //       kittens::warpgroup::decrease_registers<40>();  // decrease registers
  //         for
  //           producers
  //     }
  //     __device__ static void load(producer_load_args<layout> args) {
  //       if (kittens::warpgroup::warpid() == 0) {
  //         kittens::tma::expect(args.inputs_arrived, args.input);
  //         for (int i = 0; i < M_BLOCK; i++)
  //           kittens::tma::load_async(args.input.a[i], args.globals.A,
  //                                    {args.common.coord.x + i, args.iter},
  //                                    args.inputs_arrived);
  //         for (int i = 0; i < N_BLOCK; i++)
  //           kittens::tma::load_async(args.input.b[i], args.globals.B,
  //                                    {args.iter, args.common.coord.y + i},
  //                                    args.inputs_arrived);
  //       }
  //     }
  //   };
  //   struct consumer {
  //     __device__ static void setup(consumer_setup_args<layout> args) {
  //       kittens::warpgroup::increase_registers<232>();  // increase registers
  //       for consumers zero(args.state.accum);
  //     }
  //     __device__ static void compute(consumer_compute_args<layout> args) {
  //       kittens::warpgroup::mma_AB(
  //           args.state.accum,                              // dest
  //           registers args.input.a[kittens::warpgroup::groupid()],  // A
  //           matrix reinterpret_cast<wide_tile &>(args.input.b)    // B
  //           matrix);
  //       kittens::warpgroup::mma_async_wait();
  //       if (laneid() == 0) kittens::arrive(args.inputs_finished);
  //     }
  //     __device__ static void finish(consumer_finish_args<layout> args) {
  //       kittens::warpgroup::store(
  //           reinterpret_cast<wide_tile
  //           &>(args.finish.c[kittens::warpgroup::groupid()]),
  //           args.state.accum);
  //       kittens::warpgroup::sync(kittens::warpgroup::groupid() + 4);
  //       if (kittens::warpgroup::warpid() == 0)
  //         for (int i = 0; i < N_BLOCK; i++) {
  //           kittens::tma::store_async(args.globals.C,
  //                            args.finish.c[kittens::warpgroup::groupid()][i],
  //                            {args.common.coord.x, args.common.coord.y + i});
  //           kittens::tma::store_async_read_wait();  // wait that store is
  //           finished before
  //           // reusing finish memory
  //         }
  //       kittens::zero(args.state.accum);
  //       if (kittens::laneid() == 0) kittens::arrive(args.finish_finished);
  //     }
  //   };
};

// template <typename mmt>
// void inner_run(bf16 *d_A, bf16 *d_B, bf16 *d_C, size_t M, size_t N, size_t K,
//                dim3 grid, dim3 block) {
//   using global_layout = typename mmt::layout::global_layout;
//   using globals = typename mmt::layout::globals;
//   global_layout Ag{d_A, nullptr, nullptr, M, K};
//   global_layout Bg{d_B, nullptr, nullptr, K, N};
//   global_layout Cg{d_C, nullptr, nullptr, M, N};
//   globals G{Ag, Bg, Cg};
//   prototype::lcf::kernel<mmt><<<grid, block, MAX_SHARED_MEMORY - 1024>>>(G);
// }

}  // namespace examples::thunder_kittens