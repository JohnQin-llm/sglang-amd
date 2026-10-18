/*
 * Copyright (c) 2024 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Adapted from FlashInfer 8256ac5d070268e03a4dfa4a44ededcf041e291a.
// Accumulate mass in FP64 and use a strict upper-mass comparison to retain
// the ascending-CDF equality boundary.
// AITER supplies HIP vectors, block reductions and IEEE helpers.
#pragma once

#include <sgl_kernel/tensor.h>

#include <sgl_kernel/utils.cuh>

#include <limits>
#include <sampling.cuh>
namespace sglang {
namespace compact_sampling {
using namespace aiter;
using namespace aiter::sampling;
using MinReduceOp = AiterMinOp;
using MaxReduceOp = AiterMaxOp;
__device__ __forceinline__ float ieee_div(float a, float b) {
  return a / b;
}
namespace math {
__device__ __forceinline__ float ptx_rcp(float x) {
  return 1.0f / x;
}
}  // namespace math
template <uint32_t VEC_SIZE, uint32_t BLOCK_THREADS, BlockReduceAlgorithm REDUCE_ALGORITHM, typename TempStorage>
__device__ __forceinline__ float GetMaxValue(float* in_data, uint32_t row_idx, uint32_t d, TempStorage& temp_storage) {
  const uint32_t tx = threadIdx.x;
  vec_t<float, VEC_SIZE> in_data_vec;

  // Thread-local max accumulation (deferred reduction)
  float thread_max = 0.0f;
  for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {
    in_data_vec.fill(0);
    if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
      in_data_vec.cast_load(in_data + row_idx * d + (i * BLOCK_THREADS + tx) * VEC_SIZE);
    }
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; ++j) {
      thread_max = max(thread_max, static_cast<float>(in_data_vec[j]));
    }
  }

  // Single block reduction after loop completes
  float max_val = BlockReduce<double, BLOCK_THREADS, REDUCE_ALGORITHM>(temp_storage.block_prim.reduce)
                      .Reduce(thread_max, MaxReduceOp{});
  if (tx == 0) {
    temp_storage.max_val = max_val;
  }
  __syncthreads();
  return temp_storage.max_val;
}

template <uint32_t BLOCK_THREADS, BlockReduceAlgorithm REDUCE_ALGORITHM>
struct RenormTempStorage {
  union {
    typename BlockReduce<double, BLOCK_THREADS, REDUCE_ALGORITHM>::TempStorage reduce;
    typename BlockReduce<int, BLOCK_THREADS, REDUCE_ALGORITHM>::TempStorage reduce_int;
    typename BlockReduce<ValueCount<float>, BLOCK_THREADS, REDUCE_ALGORITHM>::TempStorage reduce_value_count;
  } block_prim;
  struct {
    float max_val;
    float min_val;
    double row_sum;
    union {
      struct {
        double values[2];
      };
      struct {
        int counts[2];
      };
      struct {
        ValueCount<float> pairs[2];
      };
    } block_aggregate;
  };
};

template <uint32_t BLOCK_THREADS, BlockReduceAlgorithm REDUCE_ALGORITHM, uint32_t VEC_SIZE, typename DType>
__global__ __launch_bounds__(BLOCK_THREADS) void TopPRenormProbKernel(
    DType* probs, DType* renormed_prob, float* top_p_arr, float top_p_val, uint32_t d, int* counts, int capacity) {
  const uint32_t bx = blockIdx.x, tx = threadIdx.x;
  const uint32_t row_idx = bx;
  if (counts != nullptr && counts[bx] <= capacity && top_p_arr[bx] < 1.0f) return;
  double p = top_p_arr == nullptr ? top_p_val : top_p_arr[bx];

  extern __shared__ __align__(alignof(RenormTempStorage<BLOCK_THREADS, REDUCE_ALGO>)) uint8_t smem_renorm[];
  auto& temp_storage = reinterpret_cast<RenormTempStorage<BLOCK_THREADS, REDUCE_ALGO>&>(smem_renorm);
  vec_t<float, VEC_SIZE> probs_vec;

  // Fast-path: when p >= 1.0 (e.g., p == 1.0), perform simple sum and normalization
  // Stage A: per-thread float accumulation over assigned lanes (vectorized)
  double thread_sum = 0.0f;
  const uint32_t num_iters = ceil_div(d, BLOCK_THREADS * VEC_SIZE);
  for (uint32_t i = 0; i < num_iters; ++i) {
    probs_vec.fill(0.0f);
    const uint32_t base_idx = (i * BLOCK_THREADS + tx) * VEC_SIZE;
    if (base_idx < d) {
      probs_vec.cast_load(probs + row_idx * d + base_idx);
    }
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; ++j) {
      const uint32_t idx = base_idx + j;
      if (idx < d) thread_sum += probs_vec[j];
    }
  }

  // Block reduce (float)
  double row_sum = BlockReduce<double, BLOCK_THREADS, REDUCE_ALGORITHM>(temp_storage.block_prim.reduce).Sum(thread_sum);
  // Broadcast via shared
  if (tx == 0) temp_storage.row_sum = row_sum;
  __syncthreads();
  row_sum = temp_storage.row_sum;

  if (p >= 1.0) {
    // Guard against zero sum
    const float denom = row_sum;
    const float normalizer = math::ptx_rcp(denom);

    // Stage B: normalize and store
    for (uint32_t i = 0; i < num_iters; ++i) {
      probs_vec.fill(0.0f);
      const uint32_t base_idx = (i * BLOCK_THREADS + tx) * VEC_SIZE;
      if (base_idx < d) {
        probs_vec.cast_load(probs + row_idx * d + base_idx);
      }
#pragma unroll
      for (uint32_t j = 0; j < VEC_SIZE; ++j) {
        const uint32_t idx = base_idx + j;
        float v = probs_vec[j];
        probs_vec[j] = (idx < d) ? (v * normalizer) : 0.0f;
      }
      if (base_idx < d) {
        probs_vec.cast_store(renormed_prob + row_idx * d + base_idx);
      }
    }
    return;  // Exit after fast-path processing
  }

  // Preserve the existing ascending-CDF threshold q=1-top_p even if FP32 inputs sum slightly off one.
  const double cutoff = static_cast<double>(1.0f - static_cast<float>(p));
  p = row_sum - cutoff;

  // Original Top-P renormalization logic
  temp_storage.max_val = 0;
  float max_val =
      GetMaxValue<VEC_SIZE, BLOCK_THREADS, REDUCE_ALGORITHM, RenormTempStorage<BLOCK_THREADS, REDUCE_ALGORITHM>>(
          probs, row_idx, d, temp_storage);

  if (cutoff >= row_sum) {
    // Match the reference's last-element clamp even for unnormalized input.
    double local_mass = 0;
    for (uint32_t i = 0; i < num_iters; ++i) {
      probs_vec.fill(0);
      const auto base = (i * BLOCK_THREADS + tx) * VEC_SIZE;
      if (base < d) probs_vec.cast_load(probs + row_idx * d + base);
      for (uint32_t j = 0; j < VEC_SIZE; ++j)
        if (base + j < d && probs_vec[j] == max_val) local_mass += probs_vec[j];
    }
    double kept_mass =
        BlockReduce<double, BLOCK_THREADS, REDUCE_ALGORITHM>(temp_storage.block_prim.reduce).Sum(local_mass);
    if (tx == 0) temp_storage.row_sum = kept_mass;
    __syncthreads();
    kept_mass = temp_storage.row_sum;
    for (uint32_t i = 0; i < num_iters; ++i) {
      const auto base = (i * BLOCK_THREADS + tx) * VEC_SIZE;
      if (base < d) {
        probs_vec.cast_load(probs + row_idx * d + base);
        for (uint32_t j = 0; j < VEC_SIZE; ++j)
          probs_vec[j] = probs_vec[j] == max_val ? float(double(probs_vec[j]) / kept_mass) : 0.0f;
        probs_vec.cast_store(renormed_prob + row_idx * d + base);
      }
    }
    return;
  }

  float low = 0, high = max_val;
  float min_gt_low, max_le_high;
  double sum_low = row_sum;
  // f(x) = sum(probs[probs > x]), f(x) is non-increasing
  // min_gt_low = min{p \in probs | p > low}, max_le_high = max{p \in probs | p <= high}
  // loop invariant:
  // - f(low) >= p, f(high) < p
  // - f(low) > f(min_gt_low) >= f(max_le_high) == f(high)
  // stopping condition
  // - f(low) >= p, f(min_gt_low) == f(max_le_high) == f(high) < p
  do {
    // Use IEEE-754 arithmetic (no FTZ) to prevent -use_fast_math from flushing
    // subnormal pivots to zero, which would stall the search (#769, #774).
    float pivot_0 = ieee_div(ieee_add(high, ieee_mul(2.f, low)), 3.f);
    float pivot_1 = ieee_div(ieee_add(ieee_mul(2.f, high), low), 3.f);

    double aggregate_gt_pivot_0 = 0, aggregate_gt_pivot_1 = 0;
    min_gt_low = high;
    max_le_high = low;
    double threadlocal_aggregate_gt_pivot_0 = 0;
    double threadlocal_aggregate_gt_pivot_1 = 0;
#pragma unroll 2
    for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {
      probs_vec.fill(0);
      if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
        probs_vec.cast_load(probs + row_idx * d + i * BLOCK_THREADS * VEC_SIZE + tx * VEC_SIZE);
      }

      float probs_gt_pivot_0[VEC_SIZE], probs_gt_pivot_1[VEC_SIZE];
#pragma unroll
      for (uint32_t j = 0; j < VEC_SIZE; ++j) {
        probs_gt_pivot_0[j] = (probs_vec[j] > pivot_0) ? probs_vec[j] : 0;
        probs_gt_pivot_1[j] = (probs_vec[j] > pivot_1) ? probs_vec[j] : 0;

        if (probs_vec[j] > low && (i * BLOCK_THREADS + tx) * VEC_SIZE + j < d) {
          min_gt_low = min(min_gt_low, probs_vec[j]);
        }
        if (probs_vec[j] <= high && (i * BLOCK_THREADS + tx) * VEC_SIZE + j < d) {
          max_le_high = max(max_le_high, probs_vec[j]);
        }
        threadlocal_aggregate_gt_pivot_0 += probs_gt_pivot_0[j];
        threadlocal_aggregate_gt_pivot_1 += probs_gt_pivot_1[j];
      }
    }
    aggregate_gt_pivot_0 = BlockReduce<double, BLOCK_THREADS, REDUCE_ALGORITHM>(temp_storage.block_prim.reduce)
                               .Sum(threadlocal_aggregate_gt_pivot_0);
    __syncthreads();
    aggregate_gt_pivot_1 = BlockReduce<double, BLOCK_THREADS, REDUCE_ALGORITHM>(temp_storage.block_prim.reduce)
                               .Sum(threadlocal_aggregate_gt_pivot_1);
    __syncthreads();

    min_gt_low = BlockReduce<double, BLOCK_THREADS, REDUCE_ALGORITHM>(temp_storage.block_prim.reduce)
                     .Reduce(min_gt_low, MinReduceOp{});
    __syncthreads();
    max_le_high = BlockReduce<double, BLOCK_THREADS, REDUCE_ALGORITHM>(temp_storage.block_prim.reduce)
                      .Reduce(max_le_high, MaxReduceOp{});
    if (tx == 0) {
      temp_storage.block_aggregate.values[0] = aggregate_gt_pivot_0;
      temp_storage.block_aggregate.values[1] = aggregate_gt_pivot_1;
      temp_storage.min_val = min_gt_low;
      temp_storage.max_val = max_le_high;
    }
    __syncthreads();
    aggregate_gt_pivot_0 = temp_storage.block_aggregate.values[0];
    aggregate_gt_pivot_1 = temp_storage.block_aggregate.values[1];
    min_gt_low = temp_storage.min_val;
    max_le_high = temp_storage.max_val;

    if (aggregate_gt_pivot_1 > p) {
      low = pivot_1;
      sum_low = aggregate_gt_pivot_1;
    } else if (aggregate_gt_pivot_0 > p) {
      low = pivot_0;
      high = min(pivot_1, max_le_high);
      sum_low = aggregate_gt_pivot_0;
    } else {
      high = min(pivot_0, max_le_high);
    }
  } while (min_gt_low < max_le_high && nextafterf(min_gt_low, max_le_high) < max_le_high);

  float normalizer = float(1.0 / max(sum_low, 1e-8));

  // normalize
#pragma unroll 2
  for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {
    probs_vec.fill(0);
    if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
      probs_vec.cast_load(probs + row_idx * d + i * BLOCK_THREADS * VEC_SIZE + tx * VEC_SIZE);
    }
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; ++j) {
      probs_vec[j] = (probs_vec[j] > low) ? probs_vec[j] * normalizer : 0;
    }
    if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
      probs_vec.cast_store(renormed_prob + row_idx * d + i * BLOCK_THREADS * VEC_SIZE + tx * VEC_SIZE);
    }
  }
}

}  // namespace compact_sampling
}  // namespace sglang
namespace sglang {
void compact_top_p_fallback(
    tvm::ffi::TensorView probs,
    tvm::ffi::TensorView out,
    tvm::ffi::TensorView top_p,
    tvm::ffi::TensorView counts,
    int64_t capacity) {
  using namespace host;
  using namespace compact_sampling;
  auto rows = SymbolicSize{"rows"};
  auto vocab = SymbolicSize{"vocab"};
  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA, kDLROCM>();
  TensorMatcher({rows, vocab}).with_dtype<float>().with_device(device).verify(probs);
  TensorMatcher({rows, vocab}).with_dtype<float>().with_device(device).verify(out);
  TensorMatcher({rows}).with_dtype<float>().with_device(device).verify(top_p);
  TensorMatcher({rows}).with_dtype<int32_t>().with_device(device).verify(counts);
  constexpr uint32_t threads = 1024;
  constexpr uint32_t vec = 4;
  RuntimeCheck(vocab.unwrap() % vec == 0, "compact top-p requires a vector-aligned vocabulary");
  RuntimeCheck(
      rows.unwrap() * vocab.unwrap() <= std::numeric_limits<uint32_t>::max(),
      "compact top-p input exceeds 32-bit indexing");
  RuntimeCheck(capacity > 0 && capacity <= std::numeric_limits<int32_t>::max(), "invalid compact top-p capacity");
  if (rows.unwrap() == 0) return;
  LaunchKernel(rows.unwrap(), threads, device.unwrap(), sizeof(RenormTempStorage<threads, REDUCE_ALGO>))(
      TopPRenormProbKernel<threads, REDUCE_ALGO, vec, float>,
      static_cast<float*>(probs.data_ptr()),
      static_cast<float*>(out.data_ptr()),
      static_cast<float*>(top_p.data_ptr()),
      0.95f,
      static_cast<uint32_t>(vocab.unwrap()),
      static_cast<int*>(counts.data_ptr()),
      static_cast<int>(capacity));
}
}  // namespace sglang
