// The peer addresses of a workspace registered with AITER's custom all-reduce (tp4_peer_workspace.py), for the
// Muse Spark TP4 row-exchange kernels, which take them by value.
#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <dlpack/dlpack.h>
#include <hip/hip_runtime.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>
#include <custom_all_reduce.cuh>  // AITER csrc/include: CustomAllreduce, RankData

namespace sglang {
namespace muse_tp4_peer_workspace {

struct PeerWorkspace {
  // Every rank's address of the registered buffer `workspace`, into `peers` (CPU int64 [world size]).
  static void addresses(int64_t comm, int64_t workspace, const tvm::ffi::TensorView peers) {
    using namespace host;
    auto* communicator = reinterpret_cast<aiter::CustomAllreduce*>(comm);
    const int world = communicator->world_size_;
    TensorMatcher({world}).with_dtype<int64_t>().with_device<kDLCPU>().verify(peers);
    // The buffer is registered, so no stream is consulted.
    const aiter::RankData* registered = communicator->get_buffer_RD(nullptr, reinterpret_cast<void*>(workspace));
    aiter::RankData copy;
    RuntimeDeviceCheck(hipMemcpy(&copy, registered, sizeof(copy), hipMemcpyDeviceToHost));
    for (int r = 0; r < world; ++r)
      static_cast<int64_t*>(peers.data_ptr())[r] = reinterpret_cast<int64_t>(copy.ptrs[r]);
  }
};

}  // namespace muse_tp4_peer_workspace
}  // namespace sglang
