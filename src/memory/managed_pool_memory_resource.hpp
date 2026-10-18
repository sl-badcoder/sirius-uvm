// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <rmm/mr/managed_memory_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>

#include <cuda/memory_resource>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdio>
#include <memory>

namespace sirius::memory {

// Ownership adapter only: all allocation and stream-ordering logic is RMM's.
// cuCascade may destroy spaces while a different device is current. The last
// owning copy must release the pool (cached blocks and events) on its device.
class managed_pool_memory_resource {
 public:
  managed_pool_memory_resource(int device_id, std::size_t maximum)
    : pool_(
        new rmm::mr::pool_memory_resource{::cuda::mr::any_resource<::cuda::mr::device_accessible>{
                                            rmm::mr::managed_memory_resource{}},
                                          0,
                                          maximum},
        [device_id](rmm::mr::pool_memory_resource* pool) noexcept {
          int previous = -1;
          auto status  = cudaGetDevice(&previous);
          if (status == cudaSuccess && previous != device_id) { status = cudaSetDevice(device_id); }
          if (status != cudaSuccess) {
            // Destruction cannot throw. Do not free device-bound pool state on
            // an unknown device after a CUDA failure; retain it until exit and
            // report the failure rather than silently freeing it incorrectly.
            std::fprintf(stderr,
                         "UVM: retaining GPU %d pool after teardown device-selection "
                         "failure: %s\n",
                         device_id,
                         cudaGetErrorString(status));
            return;
          }
          delete pool;
          if (previous != device_id) {
            status = cudaSetDevice(previous);
            if (status != cudaSuccess) {
              std::fprintf(stderr,
                           "UVM: failed to restore GPU %d after pool teardown: %s\n",
                           previous,
                           cudaGetErrorString(status));
            }
          }
        })
  {
  }

  void* allocate(::cuda::stream_ref stream,
                 std::size_t bytes,
                 std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
  {
    return pool_->allocate(stream, bytes, alignment);
  }

  void deallocate(::cuda::stream_ref stream,
                  void* ptr,
                  std::size_t bytes,
                  std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
  {
    pool_->deallocate(stream, ptr, bytes, alignment);
  }

  void* allocate_sync(std::size_t bytes, std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
  {
    return pool_->allocate_sync(bytes, alignment);
  }

  void deallocate_sync(void* ptr,
                       std::size_t bytes,
                       std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
  {
    pool_->deallocate_sync(ptr, bytes, alignment);
  }

  rmm::mr::pool_memory_resource const& pool() const noexcept { return *pool_; }

  bool operator==(managed_pool_memory_resource const& other) const noexcept
  {
    return pool_ == other.pool_;
  }

  friend void get_property(managed_pool_memory_resource const&,
                           ::cuda::mr::device_accessible) noexcept
  {
  }

 private:
  std::shared_ptr<rmm::mr::pool_memory_resource> pool_;
};

static_assert(
  ::cuda::mr::resource_with<managed_pool_memory_resource, ::cuda::mr::device_accessible>);

}  // namespace sirius::memory
