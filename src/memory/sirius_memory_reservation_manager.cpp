
/*
 * Copyright 2025, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "memory/sirius_memory_reservation_manager.hpp"

#include "cucascade/memory/common.hpp"
#include "log/logging.hpp"

#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/mr/managed_memory_resource.hpp>

#include <cuda_runtime_api.h>

#include <cucascade/memory/memory_reservation_manager.hpp>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <variant>

namespace sirius {
namespace memory {

namespace {

// Apply this at the manager boundary so explicit space.gpu configurations and
// programmatic callers cannot accidentally retain a device-memory factory.
auto managed_configs(std::vector<cucascade::memory::memory_space_config> configs)
{
  for (auto& config : configs) {
    auto* gpu = std::get_if<cucascade::memory::gpu_memory_space_config>(&config);
    if (!gpu) { continue; }
    // The accounting layer uses signed byte counters. Keep conversion to double
    // and back safely below INT64_MAX, including rounding at the upper bound.
    constexpr auto max_capacity = std::uint64_t{1} << 62;
    if (gpu->memory_capacity > max_capacity || !std::isfinite(gpu->reservation_limit_fraction) ||
        gpu->reservation_limit_fraction < 0 || gpu->reservation_limit_fraction > 1) {
      throw std::invalid_argument("UVM: invalid logical GPU memory budget");
    }
    gpu->mr_factory_fn = [](int device_id, std::size_t) {
      rmm::cuda_set_device_raii guard{rmm::cuda_device_id{device_id}};
      int managed    = 0;
      int concurrent = 0;
      if (cudaDeviceGetAttribute(&managed, cudaDevAttrManagedMemory, device_id) != cudaSuccess ||
          cudaDeviceGetAttribute(&concurrent, cudaDevAttrConcurrentManagedAccess, device_id) !=
            cudaSuccess ||
          !managed || !concurrent) {
        throw std::runtime_error(
          "UVM experiment requires managed memory and concurrent managed access on every GPU");
      }
      // No pool, prefetch, preferred-location advice, or device-memory fallback.
      return ::cuda::mr::any_resource<::cuda::mr::device_accessible>{
        rmm::mr::managed_memory_resource{}};
    };
  }
  return configs;
}

}  // namespace

sirius_memory_reservation_manager::sirius_memory_reservation_manager(
  const std::vector<cucascade::memory::memory_space_config>& configs)
  : cucascade::memory::memory_reservation_manager(managed_configs(configs))
{
  auto gpu_spaces = this->get_memory_spaces_for_tier(cucascade::memory::Tier::GPU);
  if (gpu_spaces.empty()) {
    throw std::runtime_error("At least one GPU memory space must be configured");
  }
  for (const auto* space : gpu_spaces) {
    SIRIUS_LOG_INFO(
      "UVM experiment: GPU {} uses cudaMallocManaged; logical reservation limit {} bytes; "
      "application GPU spilling disabled",
      space->get_device_id(),
      space->get_max_memory());
    auto const device_mr = space->get_default_allocator();
    rmm::cuda_set_device_raii set_device{rmm::cuda_device_id{space->get_device_id()}};
    // Capture the old resource by value (not by ref) — see comment in header for why.
    // Wrap device_async_resource_ref in any_resource to satisfy the non-deprecated API.
    prev_device_mrs_.push_back(cudf::set_current_device_resource(
      ::cuda::mr::any_resource<::cuda::mr::device_accessible>{device_mr}));
  }
}

sirius_memory_reservation_manager::~sirius_memory_reservation_manager()
{
  auto gpu_spaces = this->get_memory_spaces_for_tier(cucascade::memory::Tier::GPU);
  // Restore the previous cuDF device resources saved in the constructor.
  // Calling reset_current_device_resource_ref() would leave cuDF with a null/invalid
  // resource that crashes subsequent allocations in other tests or code paths.
  //
  // Drain GPU users before restoring resource refs and destroying their owner.
  for (std::size_t i = 0; i < gpu_spaces.size() && i < prev_device_mrs_.size(); ++i) {
    rmm::cuda_set_device_raii set_device{rmm::cuda_device_id{gpu_spaces[i]->get_device_id()}};
    cudaDeviceSynchronize();
    cudf::set_current_device_resource(std::move(prev_device_mrs_[i]));
  }
}

}  // namespace memory
}  // namespace sirius
