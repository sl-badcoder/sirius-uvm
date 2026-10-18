/*
 * Copyright 2026, Sirius Contributors.
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

#include "catch.hpp"
#include "downgrade/downgrade_executor.hpp"
#include "memory/managed_pool_memory_resource.hpp"
#include "memory/sirius_memory_reservation_manager.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/managed_memory_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>

#include <cuda_runtime_api.h>

#include <cucascade/memory/config.hpp>
#include <cucascade/memory/reservation_aware_resource_adaptor.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <future>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {

constexpr std::size_t budget = 1ULL << 20;

cucascade::memory::gpu_memory_space_config gpu_config(int device_id)
{
  cucascade::memory::gpu_memory_space_config config;
  config.device_id                  = device_id;
  config.memory_capacity            = budget;
  config.reservation_limit_fraction = 1.0;
  config.per_stream_reservation     = false;
  // An explicit caller factory must not bypass the UVM policy.
  config.mr_factory_fn = [](int,
                            std::size_t) -> cuda::mr::any_resource<cuda::mr::device_accessible> {
    throw std::logic_error("device allocator factory must not be invoked");
  };
  return config;
}

void require_managed(void const* ptr)
{
  REQUIRE(ptr != nullptr);
  cudaPointerAttributes attributes{};
  REQUIRE(cudaPointerGetAttributes(&attributes, ptr) == cudaSuccess);
  CHECK(attributes.type == cudaMemoryTypeManaged);
}

}  // namespace

TEST_CASE("Sirius GPU buffers and cuDF defaults use managed allocations", "[uvm]")
{
  int count = 0;
  REQUIRE(cudaGetDeviceCount(&count) == cudaSuccess);
  REQUIRE(count > 0);
  // Exercise every visible device without requiring a multi-GPU machine.
  for (int device = 0; device < count; ++device) {
    CAPTURE(device);
    rmm::cuda_set_device_raii guard{rmm::cuda_device_id{device}};
    sirius::memory::sirius_memory_reservation_manager manager({gpu_config(device)});
    auto* space = manager.get_memory_space(cucascade::memory::Tier::GPU, device);
    REQUIRE(space != nullptr);
    auto* adaptor =
      space->get_memory_resource_as<cucascade::memory::reservation_aware_resource_adaptor>();
    REQUIRE(adaptor != nullptr);
    auto upstream = adaptor->get_upstream_resource();
    auto* managed_pool =
      cuda::mr::resource_cast<sirius::memory::managed_pool_memory_resource>(&upstream);
    REQUIRE(managed_pool != nullptr);
    auto const* pool = &managed_pool->pool();
    auto backing     = pool->get_upstream_resource();
    REQUIRE(cuda::mr::resource_cast<rmm::mr::managed_memory_resource>(&backing) != nullptr);
    CHECK(pool->pool_size() == 0);
    rmm::cuda_stream stream;
    {
      rmm::device_buffer explicit_buffer(4096, stream, space->get_default_allocator());
      rmm::device_buffer default_buffer(4096, stream);
      auto column = cudf::make_numeric_column(
        cudf::data_type{cudf::type_id::INT32}, 1024, cudf::mask_state::ALL_VALID, stream);
      require_managed(explicit_buffer.data());
      require_managed(default_buffer.data());
      require_managed(column->view().data<std::int32_t>());
      require_managed(column->view().null_mask());
      REQUIRE(cudaMemsetAsync(explicit_buffer.data(), 0x5a, 4096, stream.value()) == cudaSuccess);
      stream.synchronize();
      // Host access after GPU work must observe the same allocation's contents.
      auto* bytes = static_cast<unsigned char*>(explicit_buffer.data());
      for (std::size_t i = 0; i < 4096; ++i) {
        REQUIRE(bytes[i] == 0x5a);
      }
    }
    stream.synchronize();
    auto const backing_size = pool->pool_size();
    REQUIRE(backing_size >= 4096);
    CHECK(backing_size <= budget);
    // A freed block can be reused without growing backing storage. Also test a
    // different stream: the RMM pool must preserve ordering across the handoff.
    rmm::cuda_stream other_stream;
    {
      rmm::device_buffer reused(4096, other_stream, space->get_default_allocator());
      require_managed(reused.data());
      REQUIRE(cudaMemsetAsync(reused.data(), 0x3c, 4096, other_stream.value()) == cudaSuccess);
      other_stream.synchronize();
      auto const* bytes = static_cast<unsigned char const*>(reused.data());
      for (std::size_t i = 0; i < 4096; ++i) {
        REQUIRE(bytes[i] == 0x3c);
      }
      CHECK(pool->pool_size() == backing_size);
    }
    other_stream.synchronize();
    // A real logical cap remains in force; UVM is not an unlimited fake budget.
    REQUIRE_THROWS(rmm::device_buffer(budget + 1, stream, space->get_default_allocator()));
    REQUIRE(space->make_reservation_or_null(budget) != nullptr);
    REQUIRE(space->make_reservation_or_null(budget + 1) == nullptr);
  }
}

TEST_CASE("Managed pool rounds its backing limit down to allocation alignment", "[uvm]")
{
  auto config            = gpu_config(0);
  config.memory_capacity = budget + 1;
  sirius::memory::sirius_memory_reservation_manager manager({config});
  auto* space = manager.get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space != nullptr);
  rmm::cuda_stream stream;
  rmm::device_buffer full(budget, stream, space->get_default_allocator());
  require_managed(full.data());
  stream.synchronize();
  // The logical budget includes the one-byte tail, but it cannot create another
  // aligned block or bypass the managed pool's finite backing limit.
  REQUIRE_THROWS(rmm::device_buffer(1, stream, space->get_default_allocator()));
}

TEST_CASE("Managed pool orders reused backing across streams", "[uvm]")
{
  sirius::memory::sirius_memory_reservation_manager manager({gpu_config(0)});
  auto* space = manager.get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space != nullptr);
  rmm::cuda_stream producer;
  rmm::cuda_stream consumer;
  void* address = nullptr;
  {
    // Fill the entire pool, so the next allocation must reuse this backing.
    rmm::device_buffer first(budget, producer, space->get_default_allocator());
    address = first.data();
    REQUIRE(cudaMemsetAsync(first.data(), 0x6d, budget, producer.value()) == cudaSuccess);
    // Deliberately do not synchronize the producer before returning its block.
  }
  rmm::device_buffer reused(budget, consumer, space->get_default_allocator());
  REQUIRE(reused.data() == address);
  require_managed(reused.data());
  // The pool must order the producer's writes before this consumer's read.
  // Host destination remains alive until the consumer completes.
  std::vector<unsigned char> bytes(budget);
  REQUIRE(cudaMemcpyAsync(
            bytes.data(), reused.data(), budget, cudaMemcpyDeviceToHost, consumer.value()) ==
          cudaSuccess);
  consumer.synchronize();
  REQUIRE(std::all_of(bytes.begin(), bytes.end(), [](auto byte) { return byte == 0x6d; }));
}

TEST_CASE("Managed pools tear down on their owning GPUs", "[uvm][multi_gpu]")
{
  int count = 0;
  REQUIRE(cudaGetDeviceCount(&count) == cudaSuccess);
  if (count < 2) {
    WARN("Managed pool multi-GPU teardown requires two visible GPUs");
    return;
  }
  rmm::cuda_set_device_raii caller_device{rmm::cuda_device_id{1}};
  auto manager = std::make_unique<sirius::memory::sirius_memory_reservation_manager>(
    std::vector<cucascade::memory::memory_space_config>{gpu_config(0), gpu_config(1)});
  for (int device = 0; device < 2; ++device) {
    rmm::cuda_set_device_raii guard{rmm::cuda_device_id{device}};
    auto* space = manager->get_memory_space(cucascade::memory::Tier::GPU, device);
    REQUIRE(space != nullptr);
    rmm::cuda_stream stream;
    {
      rmm::device_buffer buffer(4096, stream, space->get_default_allocator());
      require_managed(buffer.data());
      REQUIRE(cudaMemsetAsync(buffer.data(), 0, 4096, stream.value()) == cudaSuccess);
      stream.synchronize();
    }
    stream.synchronize();
  }
  // Both pools retain backing. Device 1 is current when device 0's pool is
  // destroyed; the owning adapter must switch devices and restore the caller.
  manager.reset();
  int current = -1;
  REQUIRE(cudaGetDevice(&current) == cudaSuccess);
  CHECK(current == 1);
  CHECK(cudaGetLastError() == cudaSuccess);
}

TEST_CASE("UVM rejects GPU application spill requests without starting workers", "[uvm]")
{
  sirius::memory::sirius_memory_reservation_manager manager({gpu_config(0)});
  auto* space = manager.get_memory_space(cucascade::memory::Tier::GPU, 0);
  sirius::exec::query_lifecycle_registry lifecycle;
  sirius::data::data_repository_manager_registry repositories;
  sirius::parallel::downgrade_executor executor(
    lifecycle, {}, repositories, space->get_id(), space, manager);
  auto bytes = executor.request_free_memory(4096);
  REQUIRE(bytes.wait_for(std::chrono::milliseconds{0}) == std::future_status::ready);
  REQUIRE_THROWS_WITH(bytes.get(), Catch::Matchers::ContainsSubstring("spilling is disabled"));
  bool predicate_called = false;
  auto predicate        = executor.request_downgrade([&] {
    predicate_called = true;
    return true;
  });
  REQUIRE(predicate.wait_for(std::chrono::milliseconds{0}) == std::future_status::ready);
  REQUIRE_THROWS_WITH(predicate.get(), Catch::Matchers::ContainsSubstring("spilling is disabled"));
  CHECK_FALSE(predicate_called);
}
