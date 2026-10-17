# UVM allocation experiment

Branch: `experiment/uvm-alloc`.

## Behavior

Every GPU memory space constructed by `sirius_memory_reservation_manager` uses
RMM's `managed_memory_resource`, which calls `cudaMallocManaged` / `cudaFree`.
The policy replaces caller-provided GPU factories too. cuDF's per-device default
resource is the same reservation-aware allocator, so explicit RMM resources,
implicit RMM defaults, cuDF outputs and RMM-backed scratch all use managed memory.
The direct allocation in Simpatico's `count_touched_chunks` also uses
`cudaMallocManaged` directly (this four-byte diagnostic counter remains outside
reservation accounting). There is no managed pool, prefetch, memory advice, or fallback
to a device allocator in this implementation. Allocation/free overhead is part of
the experiment; compare it separately from migration costs when interpreting timings.

cuCascade still supplies memory spaces, reservations and accounting. Its counters
are logical allocated/reserved bytes, **not physical VRAM residency**. GPU spill
monitoring is disabled and both public GPU spill request APIs return an exceptional
future. Host-cache-to-disk behavior is unchanged. A failed pipeline reservation
fails the query immediately rather than waiting indefinitely for forbidden GPU
spilling; this can also occur under concurrent reservations even when CUDA itself
could allocate more memory. Actual allocation limits remain enforced. Existing
allocation-OOM retry logic remains bounded by the engine's retry limit.

Managed memory and concurrent managed access are required on every configured
GPU; unsupported hardware fails initialization. CPU fallback defaults to false on
this branch. Do not enable it for a GPU performance measurement.

## Logical budget and oversubscription

The default logical GPU capacity remains 95% of VRAM for reproducibility. To test
oversubscription, explicitly set a finite byte budget larger than VRAM and below
the memory you intend to make available to the experiment. This is an accounting
limit, not an upfront managed allocation and not a reservation of physical RAM.
The following is an **example**, not a machine-independent budget recommendation:

```yaml
sirius:
  topology:
    num_gpus: 1
  memory:
    gpu:
      usage_limit_bytes: 64Gi
      reservation_limit_fraction: 1.0
```

Place this in the normal `~/.sirius/sirius.yaml` configuration. Both the high-level
byte budget and low-level `space.gpu[].memory_capacity` can exceed VRAM. Fraction
budgets still refer to physical VRAM. Logical GPU capacities are limited to 2^62
bytes so signed accounting and floating-point threshold conversion remain valid.
Setting a large budget alone does not force oversubscription: the query's live
managed working set must actually exceed available VRAM. CPU caches, pinned I/O
buffers and managed pages compete for host RAM, so choose the budget for the test
machine and workload.

## Scope and exceptions

This is a GPU **data and scratch allocation** experiment, not a process-wide
CUDA allocator interposer:

- Ordinary CPU allocations and pinned host I/O/caching resources remain unchanged.
- cuCascade's physical P2P capability probe (`cucascade/src/memory/common.cpp`)
  retains its small raw device allocations. Managed pointers would invalidate
  the test of physical peer DMA. Default device factories in cuCascade remain
  available to other consumers but are overridden at the Sirius manager boundary.
- External libraries (cuVS/RAFT, nvcomp, CUDA, etc.) may allocate internal buffers
  outside RMM. The source audit cannot guarantee these are managed. A runtime
  allocation trace is needed to determine that on the target build.
- Standalone Simpatico CLI/tests that construct their own resources, other DuckDB
  extensions, and allocations before/after the Sirius manager's lifetime are not
  covered. Previous cuDF defaults are restored at manager destruction.

These exceptions must not be hidden when reporting results. A literal guarantee
that *every allocation in the process* calls `cudaMallocManaged` is not provided.

## Validation on a Linux NVIDIA machine

This patch was authored on macOS without Pixi, CUDA headers/compiler or a GPU;
no compilation, GPU correctness or performance result is claimed from that host.

```bash
pixi run make
pixi run build/release/test/cpp/sirius_unittest '[uvm]'
pixi run build/release/test/cpp/sirius_unittest '[context_config]'
```

The UVM tests check pointer attributes for explicit RMM, default RMM, cuDF data
and null masks on each visible GPU; GPU-written bytes read directly from the
host after synchronization; retained logical limits; replacement of a caller's
allocator factory; immediate rejection of GPU spill requests; and executor failures
for oversized or contended logical reservations without executing a task. Config
tests check byte budgets larger than VRAM. These tests do not allocate a VRAM-sized
working set and cannot prove oversubscription performance.

Tests that explicitly expect application GPU spilling or CPU fallback by default
exercise a different contract on this branch; the full upstream suite has not
been validated against this intentional change. Do not interpret the targeted
checks as a claim that the entire suite passes.

For query validation, explicitly execute:

```sql
SET enable_duckdb_fallback = false;
```

Run an existing supported workload and compare its complete results with a CPU
baseline (`SET gpu_execution = false`) in a separate run. Profile the GPU run with
CUDA allocation tracing (for example Nsight Systems `--trace=cuda`) and check
`cudaMallocManaged`, remaining `cudaMalloc`/`cudaMallocAsync` calls and their call
stacks. Distinguish P2P initialization from query allocations. Check migration /
GPU fault counters on the target platform to establish that oversubscription was
actually exercised. Keep initialization, cold runs and warm runs separate and
record the logical budget, GPU model, driver, CUDA/RMM versions, host RAM, input
size and concurrent workload.
