# UVM allocation experiment

Branch: `experiment/uvm-managed-pool` (based on `experiment/uvm-alloc`).

The original direct-allocation experiment remains on `experiment/uvm-alloc`.

## Behavior

Every GPU memory space constructed by `sirius_memory_reservation_manager` uses
RMM's `pool_memory_resource` with an owning `managed_memory_resource` upstream.
Backing blocks use `cudaMallocManaged` / `cudaFree`; individual buffer requests
suballocate and reuse those blocks with RMM's stream-ordering machinery.
A small ownership adapter selects the owning CUDA device when the last pool
reference is destroyed; it delegates all allocation operations directly to RMM.
If CUDA device selection fails during teardown, it reports the error and retains
the pool until process exit rather than freeing device-bound state on an unknown
device.
The policy replaces caller-provided GPU factories too. cuDF's per-device default
resource is the same reservation-aware allocator, so explicit RMM resources,
implicit RMM defaults, cuDF outputs and RMM-backed scratch all use managed memory.
The direct allocation in Simpatico's `count_touched_chunks` also uses
`cudaMallocManaged` directly (this four-byte diagnostic counter remains outside
reservation accounting). There is no prefetch, memory advice, or fallback to a
conventional device allocator. Pinned host staging and explicit input copies
remain in place: this changes GPU backing memory and eviction, not the scan engine.

The pool starts with zero backing bytes and has a maximum equal to the logical
GPU capacity rounded down to 256-byte alignment. Smaller budgets are rejected.
RMM's capped growth policy can request half the remaining maximum on first growth
(e.g. approximately 32 GB with a 64 GB cap). Pool backing size, live suballocations,
reservations, and resident VRAM are different quantities. Freed suballocations
remain cached in managed backing until the pool releases them; this changes the
allocation lifetime and can affect UVM eviction. Do not equate a large backing
allocation with an equally large live working set or oversubscription.

Both normal and this branch now reuse allocations, but they use different pool
implementations: normal uses CUDA's async device pool; this branch uses RMM's
managed-backed pool. This is a system comparison, not an isolated allocator or
kernel-policy microbenchmark. Compare stock and modified UVM using the exact same
managed-pool executable, settings, and physical VRAM allowance.

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
the memory you intend to make available to the experiment. This bounds logical
accounting and managed pool backing. It does not allocate
the full budget upfront or reserve that amount of physical RAM/VRAM.
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
# With two visible GPUs, also exercise the [uvm][multi_gpu] teardown case.
```

The UVM tests check that GPU spaces contain an RMM pool with a managed upstream,
start empty, reuse freed storage across streams without growth, and preserve
aligned backing limits. They also check pointer attributes for explicit RMM,
default RMM, cuDF data
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

## Build and run the pooled experiment

On the Linux GPU machine, after transferring these changes and initializing submodules:

```bash
cd /home/lemes/lemes_guided/thesis/benchmark/sirius-uvm
git switch experiment/uvm-managed-pool
pixi run make clean
pixi run make
pixi run build/release/test/cpp/sirius_unittest '[uvm]'
pixi run build/release/test/cpp/sirius_unittest '[context_config]'
```

Use the existing UVM YAML (for example the 64 GB logical budget). The startup log
must say `uses an RMM pool backed by cudaMallocManaged`. Use the built DuckDB CLI
and matching extension through `test/tpch_performance/run_tpch_parquet.sh`, with
`enable_duckdb_fallback = false` configured in its DuckDB arguments. Use the same
external VRAM holder as normal; the YAML alone does not cap physical residency.

```bash
export CUDA_VISIBLE_DEVICES=0
unset SIRIUS_EXTENSION_PATH
EXPERIMENT=/home/lemes/lemes_guided/thesis/uvm-experiment
SIRIUS_CONFIG_FILE="$EXPERIMENT/configs/uvm.yaml" \
OUTPUT_DIR="$EXPERIMENT/results/uvm-pool-sf8-$(date +%Y%m%d-%H%M%S)" \
pixi run bash test/tpch_performance/run_tpch_parquet.sh \
  --parquet-dir /home/lemes/tpch/sf8 --iterations 6 --timeout 3600 sirius 8 1 9 18
```

This command does not start the VRAM holder. An existing `run_uvm.sh --build`
launcher that explicitly switches to `experiment/uvm-alloc` builds the old direct
variant: use the manual steps above for this new branch. For stock versus modified
UVM testing, rebuild/reload the driver module as needed but keep this application
binary unchanged. Verify the loaded module identity and compare validated results.
