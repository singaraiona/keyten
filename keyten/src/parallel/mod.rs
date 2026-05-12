//! Data-parallel kernel execution.
//!
//! Stage 2 of v2: kernels can partition their input range across worker
//! threads, each running an existing single-threaded `ChunkStep` over a
//! sub-range. The `ChunkStep` trait is unchanged from v1 — it happens to
//! already be the right parallelism boundary (per-chunk state with `off`,
//! borrowed slices, no shared mutable state).
//!
//! Activation is gated on `Runtime.parallel_enabled()`, controlled at the
//! REPL via `\\set parallel 1`. The sequential path is byte-identical to
//! v1 when the flag is off.

pub mod partition;

/// Element-count threshold below which the parallel path falls through to
/// the sequential implementation. Below this, partition + thread::scope
/// overhead exceeds the gain from going wide. The value is conservative —
/// at ~1-3 GB/s arithmetic throughput, 256 K i64s is ~0.7-2 ms of work,
/// which comfortably covers thread::scope's ~10-50 µs spawn-and-join cost.
pub const PARALLEL_THRESHOLD: usize = 256 * 1024;
