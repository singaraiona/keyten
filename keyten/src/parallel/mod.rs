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

use core::ops::Range;
use core::sync::atomic::{AtomicBool, Ordering};

use crate::ctx::{Ctx, KernelErr};

pub mod partition;

/// Element-count threshold below which the parallel path falls through to
/// the sequential implementation. Below this, partition + thread::scope
/// overhead exceeds the gain from going wide. The value is conservative —
/// at ~1-3 GB/s arithmetic throughput, 256 K i64s is ~0.7-2 ms of work,
/// which comfortably covers thread::scope's ~10-50 µs spawn-and-join cost.
pub const PARALLEL_THRESHOLD: usize = 256 * 1024;

/// Run `body` over disjoint sub-slices of `output` in parallel.
///
/// `output` is split into `ctx.runtime.worker_count()` contiguous sub-slices
/// via [`partition::balanced`] and each worker is spawned with its own
/// `(Range<usize>, &mut [T])` pair. Workers run on OS threads under
/// [`std::thread::scope`], which joins all workers before this function
/// returns — so borrowed references in `body`'s captures are safe.
///
/// The caller is responsible for gating on
/// [`crate::runtime::Runtime::parallel_enabled`] and
/// [`PARALLEL_THRESHOLD`]; this function assumes the decision to go wide
/// has already been made. When the resolved worker count is 1 (e.g. on a
/// single-CPU machine, or `n` rounds down to one worker), it short-circuits
/// to a single in-place call.
///
/// Each spawned worker should call [`crate::chunk::drive_sync`] inside
/// `body`. That observes `ctx.cancelled` and `ctx.runtime.global_cancel`
/// at every chunk boundary, so cancellation propagates across workers
/// within one chunk-time.
pub fn parallel_for_each_mut<T, F>(
    output: &mut [T],
    ctx: &Ctx,
    body: F,
) -> Result<(), KernelErr>
where
    T: Send,
    F: Fn(Range<usize>, &mut [T]) -> Result<(), KernelErr> + Send + Sync,
{
    let n = output.len();
    if n == 0 {
        return Ok(());
    }

    let nw = ctx.runtime.worker_count().min(n).max(1);
    if nw == 1 {
        return body(0..n, output);
    }

    let ranges = partition::balanced(n, nw);

    // Split `output` into per-worker mutable pieces matching the ranges.
    // After the loop, the pieces are non-overlapping and cover the whole
    // slice — exactly what `partition::balanced` guarantees.
    let mut chunks: Vec<&mut [T]> = Vec::with_capacity(nw);
    let mut rem: &mut [T] = output;
    for r in &ranges {
        let len = r.end - r.start;
        let (front, back) = rem.split_at_mut(len);
        chunks.push(front);
        rem = back;
    }
    debug_assert!(rem.is_empty());

    let any_err = AtomicBool::new(false);
    let body_ref = &body;
    let err_ref = &any_err;
    std::thread::scope(|s| {
        for (range, chunk) in ranges.into_iter().zip(chunks) {
            s.spawn(move || {
                if body_ref(range, chunk).is_err() {
                    err_ref.store(true, Ordering::Relaxed);
                }
            });
        }
    });

    if any_err.load(Ordering::Relaxed) {
        Err(KernelErr::Cancelled)
    } else {
        Ok(())
    }
}
