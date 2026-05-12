//! Parallel reductions (folds with a combiner).
//!
//! Unlike `parallel_for_each_mut`, reductions don't write to a shared output
//! slice — each worker accumulates a partial scalar from its sub-range, and
//! the partials are combined sequentially after all workers finish.
//!
//! **Floating-point note:** the parallel sum order differs from the
//! sequential one (associative grouping changes), so f64 reductions may
//! produce last-ULP-different results. Use `Runtime.deterministic`
//! (forthcoming) to force sequential reduction when bit-exact behavior
//! matters.

use core::ops::Range;
use core::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use crate::chunk::{drive_sync, ChunkStep};
use crate::ctx::{Ctx, KernelErr};

use super::partition;

/// Run a parallel fold over `0..n`.
///
/// Each worker:
///   1. Calls `make_step(range)` to build a fresh `ChunkStep` over its
///      sub-range.
///   2. Drives the step to completion via [`drive_sync`].
///   3. Calls `extract(&step)` to pull a partial accumulator out.
///
/// After all workers complete, the partials are folded into `init` with
/// `combine` on the calling thread.
///
/// `combine` is called sequentially over partials in arbitrary order
/// (whichever worker finished first lands first). For commutative
/// reductions (`+`, `min`, `max`, `xor`) the order doesn't matter; for
/// non-commutative reductions you should not parallelise this way.
///
/// As with [`super::parallel_for_each_mut`], this assumes the caller has
/// already decided to go parallel; below `worker_count() == 1`, it falls
/// through to a single sequential drive.
pub fn parallel_reduce<MakeStep, Step, T, Combine>(
    n: usize,
    ctx: &Ctx,
    init: T,
    make_step: MakeStep,
    extract: impl Fn(&Step) -> T + Sync,
    combine: Combine,
) -> Result<T, KernelErr>
where
    MakeStep: Fn(Range<usize>) -> Step + Sync,
    Step: ChunkStep + Send,
    T: Send + Copy,
    Combine: Fn(T, T) -> T,
{
    if n == 0 {
        return Ok(init);
    }

    let nw = ctx.runtime.worker_count().min(n).max(1);
    if nw == 1 {
        let mut step = make_step(0..n);
        drive_sync(&mut step, ctx)?;
        return Ok(combine(init, extract(&step)));
    }

    let ranges = partition::balanced(n, nw);
    let partials: Mutex<Vec<T>> = Mutex::new(Vec::with_capacity(nw));
    let any_err = AtomicBool::new(false);

    let make_step_ref = &make_step;
    let extract_ref = &extract;
    let partials_ref = &partials;
    let err_ref = &any_err;
    std::thread::scope(|s| {
        for range in ranges {
            s.spawn(move || {
                let mut step = make_step_ref(range);
                if drive_sync(&mut step, ctx).is_err() {
                    err_ref.store(true, Ordering::Relaxed);
                    return;
                }
                let partial = extract_ref(&step);
                // Mutex contention is negligible — one push per worker.
                partials_ref.lock().unwrap().push(partial);
            });
        }
    });

    if any_err.load(Ordering::Relaxed) {
        return Err(KernelErr::Cancelled);
    }

    let partials = partials.into_inner().unwrap();
    let mut acc = init;
    for p in partials {
        acc = combine(acc, p);
    }
    Ok(acc)
}
