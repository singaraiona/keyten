//! Chunked streaming kernel architecture.
//!
//! A `ChunkStep` processes one chunk of input per `step()` call. A driver
//! consumes it: `drive_sync` for synchronous use (zero async overhead);
//! `drive_async` for async use (one `YieldNow.await` between chunks).
//!
//! Cancellation and progress are observed at every chunk boundary.

use core::sync::atomic::Ordering;

use crate::ctx::{Ctx, KernelErr};
use crate::yield_now::YieldNow;

/// A streaming kernel that processes one chunk at a time.
///
/// `step` returns the number of elements processed in this chunk, or `None`
/// when the kernel is exhausted.
pub trait ChunkStep {
    fn step(&mut self) -> Option<usize>;
}

/// Drive a kernel to completion synchronously.
///
/// Per chunk: relaxed-load the cancellation flag (task-local + process-wide),
/// fetch-add into the progress counter. No async machinery.
#[inline]
pub fn drive_sync<K: ChunkStep>(k: &mut K, ctx: &Ctx) -> Result<(), KernelErr> {
    while let Some(n) = k.step() {
        ctx.progress.fetch_add(n as u64, Ordering::Relaxed);
        if ctx.cancelled.load(Ordering::Relaxed)
            || ctx.runtime.global_cancel.load(Ordering::Relaxed)
        {
            return Err(KernelErr::Cancelled);
        }
    }
    Ok(())
}

/// Drive a kernel to completion asynchronously.
///
/// Per chunk: same observation as the sync driver, plus one `YieldNow.await`.
/// The yield is executor-agnostic — anywhere a `Future` is polled correctly,
/// this works.
pub async fn drive_async<K: ChunkStep>(k: &mut K, ctx: &Ctx<'_>) -> Result<(), KernelErr> {
    while let Some(n) = k.step() {
        ctx.progress.fetch_add(n as u64, Ordering::Relaxed);
        if ctx.cancelled.load(Ordering::Relaxed)
            || ctx.runtime.global_cancel.load(Ordering::Relaxed)
        {
            return Err(KernelErr::Cancelled);
        }
        YieldNow::new().await;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ctx::Ctx;
    use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};

    struct CountUp {
        remaining: usize,
        chunk: usize,
    }

    impl ChunkStep for CountUp {
        fn step(&mut self) -> Option<usize> {
            if self.remaining == 0 {
                return None;
            }
            let n = self.chunk.min(self.remaining);
            self.remaining -= n;
            Some(n)
        }
    }

    #[test]
    fn drive_sync_progresses_to_completion() {
        let cancelled = AtomicBool::new(false);
        let progress = AtomicU64::new(0);
        let ctx = Ctx::new(&crate::runtime::RUNTIME, &cancelled, &progress);
        let mut k = CountUp { remaining: 1_000, chunk: 64 };
        drive_sync(&mut k, &ctx).unwrap();
        assert_eq!(progress.load(Ordering::Relaxed), 1_000);
    }

    #[test]
    fn drive_sync_observes_cancellation() {
        let cancelled = AtomicBool::new(false);
        let progress = AtomicU64::new(0);
        let ctx = Ctx::new(&crate::runtime::RUNTIME, &cancelled, &progress);
        // Pre-cancel; first chunk progresses then loop exits with Err.
        cancelled.store(true, Ordering::Relaxed);
        let mut k = CountUp { remaining: 1_000, chunk: 64 };
        let r = drive_sync(&mut k, &ctx);
        assert_eq!(r, Err(KernelErr::Cancelled));
        // At least one chunk's work was tallied before the loop noticed.
        assert!(progress.load(Ordering::Relaxed) > 0);
        assert!(progress.load(Ordering::Relaxed) < 1_000);
    }
}
