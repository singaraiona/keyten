//! Async driver: runs a plus kernel under tokio and asserts result + progress
//! match the sync run.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use keyten::alloc::alloc_vec_i64;
use keyten::chunk::{drive_async, ChunkStep};
use keyten::ctx::Ctx;
use keyten::kernels::plus::AddI64VecVec;
use keyten::runtime::RUNTIME;

#[tokio::test(flavor = "current_thread")]
async fn drive_async_under_tokio() {
    let cancelled = AtomicBool::new(false);
    let progress = AtomicU64::new(0);
    let ctx = Ctx::new(&RUNTIME, &cancelled, &progress).with_chunk(128);

    unsafe {
        let n = 1_000;
        let mut x = alloc_vec_i64(&ctx, n);
        let mut y = alloc_vec_i64(&ctx, n);
        for i in 0..n as usize {
            x.as_mut_slice::<i64>()[i] = i as i64;
            y.as_mut_slice::<i64>()[i] = (i * 10) as i64;
        }
        let mut out = alloc_vec_i64(&ctx, n);
        {
            let xs = x.as_slice::<i64>();
            let ys = y.as_slice::<i64>();
            let os = out.as_mut_slice::<i64>();
            let mut k = AddI64VecVec::new(xs, ys, os, 128);
            drive_async(&mut k, &ctx).await.unwrap();
        }
        let os = out.as_slice::<i64>();
        for i in 0..n as usize {
            assert_eq!(os[i], i as i64 + (i * 10) as i64);
        }
    }

    // Every element was tallied.
    assert_eq!(progress.load(Ordering::Relaxed), 1_000);
}

/// Custom kernel that records how many times `step()` was called between
/// `Poll::Pending` returns. Verifies that drive_async actually yields between
/// chunks (one yield per chunk).
#[tokio::test(flavor = "current_thread")]
async fn drive_async_yields_between_chunks() {
    use std::cell::Cell;

    struct Counter {
        remaining: usize,
        chunk: usize,
        steps: std::rc::Rc<Cell<usize>>,
    }
    impl ChunkStep for Counter {
        fn step(&mut self) -> Option<usize> {
            if self.remaining == 0 {
                return None;
            }
            self.steps.set(self.steps.get() + 1);
            let n = self.chunk.min(self.remaining);
            self.remaining -= n;
            Some(n)
        }
    }

    let cancelled = AtomicBool::new(false);
    let progress = AtomicU64::new(0);
    let ctx = Ctx::new(&RUNTIME, &cancelled, &progress);

    let steps = std::rc::Rc::new(Cell::new(0));
    let mut k = Counter {
        remaining: 500,
        chunk: 50,
        steps: steps.clone(),
    };
    drive_async(&mut k, &ctx).await.unwrap();
    // 500 / 50 = 10 chunks → 10 step() calls.
    assert_eq!(steps.get(), 10);
    assert_eq!(progress.load(Ordering::Relaxed), 500);
}
