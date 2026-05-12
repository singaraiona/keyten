//! Cancellation is observed within one chunk's worth of work.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use keyten::alloc::alloc_vec_i64;
use keyten::ctx::{Ctx, KernelErr};
use keyten::op::dispatch_plus;
use keyten::runtime::RUNTIME;

#[test]
fn cancellation_observed_promptly() {
    let cancelled = AtomicBool::new(true); // pre-cancelled
    let progress = AtomicU64::new(0);
    let ctx = Ctx::new(&RUNTIME, &cancelled, &progress).with_chunk(64);

    unsafe {
        let n = 10_000;
        let mut x = alloc_vec_i64(&ctx, n);
        for (i, slot) in x.as_mut_slice::<i64>().iter_mut().enumerate() {
            *slot = i as i64;
        }
        let mut y = alloc_vec_i64(&ctx, n);
        for (i, slot) in y.as_mut_slice::<i64>().iter_mut().enumerate() {
            *slot = i as i64;
        }
        let r = dispatch_plus(x, y, &ctx);
        assert_eq!(r.err(), Some(KernelErr::Cancelled));
    }

    let p = progress.load(Ordering::Relaxed);
    // At least one chunk did some work before the loop noticed cancellation.
    assert!(p > 0, "expected some progress before cancellation, got {p}");
    // But not all of it.
    assert!(p < 10_000, "expected partial progress, got {p}");
}

#[test]
fn no_cancellation_runs_to_completion() {
    let cancelled = AtomicBool::new(false);
    let progress = AtomicU64::new(0);
    let ctx = Ctx::new(&RUNTIME, &cancelled, &progress).with_chunk(64);

    unsafe {
        let n = 1_000;
        let mut x = alloc_vec_i64(&ctx, n);
        x.as_mut_slice::<i64>().fill(1);
        let mut y = alloc_vec_i64(&ctx, n);
        y.as_mut_slice::<i64>().fill(2);
        let r = dispatch_plus(x, y, &ctx).unwrap();
        assert!(r.as_slice::<i64>().iter().all(|&v| v == 3));
    }

    assert_eq!(progress.load(Ordering::Relaxed), 1_000);
}
