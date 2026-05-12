//! Progress counter matches total element count on a clean run.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use keyten::alloc::alloc_vec_i64;
use keyten::ctx::Ctx;
use keyten::op::dispatch_plus;
use keyten::runtime::RUNTIME;

#[test]
fn progress_equals_input_length() {
    let cancelled = AtomicBool::new(false);
    let progress = AtomicU64::new(0);
    let ctx = Ctx::new(&RUNTIME, &cancelled, &progress).with_chunk(1024);

    unsafe {
        let n = 10_000;
        let mut x = alloc_vec_i64(&ctx, n);
        x.as_mut_slice::<i64>().fill(1);
        let mut y = alloc_vec_i64(&ctx, n);
        y.as_mut_slice::<i64>().fill(2);
        let _ = dispatch_plus(x, y, &ctx).unwrap();
    }

    assert_eq!(progress.load(Ordering::Relaxed), 10_000);
}

#[test]
fn progress_accumulates_across_multiple_calls() {
    let cancelled = AtomicBool::new(false);
    let progress = AtomicU64::new(0);
    let ctx = Ctx::new(&RUNTIME, &cancelled, &progress).with_chunk(128);

    unsafe {
        for _ in 0..3 {
            let mut x = alloc_vec_i64(&ctx, 100);
            x.as_mut_slice::<i64>().fill(1);
            let mut y = alloc_vec_i64(&ctx, 100);
            y.as_mut_slice::<i64>().fill(1);
            let _ = dispatch_plus(x, y, &ctx).unwrap();
        }
    }

    assert_eq!(progress.load(Ordering::Relaxed), 300);
}
