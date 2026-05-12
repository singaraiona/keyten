//! Stage-2 correctness gate: parallel kernel paths must produce byte-identical
//! results to the sequential paths.
//!
//! Important property of the v1 kernels: every chunk calls
//! `madvise_dontneed_slice` on its consumed input pages, which zeroes those
//! pages on Linux when the allocation is mmap-backed. **Inputs are therefore
//! not idempotent — a vector that has been read by a kernel may come back
//! partially zeroed.** Each kernel call below builds fresh input RefObjs to
//! avoid that footgun. The same `xs`/`ys` Vec<i64> data is materialised into
//! a brand-new RefObj per call.
//!
//! Tests flip the global `RUNTIME.parallel` flag and therefore serialise via
//! a shared `Mutex` — cargo runs integration tests in parallel and the flag
//! is process-wide.

use std::sync::Mutex;

use keyten::alloc::alloc_vec_i64;
use keyten::block_on;
use keyten::kernels::plus::plus_i64_vec_vec_async;
use keyten::{Ctx, RefObj, RUNTIME};

static SERIAL: Mutex<()> = Mutex::new(());

fn make_vec_i64(data: &[i64]) -> RefObj {
    let ctx = Ctx::quiet();
    let mut v = unsafe { alloc_vec_i64(&ctx, data.len() as i64) };
    let s = unsafe { v.as_mut_slice::<i64>() };
    s.copy_from_slice(data);
    v
}

/// Run plus_i64_vec_vec under a specific parallel setting, with **fresh**
/// input RefObjs built from `xs` and `ys` — the kernel's madvise(DONTNEED)
/// would otherwise zero pages of any input we tried to reuse across calls.
/// Returns the result as an owned `Vec<i64>` and releases the RefObj.
fn run_plus(parallel: bool, xs: &[i64], ys: &[i64]) -> Vec<i64> {
    let _guard = SERIAL.lock().unwrap();
    let prev = RUNTIME.parallel_enabled();
    RUNTIME.set_parallel(parallel);

    let x = make_vec_i64(xs);
    let y = make_vec_i64(ys);
    let result = block_on(async move {
        unsafe {
            plus_i64_vec_vec_async(x, y, &Ctx::quiet())
                .await
                .expect("plus kernel must succeed on valid inputs")
        }
    });
    let out = unsafe { result.as_slice::<i64>() }.to_vec();
    drop(result);

    RUNTIME.set_parallel(prev);
    out
}

fn assert_results_match(seq: &[i64], par: &[i64]) {
    assert_eq!(seq.len(), par.len(), "length mismatch");
    if seq == par {
        return;
    }
    for i in 0..seq.len() {
        if seq[i] != par[i] {
            panic!("differ at index {i}: seq={} par={}", seq[i], par[i]);
        }
    }
}

#[test]
fn small_input_below_threshold() {
    // n < PARALLEL_THRESHOLD (256 K) — even when the flag is on, this falls
    // through to the sequential path. Both runs must produce the same output
    // regardless.
    let xs: Vec<i64> = (0..1024i64).collect();
    let ys: Vec<i64> = (0..1024i64).map(|i| i.wrapping_mul(7) - 13).collect();
    let seq = run_plus(false, &xs, &ys);
    let par = run_plus(true, &xs, &ys);
    assert_results_match(&seq, &par);
}

#[test]
fn at_threshold() {
    let n: i64 = 256 * 1024;
    let xs: Vec<i64> = (0..n).collect();
    let ys: Vec<i64> = (0..n).map(|i| i.wrapping_mul(3)).collect();
    let seq = run_plus(false, &xs, &ys);
    let par = run_plus(true, &xs, &ys);
    assert_results_match(&seq, &par);
}

#[test]
fn above_threshold_one_million() {
    let n: i64 = 1_000_000;
    let xs: Vec<i64> = (0..n).collect();
    let ys: Vec<i64> = (0..n).map(|i| -(i * 5) + 42).collect();
    let seq = run_plus(false, &xs, &ys);
    let par = run_plus(true, &xs, &ys);
    assert_results_match(&seq, &par);
}

#[test]
fn well_above_threshold_ten_million() {
    // Exercises a chunk count well past PARALLEL_THRESHOLD on all sensible
    // worker counts. Also crosses the I64_CHUNK boundary multiple times
    // within each worker.
    let n: i64 = 10_000_000;
    let xs: Vec<i64> = (0..n).map(|i| i.wrapping_mul(11)).collect();
    let ys: Vec<i64> = (0..n).map(|i| i.wrapping_mul(13).wrapping_sub(99)).collect();
    let seq = run_plus(false, &xs, &ys);
    let par = run_plus(true, &xs, &ys);
    assert_results_match(&seq, &par);
}

#[test]
fn prime_length_above_threshold() {
    // A prime n stresses partition::balanced's unequal-split path: ranges
    // are not all the same length. Result must still match sequential.
    let n: i64 = 1_048_573; // prime
    let xs: Vec<i64> = (0..n).collect();
    let ys: Vec<i64> = (0..n).rev().collect();
    let seq = run_plus(false, &xs, &ys);
    let par = run_plus(true, &xs, &ys);
    // Every element of seq and par should equal n-1 (i + (n-1-i) = n-1).
    assert!(seq.iter().all(|&v| v == n - 1), "seq incorrect");
    assert_results_match(&seq, &par);
}
