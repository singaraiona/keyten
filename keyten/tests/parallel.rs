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

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use keyten::adverb::plus_over_i64_async;
use keyten::alloc::alloc_vec_i64;
use keyten::block_on;
use keyten::kernels::plus::plus_i64_vec_vec_async;
use keyten::{Ctx, KernelErr, RefObj, RUNTIME};

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

// =======================================================================
// Reductions: plus_over_i64 (+/)
// =======================================================================

fn run_plus_over(parallel: bool, xs: &[i64]) -> i64 {
    let _guard = SERIAL.lock().unwrap();
    let prev = RUNTIME.parallel_enabled();
    RUNTIME.set_parallel(parallel);

    let x = make_vec_i64(xs);
    let result = block_on(async move {
        unsafe {
            plus_over_i64_async(x, &Ctx::quiet())
                .await
                .expect("plus_over kernel must succeed")
        }
    });
    let acc = unsafe { result.atom::<i64>() };
    drop(result);

    RUNTIME.set_parallel(prev);
    acc
}

#[test]
fn plus_over_parallel_matches_sequential() {
    let n: i64 = 1_000_000;
    let xs: Vec<i64> = (0..n).collect();
    let seq = run_plus_over(false, &xs);
    let par = run_plus_over(true, &xs);
    assert_eq!(seq, par, "i64 sum must be exact under wrapping_add");
    // Sanity: 0+1+...+(n-1) = n*(n-1)/2.
    assert_eq!(seq, n * (n - 1) / 2);
}

#[test]
fn plus_over_large_parallel_matches_sequential() {
    // Above threshold, large enough to put real work on each worker.
    let n: i64 = 50_000_000;
    let xs: Vec<i64> = (0..n).collect();
    let seq = run_plus_over(false, &xs);
    let par = run_plus_over(true, &xs);
    assert_eq!(seq, par);
    assert_eq!(seq, n * (n - 1) / 2);
}

// =======================================================================
// til: parallel produces the same monotonic sequence as sequential
// =======================================================================

#[test]
fn til_parallel_matches_sequential() {
    use keyten::kernels::til::til_async;
    use keyten::alloc::alloc_atom;
    use keyten::Kind;

    let _guard = SERIAL.lock().unwrap();
    let prev = RUNTIME.parallel_enabled();

    fn run(n: i64) -> Vec<i64> {
        let n_atom = unsafe { alloc_atom(Kind::I64, n) };
        let out = block_on(async {
            unsafe { til_async(n_atom, &Ctx::quiet()).await.expect("til succeeds") }
        });
        let v = unsafe { out.as_slice::<i64>() }.to_vec();
        drop(out);
        v
    }

    let n: i64 = 1_000_000;

    RUNTIME.set_parallel(false);
    let seq = run(n);
    RUNTIME.set_parallel(true);
    let par = run(n);

    RUNTIME.set_parallel(prev);

    // Every position must be its own index.
    assert_eq!(seq.len(), n as usize);
    for i in 0..seq.len() {
        assert_eq!(seq[i], i as i64, "sequential til wrong at {i}");
    }
    assert_eq!(seq, par, "parallel til diverged from sequential");
}

// =======================================================================
// Cancellation under parallel execution
// =======================================================================

// =======================================================================
// Atomic refcount under cross-thread contention
// =======================================================================

#[test]
fn refobj_clone_drop_across_threads_no_leak() {
    // Stress the atomic rc clone/drop paths from many threads simultaneously
    // on a shared RefObj. Each thread does N inner clone+drop pairs (which
    // cancel out) plus one owned clone (which is dropped on thread exit).
    // Final rc must equal the test thread's holding alone.
    //
    // Catches: wrong Ordering on fetch_add/fetch_sub (release-vs-acquire
    // mismatch produces leaks or double-frees that TSan-without-sanitization
    // misses).
    const N_THREADS: usize = 16;
    const ITERS_PER_THREAD: usize = 200_000;

    let v = make_vec_i64(&[1, 2, 3, 4]);
    assert_eq!(v.rc(), 1);

    let shared = v.clone();
    assert_eq!(v.rc(), 2);

    std::thread::scope(|s| {
        for _ in 0..N_THREADS {
            let owned = shared.clone();
            s.spawn(move || {
                for _ in 0..ITERS_PER_THREAD {
                    let c = owned.clone();
                    drop(c);
                }
                // `owned` is dropped here at thread exit.
            });
        }
    });

    drop(shared);
    assert_eq!(
        v.rc(),
        1,
        "rc leaked or double-decremented across threads: expected 1, got {}",
        v.rc(),
    );
}

#[test]
fn refobj_send_across_thread_boundary() {
    // Move (not clone) a RefObj into another thread, drop it there. Final
    // rc check would have to be done in the dropping thread — instead, just
    // verify that the original test thread doesn't see the underlying
    // memory after the move (because we don't own it any more).
    let v = make_vec_i64(&[10, 20, 30]);
    let v_moved = v.clone();
    let original_data: Vec<i64> = unsafe { v.as_slice::<i64>() }.to_vec();

    let result = std::thread::spawn(move || {
        // Read the slice from the moved RefObj — proves the data is
        // accessible across the thread boundary.
        let s: Vec<i64> = unsafe { v_moved.as_slice::<i64>() }.to_vec();
        // v_moved is dropped here when the thread exits.
        s
    })
    .join()
    .expect("thread should not panic");

    assert_eq!(result, original_data);
    // v in the original thread is still alive (we cloned, then sent the clone).
    assert_eq!(v.rc(), 1);
}

#[test]
fn cancellation_propagates_to_all_workers() {
    // Start a long-running parallel kernel (~200 ms of work at 2 ns/elem),
    // flip ctx.cancelled mid-flight, expect Err(Cancelled) back within
    // one chunk-time per worker (~60-100 ms worst case). The flag is
    // observed by drive_sync inside every spawned worker; any worker that
    // sees it sets a shared atomic and the others observe Err(Cancelled)
    // on their next chunk boundary via scope.join.
    let _guard = SERIAL.lock().unwrap();
    let prev = RUNTIME.parallel_enabled();
    RUNTIME.set_parallel(true);

    let n: i64 = 100_000_000;
    let xs: Vec<i64> = (0..n).collect();
    let ys: Vec<i64> = (0..n).map(|i| i.wrapping_mul(2)).collect();
    let x = make_vec_i64(&xs);
    let y = make_vec_i64(&ys);

    let cancelled = Arc::new(AtomicBool::new(false));
    let progress = Arc::new(AtomicU64::new(0));

    // Cancel after 20 ms — enough for workers to be deep in their chunks.
    let cancelled_c = Arc::clone(&cancelled);
    let canceller = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(20));
        cancelled_c.store(true, Ordering::Relaxed);
    });

    let start = Instant::now();
    let result = {
        let ctx = Ctx::new(&RUNTIME, &cancelled, &progress);
        block_on(async {
            unsafe { plus_i64_vec_vec_async(x, y, &ctx).await }
        })
    };
    let elapsed = start.elapsed();

    canceller.join().unwrap();
    RUNTIME.set_parallel(prev);

    assert!(
        matches!(result, Err(KernelErr::Cancelled)),
        "expected Err(Cancelled), got {:?}",
        result.as_ref().err(),
    );
    // Should observe cancellation well before the full work would finish.
    // 100M-element add takes ~200 ms sequential; 5x parallel speedup brings
    // it to ~40 ms minimum, so we have ~20 ms slack between cancel and
    // natural completion. We give 150 ms to allow for system noise.
    assert!(
        elapsed < Duration::from_millis(150),
        "cancellation took {} ms, expected <150 ms",
        elapsed.as_millis(),
    );
    // And progress should be well short of full work — proves we cancelled,
    // not just got lucky on a fast machine.
    let p = progress.load(Ordering::Relaxed);
    assert!(
        (p as i64) < n,
        "progress {} reached full work {} — cancellation didn't actually interrupt",
        p,
        n,
    );
}

#[test]
fn plus_over_below_threshold() {
    // Sub-threshold input: parallel flag has no effect, both paths sequential.
    let xs: Vec<i64> = (0..1024).collect();
    let seq = run_plus_over(false, &xs);
    let par = run_plus_over(true, &xs);
    assert_eq!(seq, par);
    assert_eq!(seq, 1024 * 1023 / 2);
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
