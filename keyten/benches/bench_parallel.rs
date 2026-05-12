//! Stage-2 microbench: parallel vec+vec speedup vs the sequential baseline.
//!
//! Run with:
//!     cargo bench -p keyten --bench bench_parallel
//!
//! For each size, measures one sequential run (parallel flag off) and a
//! sweep of worker counts {1, 2, 4, 8, cpu_count} with the flag on. Reports
//! ns/elem and speedup vs the sequential baseline. Each measurement uses
//! freshly-allocated inputs because the kernel's `madvise(MADV_DONTNEED)`
//! zeroes consumed input pages on Linux.

use std::hint::black_box;
use std::time::{Duration, Instant};

use keyten::alloc::alloc_vec_i64;
use keyten::block_on;
use keyten::kernels::plus::plus_i64_vec_vec_async;
use keyten::{Ctx, RefObj, RUNTIME};

fn make_vec(n: i64, f: impl Fn(i64) -> i64) -> RefObj {
    let mut v = unsafe { alloc_vec_i64(&Ctx::quiet(), n) };
    let s = unsafe { v.as_mut_slice::<i64>() };
    for i in 0..n as usize {
        s[i] = f(i as i64);
    }
    v
}

/// One end-to-end run: build inputs, run kernel, drop result. The
/// measurement covers only the kernel call, not input construction.
fn run_one(n: i64) -> Duration {
    let x = make_vec(n, |i| i);
    let y = make_vec(n, |i| i.wrapping_mul(7).wrapping_sub(13));
    let start = Instant::now();
    let result = block_on(async move {
        unsafe {
            plus_i64_vec_vec_async(x, y, &Ctx::quiet())
                .await
                .expect("plus kernel must succeed")
        }
    });
    let elapsed = start.elapsed();
    // Touch one element so the optimizer can't eliminate the call.
    black_box(unsafe { result.as_slice::<i64>() }[0]);
    drop(result);
    elapsed
}

fn unique_worker_counts(cpu: usize) -> Vec<usize> {
    let mut wcs: Vec<usize> = vec![1, 2, 4, 8, cpu];
    wcs.sort_unstable();
    wcs.dedup();
    wcs.into_iter().filter(|&w| w > 0).collect()
}

fn bench_size(n: i64, label: &str) {
    println!();
    println!("--- n = {} ({}) -------------------------------------", label, n);

    // Sequential baseline.
    RUNTIME.set_parallel(false);
    RUNTIME.set_worker_count(0);
    let t_seq = run_one(n);
    let ns_seq = t_seq.as_nanos() as f64 / n as f64;
    println!(
        "{:<14} {:>8.3} ms   {:>6.3} ns/elem",
        "sequential:",
        t_seq.as_secs_f64() * 1000.0,
        ns_seq,
    );

    // Parallel sweep.
    RUNTIME.set_parallel(true);
    for nw in unique_worker_counts(RUNTIME.cpu_count()) {
        RUNTIME.set_worker_count(nw);
        let t = run_one(n);
        let ns = t.as_nanos() as f64 / n as f64;
        let speedup = t_seq.as_secs_f64() / t.as_secs_f64();
        println!(
            "par  nw={:>2}:    {:>8.3} ms   {:>6.3} ns/elem   {:>5.2}x",
            nw,
            t.as_secs_f64() * 1000.0,
            ns,
            speedup,
        );
    }

    // Restore defaults.
    RUNTIME.set_parallel(false);
    RUNTIME.set_worker_count(0);
}

fn main() {
    println!("keyten v2 stage-2 parallel kernel microbench");
    println!("cpu_count: {}", RUNTIME.cpu_count());
    println!("PARALLEL_THRESHOLD: {} elements", keyten::parallel::PARALLEL_THRESHOLD);
    println!("I64_CHUNK: {} elements per chunk", keyten::kernels::I64_CHUNK);

    bench_size(1_000_000, "1M");
    bench_size(10_000_000, "10M");
    bench_size(100_000_000, "100M");
}
