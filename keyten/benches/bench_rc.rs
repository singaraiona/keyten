//! Stage-0 microbench: refcount overhead on `RefObj` after the `u32` →
//! `AtomicU32` migration. Establishes a baseline number for future stages
//! (e.g. biased refcounting in v2.1) to compare against.
//!
//! Run with:
//!     cargo bench -p keyten --bench bench_rc
//!
//! Single-threaded only — the goal here is to quantify the *single-thread*
//! tax of going atomic, since that's the path every existing keyten user
//! exercises today. Multi-threaded contention benches land alongside the
//! Stage 2 worker pool.

use std::hint::black_box;
use std::time::Instant;

use keyten::alloc::alloc_atom;
use keyten::Kind;

const WARMUP: u64 = 5_000_000;
const ITERS: u64 = 100_000_000;

fn time_loop<F: FnMut()>(label: &str, iters: u64, mut body: F) {
    // Warm caches and let the branch predictor settle.
    for _ in 0..WARMUP {
        body();
    }
    let start = Instant::now();
    for _ in 0..iters {
        body();
    }
    let elapsed = start.elapsed();
    let ns_per_op = (elapsed.as_nanos() as f64) / (iters as f64);
    let m_ops = (iters as f64) / elapsed.as_secs_f64() / 1e6;
    println!(
        "{:32} {:>7.2} ns/op   {:>8.1} M ops/s   ({} iters in {:.3}s)",
        label,
        ns_per_op,
        m_ops,
        iters,
        elapsed.as_secs_f64(),
    );
}

fn main() {
    println!("keyten v2 stage-0 refcount microbench (single-threaded)");
    println!("{}", "-".repeat(80));

    // Hold one RefObj outside the timed loop. Clone+drop exercises one
    // Relaxed fetch_add + one AcqRel fetch_sub on the AtomicU32.
    let owner = unsafe { alloc_atom(Kind::I64, 42i64) };

    time_loop("clone + drop (rc bumps only)", ITERS, || {
        let c = black_box(owner.clone());
        drop(black_box(c));
    });

    // Two-deep clone chain: rc goes 1 → 2 → 3 → 2 → 1. Verifies the linear
    // scaling assumption (should be ~2× the single-clone cost).
    time_loop("clone × 2 + drop × 2", ITERS / 2, || {
        let c1 = black_box(owner.clone());
        let c2 = black_box(c1.clone());
        drop(black_box(c2));
        drop(black_box(c1));
    });

    // Alloc + drop: exercises the freelist push/pop path plus the
    // AtomicU32::new initialisation in write_header. Hits the per-thread
    // ATOM_POOL once the freelist warms up; first iters touch the system
    // allocator. Numbers reflect steady-state after warmup.
    time_loop("alloc_atom + drop", ITERS, || {
        let r = black_box(unsafe { alloc_atom(Kind::I64, 7i64) });
        drop(r);
    });

    drop(owner);
    println!("{}", "-".repeat(80));
    println!("Baseline established. Future stages compare against these numbers.");
}
