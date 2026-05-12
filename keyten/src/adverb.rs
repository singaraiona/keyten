//! Adverbs.
//!
//! v1 implements `+/` (plus-over) for I64 and F64 with `ChunkStep`-based fold
//! kernels. Generic fallback is stubbed; full adverb surface lands in v2.
//!
//! Same layering as the verb dispatchers: kernel-pair fns are unsafe (they
//! assume kind), the public `over` / `over_async` entries are safe (they
//! verify kind and wrap the unsafe kernel calls).

use crate::alloc::alloc_atom;
use crate::chunk::{drive_async, drive_sync, ChunkStep};
use crate::ctx::{Ctx, KernelErr};
use crate::exec::block_on;
use crate::kind::Kind;
use crate::madvise::madvise_dontneed_slice;
use crate::nulls::NULL_I64;
use crate::obj::{attr_flags, RefObj};
use crate::op::OpId;
use crate::parallel;
use crate::simd;

// =======================================================================
// ChunkStep impls — sums
// =======================================================================

pub struct SumI64<'a> {
    xs: &'a [i64],
    off: usize,
    chunk: usize,
    pub acc: i64,
}

impl<'a> SumI64<'a> {
    pub fn new(xs: &'a [i64], chunk: usize) -> Self {
        SumI64 { xs, off: 0, chunk: chunk.max(1), acc: 0 }
    }
}

impl<'a> ChunkStep for SumI64<'a> {
    fn step(&mut self) -> Option<usize> {
        if self.off >= self.xs.len() { return None; }
        let end = (self.off + self.chunk).min(self.xs.len());
        self.acc = self.acc.wrapping_add(simd::sum_i64(&self.xs[self.off..end]));
        unsafe { madvise_dontneed_slice(&self.xs[self.off..end]); }
        let n = end - self.off; self.off = end; Some(n)
    }
}

pub struct SumI64SkipNulls<'a> {
    xs: &'a [i64],
    off: usize,
    chunk: usize,
    pub acc: i64,
}

impl<'a> SumI64SkipNulls<'a> {
    pub fn new(xs: &'a [i64], chunk: usize) -> Self {
        SumI64SkipNulls { xs, off: 0, chunk: chunk.max(1), acc: 0 }
    }
}

impl<'a> ChunkStep for SumI64SkipNulls<'a> {
    fn step(&mut self) -> Option<usize> {
        if self.off >= self.xs.len() { return None; }
        let end = (self.off + self.chunk).min(self.xs.len());
        self.acc = self.acc.wrapping_add(simd::sum_i64_skipping(&self.xs[self.off..end], NULL_I64));
        unsafe { madvise_dontneed_slice(&self.xs[self.off..end]); }
        let n = end - self.off; self.off = end; Some(n)
    }
}

pub struct SumF64<'a> {
    xs: &'a [f64],
    off: usize,
    chunk: usize,
    pub acc: f64,
}

impl<'a> SumF64<'a> {
    pub fn new(xs: &'a [f64], chunk: usize) -> Self {
        SumF64 { xs, off: 0, chunk: chunk.max(1), acc: 0.0 }
    }
}

impl<'a> ChunkStep for SumF64<'a> {
    fn step(&mut self) -> Option<usize> {
        if self.off >= self.xs.len() { return None; }
        let end = (self.off + self.chunk).min(self.xs.len());
        self.acc += simd::sum_f64(&self.xs[self.off..end]);
        unsafe { madvise_dontneed_slice(&self.xs[self.off..end]); }
        let n = end - self.off; self.off = end; Some(n)
    }
}

// =======================================================================
// Kernel-pair fns — unsafe; caller asserts the kind
// =======================================================================

/// # Safety
/// `x` must be an I64-kinded vector.
pub unsafe fn plus_over_i64(x: RefObj, ctx: &Ctx) -> Result<RefObj, KernelErr> {
    let xs = x.as_slice::<i64>();
    let chunk = if ctx.chunk_elems != 0 { ctx.chunk_elems } else { crate::kernels::I64_CHUNK };
    let acc = if (x.attr() & attr_flags::HAS_NULLS) == 0 {
        let mut k = SumI64::new(xs, chunk);
        drive_sync(&mut k, ctx)?;
        k.acc
    } else {
        let mut k = SumI64SkipNulls::new(xs, chunk);
        drive_sync(&mut k, ctx)?;
        k.acc
    };
    Ok(alloc_atom(Kind::I64, acc))
}

/// # Safety
/// `x` must be an F64-kinded vector.
pub unsafe fn plus_over_f64(x: RefObj, ctx: &Ctx) -> Result<RefObj, KernelErr> {
    let xs = x.as_slice::<f64>();
    let chunk = if ctx.chunk_elems != 0 { ctx.chunk_elems } else { crate::kernels::F64_CHUNK };
    let mut k = SumF64::new(xs, chunk);
    drive_sync(&mut k, ctx)?;
    Ok(alloc_atom(Kind::F64, k.acc))
}

/// # Safety
/// `x` must be an I64-kinded vector.
pub async unsafe fn plus_over_i64_async(x: RefObj, ctx: &Ctx<'_>) -> Result<RefObj, KernelErr> {
    let xs = x.as_slice::<i64>();
    let chunk = if ctx.chunk_elems != 0 { ctx.chunk_elems } else { crate::kernels::I64_CHUNK };
    let has_nulls = (x.attr() & attr_flags::HAS_NULLS) != 0;
    let go_parallel = !has_nulls
        && ctx.runtime.parallel_enabled()
        && xs.len() >= parallel::PARALLEL_THRESHOLD;

    let acc = if go_parallel {
        // i64 sum is associative under wrapping_add, so parallel partial
        // order doesn't change the final value.
        parallel::parallel_reduce(
            xs.len(),
            ctx,
            0i64,
            |r| SumI64::new(&xs[r], chunk),
            |k| k.acc,
            i64::wrapping_add,
        )?
    } else if !has_nulls {
        let mut k = SumI64::new(xs, chunk);
        drive_async(&mut k, ctx).await?;
        k.acc
    } else {
        let mut k = SumI64SkipNulls::new(xs, chunk);
        drive_async(&mut k, ctx).await?;
        k.acc
    };
    Ok(alloc_atom(Kind::I64, acc))
}

// =======================================================================
// Scan: `op\x` — running aggregate. Produces a vector the same length as x.
// =======================================================================

/// `+\x` for I64 vectors: running sum.
///
/// Sequential path is a trivial O(n) loop. Parallel path is the standard
/// two-pass prefix-sum: (1) each worker computes its local prefix sum and
/// records its sub-range total; (2) sequentially scan the totals to get
/// each worker's cumulative offset; (3) each worker adds its offset to
/// every element in its sub-range. Step 2 is O(nworkers) which is
/// negligible compared to step 1 + 3.
///
/// # Safety
/// `x` must be an I64-kinded vector.
pub async unsafe fn plus_scan_i64_async(
    x: RefObj,
    ctx: &Ctx<'_>,
) -> Result<RefObj, KernelErr> {
    let xs = x.as_slice::<i64>();
    let n = xs.len();
    let mut out = crate::alloc::alloc_vec_i64(ctx, n as i64);
    if n == 0 {
        return Ok(out);
    }
    let go_parallel = ctx.runtime.parallel_enabled() && n >= parallel::PARALLEL_THRESHOLD;

    if !go_parallel {
        let os = out.as_mut_slice::<i64>();
        let mut acc: i64 = 0;
        for i in 0..n {
            acc = acc.wrapping_add(xs[i]);
            os[i] = acc;
        }
        return Ok(out);
    }

    // --- parallel two-pass prefix sum ---
    let nw = ctx.runtime.worker_count().min(n).max(1);
    let ranges = parallel::partition::balanced(n, nw);

    let os = out.as_mut_slice::<i64>();
    let xs_addr = xs.as_ptr() as usize;
    let os_addr = os.as_mut_ptr() as usize;

    // Each worker writes local prefix sum into its sub-range and reports
    // its local total. We use a Mutex<Vec<(idx, total)>> to gather totals
    // out of the parallel scope; per-worker push, ordering is range index.
    let totals: std::sync::Mutex<Vec<(usize, i64)>> =
        std::sync::Mutex::new(Vec::with_capacity(nw));

    std::thread::scope(|s| {
        for (worker_idx, range) in ranges.iter().enumerate() {
            let r = range.clone();
            let totals_ref = &totals;
            s.spawn(move || {
                let xs = xs_addr as *const i64;
                let os = os_addr as *mut i64;
                let mut acc: i64 = 0;
                unsafe {
                    for i in r.start..r.end {
                        acc = acc.wrapping_add(*xs.add(i));
                        *os.add(i) = acc;
                    }
                }
                totals_ref.lock().unwrap().push((worker_idx, acc));
            });
        }
    });

    // Sequentially compute cumulative offsets across workers (excluding
    // worker 0, whose local prefix is already global).
    let mut totals_vec = totals.into_inner().unwrap();
    totals_vec.sort_by_key(|&(i, _)| i);
    let mut offsets: Vec<i64> = vec![0; nw];
    let mut running: i64 = 0;
    for (idx, t) in totals_vec.iter() {
        offsets[*idx] = running;
        running = running.wrapping_add(*t);
    }

    // Second parallel pass: each worker adds its offset to every element
    // in its range. (Worker 0 has offset 0 — skip.)
    std::thread::scope(|s| {
        for (worker_idx, range) in ranges.into_iter().enumerate() {
            if offsets[worker_idx] == 0 {
                continue;
            }
            let off = offsets[worker_idx];
            let os_addr = os_addr;
            let r = range;
            s.spawn(move || unsafe {
                let os = os_addr as *mut i64;
                for i in r.start..r.end {
                    *os.add(i) = (*os.add(i)).wrapping_add(off);
                }
            });
        }
    });

    let _ = &mut out;
    Ok(out)
}

/// `+\x` for F64 vectors. Same two-pass shape but with f64 accumulators.
/// Floating-point summation is not associative, so the parallel result may
/// differ from sequential by last-ULP on pathological inputs. Documented
/// per the Runtime.deterministic note in `parallel/reduce.rs`.
///
/// # Safety
/// `x` must be an F64-kinded vector.
pub async unsafe fn plus_scan_f64_async(
    x: RefObj,
    ctx: &Ctx<'_>,
) -> Result<RefObj, KernelErr> {
    let xs = x.as_slice::<f64>();
    let n = xs.len();
    let mut out = crate::alloc::alloc_vec_f64(ctx, n as i64);
    if n == 0 {
        return Ok(out);
    }
    // Always sequential for f64 scan in this commit — the order-sensitive
    // two-pass would deserve its own tested implementation. Drop in later.
    let os = out.as_mut_slice::<f64>();
    let mut acc: f64 = 0.0;
    for i in 0..n {
        acc += xs[i];
        os[i] = acc;
    }
    Ok(out)
}

/// # Safety
/// `x` must be an F64-kinded vector.
pub async unsafe fn plus_over_f64_async(x: RefObj, ctx: &Ctx<'_>) -> Result<RefObj, KernelErr> {
    let xs = x.as_slice::<f64>();
    let chunk = if ctx.chunk_elems != 0 { ctx.chunk_elems } else { crate::kernels::F64_CHUNK };
    let go_parallel = ctx.runtime.parallel_enabled() && xs.len() >= parallel::PARALLEL_THRESHOLD;

    let acc = if go_parallel {
        // f64 sum is NOT associative — parallel partials sum in a different
        // order from sequential, giving last-ULP-different results on
        // pathological inputs. Documented behavior; a future
        // Runtime.deterministic flag will route through the sequential
        // path when bit-exact behavior is required.
        parallel::parallel_reduce(
            xs.len(),
            ctx,
            0.0f64,
            |r| SumF64::new(&xs[r], chunk),
            |k| k.acc,
            |a, b| a + b,
        )?
    } else {
        let mut k = SumF64::new(xs, chunk);
        drive_async(&mut k, ctx).await?;
        k.acc
    };
    Ok(alloc_atom(Kind::F64, acc))
}

// =======================================================================
// Adverb entry — safe API boundary
// =======================================================================

/// Synchronous over: `op/x`. Returns the fold of `op` across `x`.
pub fn over(op: OpId, x: RefObj, ctx: &Ctx) -> Result<RefObj, KernelErr> {
    if x.is_atom() {
        return Ok(x);
    }
    let k = x.kind_raw().unsigned_abs();
    match (op, k) {
        // SAFETY: kinds verified by the match arm.
        (OpId::Plus, k) if k == Kind::I64 as u8 => unsafe { plus_over_i64(x, ctx) },
        (OpId::Plus, k) if k == Kind::F64 as u8 => unsafe { plus_over_f64(x, ctx) },
        _ => Err(KernelErr::Type),
    }
}

/// Async over: same semantics, yields between chunks.
pub async fn over_async(op: OpId, x: RefObj, ctx: &Ctx<'_>) -> Result<RefObj, KernelErr> {
    if x.is_atom() {
        return Ok(x);
    }
    let k = x.kind_raw().unsigned_abs();
    match (op, k) {
        (OpId::Plus, k) if k == Kind::I64 as u8 => unsafe { plus_over_i64_async(x, ctx) }.await,
        (OpId::Plus, k) if k == Kind::F64 as u8 => unsafe { plus_over_f64_async(x, ctx) }.await,
        _ => Err(KernelErr::Type),
    }
}

/// Async scan: `op\x` — running aggregate, same length as `x`.
pub async fn scan_async(op: OpId, x: RefObj, ctx: &Ctx<'_>) -> Result<RefObj, KernelErr> {
    if x.is_atom() {
        return Ok(x);
    }
    let k = x.kind_raw().unsigned_abs();
    match (op, k) {
        (OpId::Plus, k) if k == Kind::I64 as u8 => unsafe { plus_scan_i64_async(x, ctx) }.await,
        (OpId::Plus, k) if k == Kind::F64 as u8 => unsafe { plus_scan_f64_async(x, ctx) }.await,
        _ => Err(KernelErr::Type),
    }
}

// Sync shim using block_on (kept for callers that want a uniform sync API).
pub fn over_blocking(op: OpId, x: RefObj, ctx: &Ctx) -> Result<RefObj, KernelErr> {
    block_on(over_async(op, x, ctx))
}
