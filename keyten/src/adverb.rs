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
    let acc = if (x.attr() & attr_flags::HAS_NULLS) == 0 {
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

/// # Safety
/// `x` must be an F64-kinded vector.
pub async unsafe fn plus_over_f64_async(x: RefObj, ctx: &Ctx<'_>) -> Result<RefObj, KernelErr> {
    let xs = x.as_slice::<f64>();
    let chunk = if ctx.chunk_elems != 0 { ctx.chunk_elems } else { crate::kernels::F64_CHUNK };
    let mut k = SumF64::new(xs, chunk);
    drive_async(&mut k, ctx).await?;
    Ok(alloc_atom(Kind::F64, k.acc))
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

// Sync shim using block_on (kept for callers that want a uniform sync API).
pub fn over_blocking(op: OpId, x: RefObj, ctx: &Ctx) -> Result<RefObj, KernelErr> {
    block_on(over_async(op, x, ctx))
}
