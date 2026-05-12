//! Adverbs.
//!
//! v1 implements `+/` (plus-over) for I64 and F64 with `ChunkStep`-based fold
//! kernels. Generic fallback is stubbed; full adverb surface lands in v2.

use crate::alloc::alloc_atom;
use crate::chunk::{drive_sync, ChunkStep};
use crate::ctx::{Ctx, KernelErr};
use crate::kind::Kind;
use crate::madvise::madvise_dontneed_slice;
use crate::nulls::NULL_I64;
use crate::obj::{attr_flags, RefObj};
use crate::op::OpId;
use crate::simd;

// ---- I64 sum ---------------------------------------------------------

pub struct SumI64<'a> {
    xs: &'a [i64],
    off: usize,
    chunk: usize,
    /// Accumulator; readable after `step()` returns `None`.
    pub acc: i64,
}

impl<'a> SumI64<'a> {
    pub fn new(xs: &'a [i64], chunk: usize) -> Self {
        SumI64 {
            xs,
            off: 0,
            chunk: chunk.max(1),
            acc: 0,
        }
    }
}

impl<'a> ChunkStep for SumI64<'a> {
    fn step(&mut self) -> Option<usize> {
        if self.off >= self.xs.len() {
            return None;
        }
        let end = (self.off + self.chunk).min(self.xs.len());
        let s = simd::sum_i64(&self.xs[self.off..end]);
        self.acc = self.acc.wrapping_add(s);
        unsafe { madvise_dontneed_slice(&self.xs[self.off..end]); }
        let n = end - self.off;
        self.off = end;
        Some(n)
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
        if self.off >= self.xs.len() {
            return None;
        }
        let end = (self.off + self.chunk).min(self.xs.len());
        let s = simd::sum_i64_skipping(&self.xs[self.off..end], NULL_I64);
        self.acc = self.acc.wrapping_add(s);
        unsafe { madvise_dontneed_slice(&self.xs[self.off..end]); }
        let n = end - self.off;
        self.off = end;
        Some(n)
    }
}

pub unsafe fn plus_over_i64(x: RefObj, ctx: &Ctx) -> Result<RefObj, KernelErr> {
    let xs = x.as_slice::<i64>();
    let chunk = if ctx.chunk_elems != 0 {
        ctx.chunk_elems
    } else {
        crate::kernels::I64_CHUNK
    };
    if (x.attr() & attr_flags::HAS_NULLS) == 0 {
        let mut k = SumI64::new(xs, chunk);
        drive_sync(&mut k, ctx)?;
        Ok(alloc_atom(Kind::I64, k.acc))
    } else {
        let mut k = SumI64SkipNulls::new(xs, chunk);
        drive_sync(&mut k, ctx)?;
        Ok(alloc_atom(Kind::I64, k.acc))
    }
}

// ---- F64 sum ---------------------------------------------------------

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
        if self.off >= self.xs.len() {
            return None;
        }
        let end = (self.off + self.chunk).min(self.xs.len());
        self.acc += simd::sum_f64(&self.xs[self.off..end]);
        unsafe { madvise_dontneed_slice(&self.xs[self.off..end]); }
        let n = end - self.off;
        self.off = end;
        Some(n)
    }
}

pub unsafe fn plus_over_f64(x: RefObj, ctx: &Ctx) -> Result<RefObj, KernelErr> {
    let xs = x.as_slice::<f64>();
    let chunk = if ctx.chunk_elems != 0 {
        ctx.chunk_elems
    } else {
        crate::kernels::F64_CHUNK
    };
    // NaN propagation is automatic; no separate skip-nulls kernel needed for f64.
    let mut k = SumF64::new(xs, chunk);
    drive_sync(&mut k, ctx)?;
    Ok(alloc_atom(Kind::F64, k.acc))
}

// ---- adverb entry --------------------------------------------------

pub unsafe fn over(op: OpId, x: RefObj, ctx: &Ctx) -> Result<RefObj, KernelErr> {
    // Atom argument: identity.
    if x.is_atom() {
        return Ok(x);
    }
    let k = x.kind_raw().unsigned_abs();
    match (op, k) {
        (OpId::Plus, k) if k == Kind::I64 as u8 => plus_over_i64(x, ctx),
        (OpId::Plus, k) if k == Kind::F64 as u8 => plus_over_f64(x, ctx),
        _ => Err(KernelErr::Type),
    }
}
