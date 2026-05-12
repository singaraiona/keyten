//! `÷` (div) kernels — F64 only in v1.
//!
//! Per K convention, division promotes to F64. v1 implements `F64/F64`; integer
//! operands get promoted by the dispatcher before reaching this kernel.

use crate::alloc::{alloc_atom, alloc_vec_f64};
use crate::chunk::{drive_sync, ChunkStep};
use crate::ctx::{Ctx, KernelErr};
use crate::kernels::F64_CHUNK;
use crate::kind::Kind;
use crate::madvise::madvise_dontneed_slice;
use crate::obj::{attr_flags, RefObj};
use crate::simd;

#[inline]
pub unsafe fn div_f64_atom_atom(x: RefObj, y: RefObj) -> RefObj {
    alloc_atom(Kind::F64, x.atom::<f64>() / y.atom::<f64>())
}

pub unsafe fn div_f64_vec_scalar(x: RefObj, s: f64, ctx: &Ctx) -> Result<RefObj, KernelErr> {
    let xs = x.as_slice::<f64>();
    let mut out = alloc_vec_f64(ctx, xs.len() as i64);
    let chunk = if ctx.chunk_elems != 0 { ctx.chunk_elems } else { F64_CHUNK };
    {
        let os = out.as_mut_slice::<f64>();
        struct Step<'a> { x: &'a [f64], s: f64, out: &'a mut [f64], off: usize, chunk: usize }
        impl<'a> ChunkStep for Step<'a> {
            fn step(&mut self) -> Option<usize> {
                if self.off >= self.x.len() { return None; }
                let end = (self.off + self.chunk).min(self.x.len());
                simd::div_scalar_f64(&mut self.out[self.off..end], &self.x[self.off..end], self.s);
                unsafe { madvise_dontneed_slice(&self.x[self.off..end]); }
                let m = end - self.off;
                self.off = end;
                Some(m)
            }
        }
        let mut k = Step { x: xs, s, out: os, off: 0, chunk };
        drive_sync(&mut k, ctx)?;
    }
    // Division by zero produces inf or NaN; we conservatively mark HAS_NULLS
    // when any input nulls were present, or when divisor is zero/NaN.
    if (x.attr() & attr_flags::HAS_NULLS) != 0 || s == 0.0 || s.is_nan() {
        out.set_attr(attr_flags::HAS_NULLS);
    }
    Ok(out)
}

pub unsafe fn div_f64_scalar_vec(s: f64, y: RefObj, ctx: &Ctx) -> Result<RefObj, KernelErr> {
    let ys = y.as_slice::<f64>();
    let mut out = alloc_vec_f64(ctx, ys.len() as i64);
    let chunk = if ctx.chunk_elems != 0 { ctx.chunk_elems } else { F64_CHUNK };
    {
        let os = out.as_mut_slice::<f64>();
        struct Step<'a> { y: &'a [f64], s: f64, out: &'a mut [f64], off: usize, chunk: usize }
        impl<'a> ChunkStep for Step<'a> {
            fn step(&mut self) -> Option<usize> {
                if self.off >= self.y.len() { return None; }
                let end = (self.off + self.chunk).min(self.y.len());
                simd::scalar_div_vec_f64(&mut self.out[self.off..end], self.s, &self.y[self.off..end]);
                unsafe { madvise_dontneed_slice(&self.y[self.off..end]); }
                let m = end - self.off;
                self.off = end;
                Some(m)
            }
        }
        let mut k = Step { y: ys, s, out: os, off: 0, chunk };
        drive_sync(&mut k, ctx)?;
    }
    if (y.attr() & attr_flags::HAS_NULLS) != 0 || s.is_nan() {
        // Output may contain inf/NaN if any divisor in y was zero — set
        // HAS_NULLS conservatively (a precise scan would cost a pass; skipped).
        out.set_attr(attr_flags::HAS_NULLS);
    }
    Ok(out)
}

pub unsafe fn div_f64_vec_vec(x: RefObj, y: RefObj, ctx: &Ctx) -> Result<RefObj, KernelErr> {
    let xs = x.as_slice::<f64>();
    let ys = y.as_slice::<f64>();
    if xs.len() != ys.len() { return Err(KernelErr::Shape); }
    let mut out = alloc_vec_f64(ctx, xs.len() as i64);
    let chunk = if ctx.chunk_elems != 0 { ctx.chunk_elems } else { F64_CHUNK };
    {
        let os = out.as_mut_slice::<f64>();
        struct Step<'a> { x: &'a [f64], y: &'a [f64], out: &'a mut [f64], off: usize, chunk: usize }
        impl<'a> ChunkStep for Step<'a> {
            fn step(&mut self) -> Option<usize> {
                if self.off >= self.x.len() { return None; }
                let end = (self.off + self.chunk).min(self.x.len());
                simd::div_f64(&mut self.out[self.off..end], &self.x[self.off..end], &self.y[self.off..end]);
                unsafe {
                    madvise_dontneed_slice(&self.x[self.off..end]);
                    madvise_dontneed_slice(&self.y[self.off..end]);
                }
                let m = end - self.off;
                self.off = end;
                Some(m)
            }
        }
        let mut k = Step { x: xs, y: ys, out: os, off: 0, chunk };
        drive_sync(&mut k, ctx)?;
    }
    // Always mark HAS_NULLS on division: zero-divisors in y produce inf/NaN,
    // and we don't pre-scan to prove otherwise.
    out.set_attr(attr_flags::HAS_NULLS);
    Ok(out)
}

pub unsafe fn div_f64_f64(x: RefObj, y: RefObj, ctx: &Ctx) -> Result<RefObj, KernelErr> {
    match (x.is_atom(), y.is_atom()) {
        (true, true) => Ok(div_f64_atom_atom(x, y)),
        (true, false) => div_f64_scalar_vec(x.atom::<f64>(), y, ctx),
        (false, true) => div_f64_vec_scalar(x, y.atom::<f64>(), ctx),
        (false, false) => div_f64_vec_vec(x, y, ctx),
    }
}
