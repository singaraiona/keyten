//! `×` (times) kernels for I64 and F64.

use crate::alloc::{alloc_atom, alloc_vec_f64, alloc_vec_i64};
use crate::chunk::{drive_sync, ChunkStep};
use crate::ctx::{Ctx, KernelErr};
use crate::kernels::{F64_CHUNK, I64_CHUNK};
use crate::kind::Kind;
use crate::madvise::madvise_dontneed_slice;
use crate::nulls::NULL_I64;
use crate::obj::{attr_flags, RefObj};
use crate::simd;

// ---------- I64 ----------

#[inline]
pub unsafe fn times_i64_atom_atom(x: RefObj, y: RefObj) -> RefObj {
    alloc_atom(Kind::I64, x.atom::<i64>().wrapping_mul(y.atom::<i64>()))
}

pub unsafe fn times_i64_scalar_vec(s: i64, y: RefObj, ctx: &Ctx) -> Result<RefObj, KernelErr> {
    let ys = y.as_slice::<i64>();
    let mut out = alloc_vec_i64(ctx, ys.len() as i64);
    let chunk = if ctx.chunk_elems != 0 { ctx.chunk_elems } else { I64_CHUNK };
    {
        let os = out.as_mut_slice::<i64>();
        struct Step<'a> { y: &'a [i64], s: i64, out: &'a mut [i64], off: usize, chunk: usize }
        impl<'a> ChunkStep for Step<'a> {
            fn step(&mut self) -> Option<usize> {
                if self.off >= self.y.len() { return None; }
                let end = (self.off + self.chunk).min(self.y.len());
                simd::mul_scalar_i64(&mut self.out[self.off..end], &self.y[self.off..end], self.s);
                unsafe { madvise_dontneed_slice(&self.y[self.off..end]); }
                let m = end - self.off;
                self.off = end;
                Some(m)
            }
        }
        let mut k = Step { y: ys, s, out: os, off: 0, chunk };
        drive_sync(&mut k, ctx)?;
    }
    if (y.attr() & attr_flags::HAS_NULLS) != 0 {
        let os = out.as_mut_slice::<i64>();
        for i in 0..ys.len() {
            if ys[i] == NULL_I64 { os[i] = NULL_I64; }
        }
        out.set_attr(attr_flags::HAS_NULLS);
    }
    Ok(out)
}

pub unsafe fn times_i64_vec_scalar(x: RefObj, s: i64, ctx: &Ctx) -> Result<RefObj, KernelErr> {
    times_i64_scalar_vec(s, x, ctx)
}

pub unsafe fn times_i64_vec_vec(x: RefObj, y: RefObj, ctx: &Ctx) -> Result<RefObj, KernelErr> {
    let xs = x.as_slice::<i64>();
    let ys = y.as_slice::<i64>();
    if xs.len() != ys.len() { return Err(KernelErr::Shape); }
    let mut out = alloc_vec_i64(ctx, xs.len() as i64);
    let chunk = if ctx.chunk_elems != 0 { ctx.chunk_elems } else { I64_CHUNK };
    let has_nulls = (x.attr() | y.attr()) & attr_flags::HAS_NULLS != 0;
    {
        let os = out.as_mut_slice::<i64>();
        struct Step<'a> { x: &'a [i64], y: &'a [i64], out: &'a mut [i64], off: usize, chunk: usize }
        impl<'a> ChunkStep for Step<'a> {
            fn step(&mut self) -> Option<usize> {
                if self.off >= self.x.len() { return None; }
                let end = (self.off + self.chunk).min(self.x.len());
                simd::mul_i64(&mut self.out[self.off..end], &self.x[self.off..end], &self.y[self.off..end]);
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
        if has_nulls {
            for i in 0..xs.len() {
                if xs[i] == NULL_I64 || ys[i] == NULL_I64 { os[i] = NULL_I64; }
            }
        }
    }
    if has_nulls { out.set_attr(attr_flags::HAS_NULLS); }
    Ok(out)
}

pub unsafe fn times_i64_i64(x: RefObj, y: RefObj, ctx: &Ctx) -> Result<RefObj, KernelErr> {
    match (x.is_atom(), y.is_atom()) {
        (true, true) => Ok(times_i64_atom_atom(x, y)),
        (true, false) => times_i64_scalar_vec(x.atom::<i64>(), y, ctx),
        (false, true) => times_i64_vec_scalar(x, y.atom::<i64>(), ctx),
        (false, false) => times_i64_vec_vec(x, y, ctx),
    }
}

// ---------- F64 ----------

#[inline]
pub unsafe fn times_f64_atom_atom(x: RefObj, y: RefObj) -> RefObj {
    alloc_atom(Kind::F64, x.atom::<f64>() * y.atom::<f64>())
}

pub unsafe fn times_f64_scalar_vec(s: f64, y: RefObj, ctx: &Ctx) -> Result<RefObj, KernelErr> {
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
                simd::mul_scalar_f64(&mut self.out[self.off..end], &self.y[self.off..end], self.s);
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
        out.set_attr(attr_flags::HAS_NULLS);
    }
    Ok(out)
}

pub unsafe fn times_f64_vec_scalar(x: RefObj, s: f64, ctx: &Ctx) -> Result<RefObj, KernelErr> {
    times_f64_scalar_vec(s, x, ctx)
}

pub unsafe fn times_f64_vec_vec(x: RefObj, y: RefObj, ctx: &Ctx) -> Result<RefObj, KernelErr> {
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
                simd::mul_f64(&mut self.out[self.off..end], &self.x[self.off..end], &self.y[self.off..end]);
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
    if (x.attr() | y.attr()) & attr_flags::HAS_NULLS != 0 {
        out.set_attr(attr_flags::HAS_NULLS);
    }
    Ok(out)
}

pub unsafe fn times_f64_f64(x: RefObj, y: RefObj, ctx: &Ctx) -> Result<RefObj, KernelErr> {
    match (x.is_atom(), y.is_atom()) {
        (true, true) => Ok(times_f64_atom_atom(x, y)),
        (true, false) => times_f64_scalar_vec(x.atom::<f64>(), y, ctx),
        (false, true) => times_f64_vec_scalar(x, y.atom::<f64>(), ctx),
        (false, false) => times_f64_vec_vec(x, y, ctx),
    }
}
