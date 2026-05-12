//! `×` (times) kernels for I64 and F64.

use crate::alloc::{alloc_atom, alloc_vec_f64, alloc_vec_i64};
use crate::chunk::{drive_async, drive_sync, ChunkStep};
use crate::ctx::{Ctx, KernelErr};
use crate::exec::block_on;
use crate::kernels::{F64_CHUNK, I64_CHUNK};
use crate::kind::Kind;
use crate::madvise::madvise_dontneed_slice;
use crate::nulls::NULL_I64;
use crate::obj::{attr_flags, RefObj};
use crate::parallel;
use crate::simd;

// ---- I64 ChunkStep impls ----

pub struct MulI64VecVec<'a> { x: &'a [i64], y: &'a [i64], out: &'a mut [i64], off: usize, chunk: usize }
impl<'a> ChunkStep for MulI64VecVec<'a> {
    fn step(&mut self) -> Option<usize> {
        if self.off >= self.x.len() { return None; }
        let end = (self.off + self.chunk).min(self.x.len());
        simd::mul_i64(&mut self.out[self.off..end], &self.x[self.off..end], &self.y[self.off..end]);
        unsafe {
            madvise_dontneed_slice(&self.x[self.off..end]);
            madvise_dontneed_slice(&self.y[self.off..end]);
        }
        let n = end - self.off; self.off = end; Some(n)
    }
}

pub struct MulScalarI64<'a> { y: &'a [i64], s: i64, out: &'a mut [i64], off: usize, chunk: usize }
impl<'a> ChunkStep for MulScalarI64<'a> {
    fn step(&mut self) -> Option<usize> {
        if self.off >= self.y.len() { return None; }
        let end = (self.off + self.chunk).min(self.y.len());
        simd::mul_scalar_i64(&mut self.out[self.off..end], &self.y[self.off..end], self.s);
        unsafe { madvise_dontneed_slice(&self.y[self.off..end]); }
        let n = end - self.off; self.off = end; Some(n)
    }
}

// ---- I64 async dispatch ----

pub async unsafe fn times_i64_i64_async(x: RefObj, y: RefObj, ctx: &Ctx<'_>) -> Result<RefObj, KernelErr> {
    let chunk = if ctx.chunk_elems != 0 { ctx.chunk_elems } else { I64_CHUNK };
    match (x.is_atom(), y.is_atom()) {
        (true, true) => Ok(alloc_atom(Kind::I64, x.atom::<i64>().wrapping_mul(y.atom::<i64>()))),
        (true, false) | (false, true) => {
            let (s, v) = if x.is_atom() { (x.atom::<i64>(), y) } else { (y.atom::<i64>(), x) };
            let ys = v.as_slice::<i64>();
            let mut out = alloc_vec_i64(ctx, ys.len() as i64);
            {
                let os = out.as_mut_slice::<i64>();
                let mut k = MulScalarI64 { y: ys, s, out: os, off: 0, chunk };
                drive_async(&mut k, ctx).await?;
            }
            if (v.attr() & attr_flags::HAS_NULLS) != 0 {
                let os = out.as_mut_slice::<i64>();
                for i in 0..ys.len() { if ys[i] == NULL_I64 { os[i] = NULL_I64; } }
                out.set_attr(attr_flags::HAS_NULLS);
            }
            Ok(out)
        }
        (false, false) => {
            let xs = x.as_slice::<i64>();
            let ys = y.as_slice::<i64>();
            if xs.len() != ys.len() { return Err(KernelErr::Shape); }
            let mut out = alloc_vec_i64(ctx, xs.len() as i64);
            let has_nulls = (x.attr() | y.attr()) & attr_flags::HAS_NULLS != 0;
            let go_parallel = !has_nulls
                && ctx.runtime.parallel_enabled()
                && xs.len() >= parallel::PARALLEL_THRESHOLD;
            {
                let os = out.as_mut_slice::<i64>();
                if go_parallel {
                    parallel::parallel_for_each_mut(os, ctx, |range, my_os| {
                        let mut k = MulI64VecVec {
                            x: &xs[range.clone()],
                            y: &ys[range],
                            out: my_os,
                            off: 0,
                            chunk,
                        };
                        drive_sync(&mut k, ctx)
                    })?;
                } else {
                    let mut k = MulI64VecVec { x: xs, y: ys, out: os, off: 0, chunk };
                    drive_async(&mut k, ctx).await?;
                    if has_nulls {
                        for i in 0..xs.len() {
                            if xs[i] == NULL_I64 || ys[i] == NULL_I64 { os[i] = NULL_I64; }
                        }
                    }
                }
            }
            if has_nulls { out.set_attr(attr_flags::HAS_NULLS); }
            Ok(out)
        }
    }
}

pub unsafe fn times_i64_i64(x: RefObj, y: RefObj, ctx: &Ctx) -> Result<RefObj, KernelErr> {
    block_on(times_i64_i64_async(x, y, ctx))
}

// ---- F64 ChunkStep impls ----

pub struct MulF64VecVec<'a> { x: &'a [f64], y: &'a [f64], out: &'a mut [f64], off: usize, chunk: usize }
impl<'a> ChunkStep for MulF64VecVec<'a> {
    fn step(&mut self) -> Option<usize> {
        if self.off >= self.x.len() { return None; }
        let end = (self.off + self.chunk).min(self.x.len());
        simd::mul_f64(&mut self.out[self.off..end], &self.x[self.off..end], &self.y[self.off..end]);
        unsafe {
            madvise_dontneed_slice(&self.x[self.off..end]);
            madvise_dontneed_slice(&self.y[self.off..end]);
        }
        let n = end - self.off; self.off = end; Some(n)
    }
}

pub struct MulScalarF64<'a> { y: &'a [f64], s: f64, out: &'a mut [f64], off: usize, chunk: usize }
impl<'a> ChunkStep for MulScalarF64<'a> {
    fn step(&mut self) -> Option<usize> {
        if self.off >= self.y.len() { return None; }
        let end = (self.off + self.chunk).min(self.y.len());
        simd::mul_scalar_f64(&mut self.out[self.off..end], &self.y[self.off..end], self.s);
        unsafe { madvise_dontneed_slice(&self.y[self.off..end]); }
        let n = end - self.off; self.off = end; Some(n)
    }
}

// ---- F64 async dispatch ----

pub async unsafe fn times_f64_f64_async(x: RefObj, y: RefObj, ctx: &Ctx<'_>) -> Result<RefObj, KernelErr> {
    let chunk = if ctx.chunk_elems != 0 { ctx.chunk_elems } else { F64_CHUNK };
    match (x.is_atom(), y.is_atom()) {
        (true, true) => Ok(alloc_atom(Kind::F64, x.atom::<f64>() * y.atom::<f64>())),
        (true, false) | (false, true) => {
            let (s, v) = if x.is_atom() { (x.atom::<f64>(), y) } else { (y.atom::<f64>(), x) };
            let ys = v.as_slice::<f64>();
            let mut out = alloc_vec_f64(ctx, ys.len() as i64);
            {
                let os = out.as_mut_slice::<f64>();
                let mut k = MulScalarF64 { y: ys, s, out: os, off: 0, chunk };
                drive_async(&mut k, ctx).await?;
            }
            if (v.attr() & attr_flags::HAS_NULLS) != 0 || s.is_nan() {
                out.set_attr(attr_flags::HAS_NULLS);
            }
            Ok(out)
        }
        (false, false) => {
            let xs = x.as_slice::<f64>();
            let ys = y.as_slice::<f64>();
            if xs.len() != ys.len() { return Err(KernelErr::Shape); }
            let mut out = alloc_vec_f64(ctx, xs.len() as i64);
            let go_parallel = ctx.runtime.parallel_enabled()
                && xs.len() >= parallel::PARALLEL_THRESHOLD;
            {
                let os = out.as_mut_slice::<f64>();
                if go_parallel {
                    parallel::parallel_for_each_mut(os, ctx, |range, my_os| {
                        let mut k = MulF64VecVec {
                            x: &xs[range.clone()],
                            y: &ys[range],
                            out: my_os,
                            off: 0,
                            chunk,
                        };
                        drive_sync(&mut k, ctx)
                    })?;
                } else {
                    let mut k = MulF64VecVec { x: xs, y: ys, out: os, off: 0, chunk };
                    drive_async(&mut k, ctx).await?;
                }
            }
            if (x.attr() | y.attr()) & attr_flags::HAS_NULLS != 0 {
                out.set_attr(attr_flags::HAS_NULLS);
            }
            Ok(out)
        }
    }
}

pub unsafe fn times_f64_f64(x: RefObj, y: RefObj, ctx: &Ctx) -> Result<RefObj, KernelErr> {
    block_on(times_f64_f64_async(x, y, ctx))
}
