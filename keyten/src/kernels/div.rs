//! `÷` (div) kernels — F64 only. Integer operands are promoted to F64 by the
//! dispatcher.

use crate::alloc::{alloc_atom, alloc_vec_f64};
use crate::chunk::{drive_async, ChunkStep};
use crate::ctx::{Ctx, KernelErr};
use crate::exec::block_on;
use crate::kernels::F64_CHUNK;
use crate::kind::Kind;
use crate::madvise::madvise_dontneed_slice;
use crate::obj::{attr_flags, RefObj};
use crate::simd;

pub struct DivF64VecVec<'a> { x: &'a [f64], y: &'a [f64], out: &'a mut [f64], off: usize, chunk: usize }
impl<'a> ChunkStep for DivF64VecVec<'a> {
    fn step(&mut self) -> Option<usize> {
        if self.off >= self.x.len() { return None; }
        let end = (self.off + self.chunk).min(self.x.len());
        simd::div_f64(&mut self.out[self.off..end], &self.x[self.off..end], &self.y[self.off..end]);
        unsafe {
            madvise_dontneed_slice(&self.x[self.off..end]);
            madvise_dontneed_slice(&self.y[self.off..end]);
        }
        let n = end - self.off; self.off = end; Some(n)
    }
}

pub struct DivVecScalarF64<'a> { x: &'a [f64], s: f64, out: &'a mut [f64], off: usize, chunk: usize }
impl<'a> ChunkStep for DivVecScalarF64<'a> {
    fn step(&mut self) -> Option<usize> {
        if self.off >= self.x.len() { return None; }
        let end = (self.off + self.chunk).min(self.x.len());
        simd::div_scalar_f64(&mut self.out[self.off..end], &self.x[self.off..end], self.s);
        unsafe { madvise_dontneed_slice(&self.x[self.off..end]); }
        let n = end - self.off; self.off = end; Some(n)
    }
}

pub struct DivScalarVecF64<'a> { y: &'a [f64], s: f64, out: &'a mut [f64], off: usize, chunk: usize }
impl<'a> ChunkStep for DivScalarVecF64<'a> {
    fn step(&mut self) -> Option<usize> {
        if self.off >= self.y.len() { return None; }
        let end = (self.off + self.chunk).min(self.y.len());
        simd::scalar_div_vec_f64(&mut self.out[self.off..end], self.s, &self.y[self.off..end]);
        unsafe { madvise_dontneed_slice(&self.y[self.off..end]); }
        let n = end - self.off; self.off = end; Some(n)
    }
}

pub async unsafe fn div_f64_f64_async(x: RefObj, y: RefObj, ctx: &Ctx<'_>) -> Result<RefObj, KernelErr> {
    let chunk = if ctx.chunk_elems != 0 { ctx.chunk_elems } else { F64_CHUNK };
    match (x.is_atom(), y.is_atom()) {
        (true, true) => Ok(alloc_atom(Kind::F64, x.atom::<f64>() / y.atom::<f64>())),
        (true, false) => {
            let s = x.atom::<f64>();
            let ys = y.as_slice::<f64>();
            let mut out = alloc_vec_f64(ctx, ys.len() as i64);
            {
                let os = out.as_mut_slice::<f64>();
                let mut k = DivScalarVecF64 { y: ys, s, out: os, off: 0, chunk };
                drive_async(&mut k, ctx).await?;
            }
            if (y.attr() & attr_flags::HAS_NULLS) != 0 || s.is_nan() {
                out.set_attr(attr_flags::HAS_NULLS);
            }
            Ok(out)
        }
        (false, true) => {
            let s = y.atom::<f64>();
            let xs = x.as_slice::<f64>();
            let mut out = alloc_vec_f64(ctx, xs.len() as i64);
            {
                let os = out.as_mut_slice::<f64>();
                let mut k = DivVecScalarF64 { x: xs, s, out: os, off: 0, chunk };
                drive_async(&mut k, ctx).await?;
            }
            if (x.attr() & attr_flags::HAS_NULLS) != 0 || s == 0.0 || s.is_nan() {
                out.set_attr(attr_flags::HAS_NULLS);
            }
            Ok(out)
        }
        (false, false) => {
            let xs = x.as_slice::<f64>();
            let ys = y.as_slice::<f64>();
            if xs.len() != ys.len() { return Err(KernelErr::Shape); }
            let mut out = alloc_vec_f64(ctx, xs.len() as i64);
            {
                let os = out.as_mut_slice::<f64>();
                let mut k = DivF64VecVec { x: xs, y: ys, out: os, off: 0, chunk };
                drive_async(&mut k, ctx).await?;
            }
            out.set_attr(attr_flags::HAS_NULLS);
            Ok(out)
        }
    }
}

pub unsafe fn div_f64_f64(x: RefObj, y: RefObj, ctx: &Ctx) -> Result<RefObj, KernelErr> {
    block_on(div_f64_f64_async(x, y, ctx))
}
