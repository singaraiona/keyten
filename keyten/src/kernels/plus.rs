//! `+` kernels for I64 and F64.
//!
//! Layout:
//! 1. `ChunkStep` state machines (`Add*VecVec`, `AddScalarVec*`, `*InPlace`) —
//!    the actual per-chunk work.
//! 2. `*_async` dispatch entries — the canonical async API. They build a
//!    state machine and drive it with `drive_async`.
//! 3. Sync entries (no suffix) — one-line `block_on` shims around the async
//!    versions. Identical end-to-end work; ~0 throughput overhead since the
//!    only suspension point is `YieldNow` which terminates after one re-poll.

use crate::alloc::{alloc_atom, alloc_vec_f64, alloc_vec_i64};
use crate::chunk::{drive_async, ChunkStep};
use crate::ctx::{Ctx, KernelErr};
use crate::exec::block_on;
use crate::kernels::{F64_CHUNK, I64_CHUNK};
use crate::kind::Kind;
use crate::madvise::madvise_dontneed_slice;
use crate::nulls::NULL_I64;
use crate::obj::{attr_flags, RefObj};
use crate::simd;

// =======================================================================
// I64 — ChunkStep impls
// =======================================================================

pub struct AddI64VecVec<'a> {
    x: &'a [i64],
    y: &'a [i64],
    out: &'a mut [i64],
    off: usize,
    chunk: usize,
}

impl<'a> AddI64VecVec<'a> {
    #[inline]
    pub fn new(x: &'a [i64], y: &'a [i64], out: &'a mut [i64], chunk: usize) -> Self {
        debug_assert_eq!(x.len(), y.len());
        debug_assert_eq!(x.len(), out.len());
        Self { x, y, out, off: 0, chunk: chunk.max(1) }
    }
}

impl<'a> ChunkStep for AddI64VecVec<'a> {
    #[inline]
    fn step(&mut self) -> Option<usize> {
        if self.off >= self.x.len() { return None; }
        let end = (self.off + self.chunk).min(self.x.len());
        simd::add_i64(
            &mut self.out[self.off..end],
            &self.x[self.off..end],
            &self.y[self.off..end],
        );
        unsafe {
            madvise_dontneed_slice(&self.x[self.off..end]);
            madvise_dontneed_slice(&self.y[self.off..end]);
        }
        let n = end - self.off;
        self.off = end;
        Some(n)
    }
}

/// Null-preserving variant. Runs the add unconditionally, then stamps NULL_I64
/// back at masked positions.
pub struct AddI64VecVecNulls<'a> {
    x: &'a [i64],
    y: &'a [i64],
    out: &'a mut [i64],
    off: usize,
    chunk: usize,
}

impl<'a> AddI64VecVecNulls<'a> {
    #[inline]
    pub fn new(x: &'a [i64], y: &'a [i64], out: &'a mut [i64], chunk: usize) -> Self {
        Self { x, y, out, off: 0, chunk: chunk.max(1) }
    }
}

impl<'a> ChunkStep for AddI64VecVecNulls<'a> {
    #[inline]
    fn step(&mut self) -> Option<usize> {
        if self.off >= self.x.len() { return None; }
        let end = (self.off + self.chunk).min(self.x.len());
        let xs = &self.x[self.off..end];
        let ys = &self.y[self.off..end];
        let os = &mut self.out[self.off..end];
        simd::add_i64(os, xs, ys);
        for i in 0..os.len() {
            if xs[i] == NULL_I64 || ys[i] == NULL_I64 { os[i] = NULL_I64; }
        }
        unsafe {
            madvise_dontneed_slice(xs);
            madvise_dontneed_slice(ys);
        }
        let n = end - self.off;
        self.off = end;
        Some(n)
    }
}

pub struct AddScalarVecI64<'a> {
    s: i64,
    x: &'a [i64],
    out: &'a mut [i64],
    off: usize,
    chunk: usize,
}

impl<'a> AddScalarVecI64<'a> {
    #[inline]
    pub fn new(s: i64, x: &'a [i64], out: &'a mut [i64], chunk: usize) -> Self {
        debug_assert_eq!(x.len(), out.len());
        Self { s, x, out, off: 0, chunk: chunk.max(1) }
    }
}

impl<'a> ChunkStep for AddScalarVecI64<'a> {
    #[inline]
    fn step(&mut self) -> Option<usize> {
        if self.off >= self.x.len() { return None; }
        let end = (self.off + self.chunk).min(self.x.len());
        simd::add_scalar_i64(&mut self.out[self.off..end], &self.x[self.off..end], self.s);
        unsafe { madvise_dontneed_slice(&self.x[self.off..end]); }
        let n = end - self.off;
        self.off = end;
        Some(n)
    }
}

pub struct AddI64InPlace<'a> {
    x: &'a mut [i64],
    y: &'a [i64],
    off: usize,
    chunk: usize,
}

impl<'a> AddI64InPlace<'a> {
    #[inline]
    pub fn new(x: &'a mut [i64], y: &'a [i64], chunk: usize) -> Self {
        debug_assert_eq!(x.len(), y.len());
        Self { x, y, off: 0, chunk: chunk.max(1) }
    }
}

impl<'a> ChunkStep for AddI64InPlace<'a> {
    #[inline]
    fn step(&mut self) -> Option<usize> {
        if self.off >= self.x.len() { return None; }
        let end = (self.off + self.chunk).min(self.x.len());
        let xs = &mut self.x[self.off..end];
        let ys = &self.y[self.off..end];
        for i in 0..xs.len() { xs[i] = xs[i].wrapping_add(ys[i]); }
        unsafe { madvise_dontneed_slice(ys); }
        let n = end - self.off;
        self.off = end;
        Some(n)
    }
}

// =======================================================================
// I64 — async dispatch (canonical)
// =======================================================================

#[inline]
pub unsafe fn plus_i64_atom_atom(x: RefObj, y: RefObj) -> RefObj {
    alloc_atom(Kind::I64, x.atom::<i64>().wrapping_add(y.atom::<i64>()))
}

pub async unsafe fn plus_i64_scalar_vec_async(s: i64, y: RefObj, ctx: &Ctx<'_>) -> Result<RefObj, KernelErr> {
    let ys = y.as_slice::<i64>();
    let mut out = alloc_vec_i64(ctx, ys.len() as i64);
    let chunk = if ctx.chunk_elems != 0 { ctx.chunk_elems } else { I64_CHUNK };
    {
        let os = out.as_mut_slice::<i64>();
        let mut k = AddScalarVecI64::new(s, ys, os, chunk);
        drive_async(&mut k, ctx).await?;
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

pub async unsafe fn plus_i64_vec_scalar_async(x: RefObj, s: i64, ctx: &Ctx<'_>) -> Result<RefObj, KernelErr> {
    plus_i64_scalar_vec_async(s, x, ctx).await
}

pub async unsafe fn plus_i64_vec_vec_async(x: RefObj, y: RefObj, ctx: &Ctx<'_>) -> Result<RefObj, KernelErr> {
    let xs = x.as_slice::<i64>();
    let ys = y.as_slice::<i64>();
    if xs.len() != ys.len() { return Err(KernelErr::Shape); }
    let mut out = alloc_vec_i64(ctx, xs.len() as i64);
    let chunk = if ctx.chunk_elems != 0 { ctx.chunk_elems } else { I64_CHUNK };
    let has_nulls = (x.attr() | y.attr()) & attr_flags::HAS_NULLS != 0;
    {
        let os = out.as_mut_slice::<i64>();
        if !has_nulls {
            let mut k = AddI64VecVec::new(xs, ys, os, chunk);
            drive_async(&mut k, ctx).await?;
        } else {
            let mut k = AddI64VecVecNulls::new(xs, ys, os, chunk);
            drive_async(&mut k, ctx).await?;
        }
    }
    if has_nulls { out.set_attr(attr_flags::HAS_NULLS); }
    Ok(out)
}

/// I64+I64 kernel-pair (async). Used by `dispatch_plus_async`.
pub async unsafe fn plus_i64_i64_async(x: RefObj, y: RefObj, ctx: &Ctx<'_>) -> Result<RefObj, KernelErr> {
    match (x.is_atom(), y.is_atom()) {
        (true, true)  => Ok(plus_i64_atom_atom(x, y)),
        (true, false) => plus_i64_scalar_vec_async(x.atom::<i64>(), y, ctx).await,
        (false, true) => plus_i64_vec_scalar_async(x, y.atom::<i64>(), ctx).await,
        (false, false) => plus_i64_vec_vec_async(x, y, ctx).await,
    }
}

// =======================================================================
// I64 — sync shims (block_on wrappers around async)
// =======================================================================

pub unsafe fn plus_i64_i64(x: RefObj, y: RefObj, ctx: &Ctx) -> Result<RefObj, KernelErr> {
    block_on(plus_i64_i64_async(x, y, ctx))
}

// =======================================================================
// F64 — ChunkStep impls
// =======================================================================

pub struct AddF64VecVec<'a> {
    x: &'a [f64],
    y: &'a [f64],
    out: &'a mut [f64],
    off: usize,
    chunk: usize,
}

impl<'a> AddF64VecVec<'a> {
    #[inline]
    pub fn new(x: &'a [f64], y: &'a [f64], out: &'a mut [f64], chunk: usize) -> Self {
        debug_assert_eq!(x.len(), y.len());
        debug_assert_eq!(x.len(), out.len());
        Self { x, y, out, off: 0, chunk: chunk.max(1) }
    }
}

impl<'a> ChunkStep for AddF64VecVec<'a> {
    #[inline]
    fn step(&mut self) -> Option<usize> {
        if self.off >= self.x.len() { return None; }
        let end = (self.off + self.chunk).min(self.x.len());
        simd::add_f64(
            &mut self.out[self.off..end],
            &self.x[self.off..end],
            &self.y[self.off..end],
        );
        unsafe {
            madvise_dontneed_slice(&self.x[self.off..end]);
            madvise_dontneed_slice(&self.y[self.off..end]);
        }
        let n = end - self.off;
        self.off = end;
        Some(n)
    }
}

pub struct AddScalarVecF64<'a> {
    s: f64,
    x: &'a [f64],
    out: &'a mut [f64],
    off: usize,
    chunk: usize,
}

impl<'a> AddScalarVecF64<'a> {
    #[inline]
    pub fn new(s: f64, x: &'a [f64], out: &'a mut [f64], chunk: usize) -> Self {
        Self { s, x, out, off: 0, chunk: chunk.max(1) }
    }
}

impl<'a> ChunkStep for AddScalarVecF64<'a> {
    #[inline]
    fn step(&mut self) -> Option<usize> {
        if self.off >= self.x.len() { return None; }
        let end = (self.off + self.chunk).min(self.x.len());
        simd::add_scalar_f64(&mut self.out[self.off..end], &self.x[self.off..end], self.s);
        unsafe { madvise_dontneed_slice(&self.x[self.off..end]); }
        let n = end - self.off;
        self.off = end;
        Some(n)
    }
}

// =======================================================================
// F64 — async dispatch
// =======================================================================

#[inline]
pub unsafe fn plus_f64_atom_atom(x: RefObj, y: RefObj) -> RefObj {
    alloc_atom(Kind::F64, x.atom::<f64>() + y.atom::<f64>())
}

pub async unsafe fn plus_f64_scalar_vec_async(s: f64, y: RefObj, ctx: &Ctx<'_>) -> Result<RefObj, KernelErr> {
    let ys = y.as_slice::<f64>();
    let mut out = alloc_vec_f64(ctx, ys.len() as i64);
    let chunk = if ctx.chunk_elems != 0 { ctx.chunk_elems } else { F64_CHUNK };
    {
        let os = out.as_mut_slice::<f64>();
        let mut k = AddScalarVecF64::new(s, ys, os, chunk);
        drive_async(&mut k, ctx).await?;
    }
    if (y.attr() & attr_flags::HAS_NULLS) != 0 || s.is_nan() {
        out.set_attr(attr_flags::HAS_NULLS);
    }
    Ok(out)
}

pub async unsafe fn plus_f64_vec_scalar_async(x: RefObj, s: f64, ctx: &Ctx<'_>) -> Result<RefObj, KernelErr> {
    plus_f64_scalar_vec_async(s, x, ctx).await
}

pub async unsafe fn plus_f64_vec_vec_async(x: RefObj, y: RefObj, ctx: &Ctx<'_>) -> Result<RefObj, KernelErr> {
    let xs = x.as_slice::<f64>();
    let ys = y.as_slice::<f64>();
    if xs.len() != ys.len() { return Err(KernelErr::Shape); }
    let mut out = alloc_vec_f64(ctx, xs.len() as i64);
    let chunk = if ctx.chunk_elems != 0 { ctx.chunk_elems } else { F64_CHUNK };
    {
        let os = out.as_mut_slice::<f64>();
        let mut k = AddF64VecVec::new(xs, ys, os, chunk);
        drive_async(&mut k, ctx).await?;
    }
    if (x.attr() | y.attr()) & attr_flags::HAS_NULLS != 0 {
        out.set_attr(attr_flags::HAS_NULLS);
    }
    Ok(out)
}

pub async unsafe fn plus_f64_f64_async(x: RefObj, y: RefObj, ctx: &Ctx<'_>) -> Result<RefObj, KernelErr> {
    match (x.is_atom(), y.is_atom()) {
        (true, true)  => Ok(plus_f64_atom_atom(x, y)),
        (true, false) => plus_f64_scalar_vec_async(x.atom::<f64>(), y, ctx).await,
        (false, true) => plus_f64_vec_scalar_async(x, y.atom::<f64>(), ctx).await,
        (false, false) => plus_f64_vec_vec_async(x, y, ctx).await,
    }
}

// =======================================================================
// F64 — sync shim
// =======================================================================

pub unsafe fn plus_f64_f64(x: RefObj, y: RefObj, ctx: &Ctx) -> Result<RefObj, KernelErr> {
    block_on(plus_f64_f64_async(x, y, ctx))
}
