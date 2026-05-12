//! `&` (min) and `|` (max) dyadic kernels. Same-kind same-output (unlike
//! comparison kernels which produce Bool). Follows the plus/minus shape.

use crate::alloc::{alloc_atom, alloc_vec_f64, alloc_vec_i64};
use crate::chunk::{drive_async, drive_sync, ChunkStep};
use crate::ctx::{Ctx, KernelErr};
use crate::kind::Kind;
use crate::madvise::madvise_dontneed_slice;
use crate::obj::RefObj;
use crate::parallel;

#[derive(Copy, Clone)]
pub enum MinMax {
    Min,
    Max,
}

impl MinMax {
    #[inline]
    fn apply_i64(self, a: i64, b: i64) -> i64 {
        match self {
            MinMax::Min => a.min(b),
            MinMax::Max => a.max(b),
        }
    }
    #[inline]
    fn apply_f64(self, a: f64, b: f64) -> f64 {
        match self {
            // NaN handling: f64::min/max propagate NaN per IEEE 754-2008 if
            // either input is NaN. K9 with null-as-NaN would prefer
            // null-propagating, but null semantics are deferred (Phase 0
            // open question). For now we use the stdlib semantics.
            MinMax::Min => a.min(b),
            MinMax::Max => a.max(b),
        }
    }
}

const MM_CHUNK: usize = 64 * 1024;

pub async unsafe fn minmax_i64(
    op: MinMax,
    x: RefObj,
    y: RefObj,
    ctx: &Ctx<'_>,
) -> Result<RefObj, KernelErr> {
    if x.is_atom() && y.is_atom() {
        return Ok(alloc_atom(
            Kind::I64,
            op.apply_i64(x.atom::<i64>(), y.atom::<i64>()),
        ));
    }
    let (xs, xl) = i64_slice(&x);
    let (ys, yl) = i64_slice(&y);
    let n = broadcast_len(xl, yl)?;

    let mut out = alloc_vec_i64(ctx, n as i64);
    let os = (out.as_ptr() as *mut u8).add(16) as *mut i64;
    let total = n as usize;

    let go_parallel =
        ctx.runtime.parallel_enabled() && total >= parallel::PARALLEL_THRESHOLD;
    if go_parallel {
        let xs_a = xs as usize;
        let ys_a = ys as usize;
        let os_a = os as usize;
        let xs1 = xl == 1;
        let ys1 = yl == 1;
        let out_slice = core::slice::from_raw_parts_mut(os as *mut u8, total * 8);
        parallel::parallel_for_each_mut(out_slice, ctx, move |range, my_slice| {
            // Each byte slice covers `n_elems * 8` bytes; convert offset to
            // element index.
            let elem_start = range.start / 8;
            let elem_end = elem_start + my_slice.len() / 8;
            unsafe {
                let xs = xs_a as *const i64;
                let ys = ys_a as *const i64;
                let os = os_a as *mut i64;
                for i in elem_start..elem_end {
                    let a = if xs1 { *xs } else { *xs.add(i) };
                    let b = if ys1 { *ys } else { *ys.add(i) };
                    *os.add(i) = op.apply_i64(a, b);
                }
            }
            Ok(())
        })?;
    } else {
        let mut k = MmI64Step::new(op, xs, xl == 1, ys, yl == 1, os, total);
        drive_async(&mut k, ctx).await?;
    }

    let _ = drive_sync::<MmI64Step>;
    let _ = &mut out;
    Ok(out)
}

pub async unsafe fn minmax_f64(
    op: MinMax,
    x: RefObj,
    y: RefObj,
    ctx: &Ctx<'_>,
) -> Result<RefObj, KernelErr> {
    if x.is_atom() && y.is_atom() {
        return Ok(alloc_atom(
            Kind::F64,
            op.apply_f64(x.atom::<f64>(), y.atom::<f64>()),
        ));
    }
    let (xs, xl) = f64_slice(&x);
    let (ys, yl) = f64_slice(&y);
    let n = broadcast_len(xl, yl)?;

    let mut out = alloc_vec_f64(ctx, n as i64);
    let os = (out.as_ptr() as *mut u8).add(16) as *mut f64;
    let total = n as usize;
    let go_parallel =
        ctx.runtime.parallel_enabled() && total >= parallel::PARALLEL_THRESHOLD;
    if go_parallel {
        let xs_a = xs as usize;
        let ys_a = ys as usize;
        let os_a = os as usize;
        let xs1 = xl == 1;
        let ys1 = yl == 1;
        let out_slice = core::slice::from_raw_parts_mut(os as *mut u8, total * 8);
        parallel::parallel_for_each_mut(out_slice, ctx, move |range, my_slice| {
            let elem_start = range.start / 8;
            let elem_end = elem_start + my_slice.len() / 8;
            unsafe {
                let xs = xs_a as *const f64;
                let ys = ys_a as *const f64;
                let os = os_a as *mut f64;
                for i in elem_start..elem_end {
                    let a = if xs1 { *xs } else { *xs.add(i) };
                    let b = if ys1 { *ys } else { *ys.add(i) };
                    *os.add(i) = op.apply_f64(a, b);
                }
            }
            Ok(())
        })?;
    } else {
        let mut k = MmF64Step::new(op, xs, xl == 1, ys, yl == 1, os, total);
        drive_async(&mut k, ctx).await?;
    }
    let _ = &mut out;
    Ok(out)
}

#[inline]
fn broadcast_len(xl: i64, yl: i64) -> Result<i64, KernelErr> {
    if xl == yl {
        Ok(xl)
    } else if xl == 1 {
        Ok(yl)
    } else if yl == 1 {
        Ok(xl)
    } else {
        Err(KernelErr::Shape)
    }
}

#[inline]
unsafe fn i64_slice(x: &RefObj) -> (*const i64, i64) {
    if x.is_atom() {
        ((x.as_ptr() as *const u8).add(8) as *const i64, 1)
    } else {
        ((x.as_ptr() as *const u8).add(16) as *const i64, x.len())
    }
}
#[inline]
unsafe fn f64_slice(x: &RefObj) -> (*const f64, i64) {
    if x.is_atom() {
        ((x.as_ptr() as *const u8).add(8) as *const f64, 1)
    } else {
        ((x.as_ptr() as *const u8).add(16) as *const f64, x.len())
    }
}

struct MmI64Step {
    op: MinMax,
    xs: *const i64,
    xs1: bool,
    ys: *const i64,
    ys1: bool,
    out: *mut i64,
    total: usize,
    off: usize,
}
impl MmI64Step {
    fn new(
        op: MinMax,
        xs: *const i64,
        xs1: bool,
        ys: *const i64,
        ys1: bool,
        out: *mut i64,
        total: usize,
    ) -> Self {
        Self { op, xs, xs1, ys, ys1, out, total, off: 0 }
    }
}
impl ChunkStep for MmI64Step {
    fn step(&mut self) -> Option<usize> {
        if self.off >= self.total {
            return None;
        }
        let end = (self.off + MM_CHUNK).min(self.total);
        unsafe {
            for i in self.off..end {
                let a = if self.xs1 { *self.xs } else { *self.xs.add(i) };
                let b = if self.ys1 { *self.ys } else { *self.ys.add(i) };
                *self.out.add(i) = self.op.apply_i64(a, b);
            }
            if !self.xs1 {
                let s = core::slice::from_raw_parts(self.xs.add(self.off), end - self.off);
                madvise_dontneed_slice(s);
            }
            if !self.ys1 {
                let s = core::slice::from_raw_parts(self.ys.add(self.off), end - self.off);
                madvise_dontneed_slice(s);
            }
        }
        let n = end - self.off;
        self.off = end;
        Some(n)
    }
}

struct MmF64Step {
    op: MinMax,
    xs: *const f64,
    xs1: bool,
    ys: *const f64,
    ys1: bool,
    out: *mut f64,
    total: usize,
    off: usize,
}
impl MmF64Step {
    fn new(
        op: MinMax,
        xs: *const f64,
        xs1: bool,
        ys: *const f64,
        ys1: bool,
        out: *mut f64,
        total: usize,
    ) -> Self {
        Self { op, xs, xs1, ys, ys1, out, total, off: 0 }
    }
}
impl ChunkStep for MmF64Step {
    fn step(&mut self) -> Option<usize> {
        if self.off >= self.total {
            return None;
        }
        let end = (self.off + MM_CHUNK).min(self.total);
        unsafe {
            for i in self.off..end {
                let a = if self.xs1 { *self.xs } else { *self.xs.add(i) };
                let b = if self.ys1 { *self.ys } else { *self.ys.add(i) };
                *self.out.add(i) = self.op.apply_f64(a, b);
            }
            if !self.xs1 {
                let s = core::slice::from_raw_parts(self.xs.add(self.off), end - self.off);
                madvise_dontneed_slice(s);
            }
            if !self.ys1 {
                let s = core::slice::from_raw_parts(self.ys.add(self.off), end - self.off);
                madvise_dontneed_slice(s);
            }
        }
        let n = end - self.off;
        self.off = end;
        Some(n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alloc::{alloc_atom, alloc_vec_i64};
    use crate::exec::block_on;

    fn make_i64_vec(d: &[i64]) -> RefObj {
        let ctx = Ctx::quiet();
        let mut v = unsafe { alloc_vec_i64(&ctx, d.len() as i64) };
        unsafe { v.as_mut_slice::<i64>().copy_from_slice(d); }
        v
    }

    #[test]
    fn min_atoms() {
        let ctx = Ctx::quiet();
        let a = unsafe { alloc_atom(Kind::I64, 5i64) };
        let b = unsafe { alloc_atom(Kind::I64, 3i64) };
        let r = block_on(async { unsafe { minmax_i64(MinMax::Min, a, b, &ctx).await.unwrap() } });
        assert_eq!(unsafe { r.atom::<i64>() }, 3);
    }

    #[test]
    fn max_vec_vec() {
        let ctx = Ctx::quiet();
        let x = make_i64_vec(&[1, 5, 3]);
        let y = make_i64_vec(&[2, 4, 7]);
        let r = block_on(async { unsafe { minmax_i64(MinMax::Max, x, y, &ctx).await.unwrap() } });
        assert_eq!(unsafe { r.as_slice::<i64>() }, &[2, 5, 7]);
    }

    #[test]
    fn min_vec_scalar_broadcast() {
        let ctx = Ctx::quiet();
        let x = make_i64_vec(&[1, 5, 3, 10]);
        let y = unsafe { alloc_atom(Kind::I64, 3i64) };
        let r = block_on(async { unsafe { minmax_i64(MinMax::Min, x, y, &ctx).await.unwrap() } });
        assert_eq!(unsafe { r.as_slice::<i64>() }, &[1, 3, 3, 3]);
    }
}
