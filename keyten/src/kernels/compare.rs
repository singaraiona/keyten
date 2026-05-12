//! Comparison verb kernels: `=`, `<`, `>`, `~`.
//!
//! All four are bool-producing. `=`/`<`/`>` are pairwise — atom-atom gives a
//! Bool atom; vec-vec (same length) gives a Bool vector; vec-atom broadcasts.
//! `~` is "match", a single Bool atom for any input pair, true iff the inputs
//! are byte-identical at the kernel level (same kind, length, content).
//!
//! Bool result is stored as 1 byte per element (the K9 boolean storage we
//! already use). Chunked + parallel per the performance mandate.

use crate::alloc::{alloc_atom, alloc_vec};
use crate::chunk::{drive_async, drive_sync, ChunkStep};
use crate::ctx::{Ctx, KernelErr};
use crate::kind::Kind;
use crate::madvise::madvise_dontneed_slice;
use crate::obj::RefObj;
use crate::parallel;

/// Comparison operator selector — used to monomorphise the kernel for `=`,
/// `<`, `>` with a single body.
#[derive(Copy, Clone)]
pub enum Cmp {
    Eq,
    Lt,
    Gt,
}

impl Cmp {
    #[inline]
    fn apply_i64(self, a: i64, b: i64) -> bool {
        match self {
            Cmp::Eq => a == b,
            Cmp::Lt => a < b,
            Cmp::Gt => a > b,
        }
    }

    #[inline]
    fn apply_f64(self, a: f64, b: f64) -> bool {
        match self {
            // NaN comparisons follow Rust semantics: NaN != anything,
            // NaN-vs-anything ordering is false. K9 treats nulls (NaN for
            // f64) specially — we may revisit when nulls land per Phase 0.
            Cmp::Eq => a == b,
            Cmp::Lt => a < b,
            Cmp::Gt => a > b,
        }
    }
}

const CMP_CHUNK: usize = 64 * 1024;

/// `x cmp y` for I64 atom/vector operands. Returns a Bool atom (atom-atom)
/// or a Bool vector (any vec involved).
pub async unsafe fn compare_i64(
    cmp: Cmp,
    x: RefObj,
    y: RefObj,
    ctx: &Ctx<'_>,
) -> Result<RefObj, KernelErr> {
    if x.is_atom() && y.is_atom() {
        let a = x.atom::<i64>();
        let b = y.atom::<i64>();
        let r = if cmp.apply_i64(a, b) { 1u8 } else { 0u8 };
        return Ok(alloc_atom(Kind::Bool, r));
    }

    let (xs_ptr, xs_len) = slice_or_atom_i64(&x);
    let (ys_ptr, ys_len) = slice_or_atom_i64(&y);

    // Lengths must match unless one side is a scalar (length 1 from atom).
    let n = if xs_len == ys_len {
        xs_len
    } else if xs_len == 1 {
        ys_len
    } else if ys_len == 1 {
        xs_len
    } else {
        return Err(KernelErr::Shape);
    };

    let mut out = alloc_vec(ctx, Kind::Bool.vec(), n as i64, Kind::Bool.elem_size());
    let os_ptr = (out.as_ptr() as *mut u8).add(16);
    let total = n as usize;

    let go_parallel =
        ctx.runtime.parallel_enabled() && total >= parallel::PARALLEL_THRESHOLD;
    if go_parallel {
        let xs_addr = xs_ptr as usize;
        let ys_addr = ys_ptr as usize;
        let x_scalar = xs_len == 1;
        let y_scalar = ys_len == 1;
        let out_slice = core::slice::from_raw_parts_mut(os_ptr, total);
        parallel::parallel_for_each_mut(out_slice, ctx, move |range, my_slice| {
            unsafe {
                let xs = xs_addr as *const i64;
                let ys = ys_addr as *const i64;
                for (local, byte) in my_slice.iter_mut().enumerate() {
                    let g = range.start + local;
                    let a = if x_scalar { *xs } else { *xs.add(g) };
                    let b = if y_scalar { *ys } else { *ys.add(g) };
                    *byte = if cmp.apply_i64(a, b) { 1 } else { 0 };
                }
            }
            Ok(())
        })?;
    } else {
        let mut k = CmpI64Step::new(cmp, xs_ptr, xs_len == 1, ys_ptr, ys_len == 1, os_ptr, total);
        drive_async(&mut k, ctx).await?;
    }

    let _ = &mut out;
    let _ = drive_sync::<CmpI64Step>;
    Ok(out)
}

/// `x cmp y` for F64 operands. Mirror of the I64 path.
pub async unsafe fn compare_f64(
    cmp: Cmp,
    x: RefObj,
    y: RefObj,
    ctx: &Ctx<'_>,
) -> Result<RefObj, KernelErr> {
    if x.is_atom() && y.is_atom() {
        let r = if cmp.apply_f64(x.atom::<f64>(), y.atom::<f64>()) { 1u8 } else { 0u8 };
        return Ok(alloc_atom(Kind::Bool, r));
    }
    let (xs_ptr, xs_len) = slice_or_atom_f64(&x);
    let (ys_ptr, ys_len) = slice_or_atom_f64(&y);
    let n = if xs_len == ys_len {
        xs_len
    } else if xs_len == 1 {
        ys_len
    } else if ys_len == 1 {
        xs_len
    } else {
        return Err(KernelErr::Shape);
    };

    let mut out = alloc_vec(ctx, Kind::Bool.vec(), n as i64, Kind::Bool.elem_size());
    let os_ptr = (out.as_ptr() as *mut u8).add(16);
    let total = n as usize;
    let go_parallel =
        ctx.runtime.parallel_enabled() && total >= parallel::PARALLEL_THRESHOLD;

    if go_parallel {
        let xs_addr = xs_ptr as usize;
        let ys_addr = ys_ptr as usize;
        let x_scalar = xs_len == 1;
        let y_scalar = ys_len == 1;
        let out_slice = core::slice::from_raw_parts_mut(os_ptr, total);
        parallel::parallel_for_each_mut(out_slice, ctx, move |range, my_slice| {
            unsafe {
                let xs = xs_addr as *const f64;
                let ys = ys_addr as *const f64;
                for (local, byte) in my_slice.iter_mut().enumerate() {
                    let g = range.start + local;
                    let a = if x_scalar { *xs } else { *xs.add(g) };
                    let b = if y_scalar { *ys } else { *ys.add(g) };
                    *byte = if cmp.apply_f64(a, b) { 1 } else { 0 };
                }
            }
            Ok(())
        })?;
    } else {
        let mut k = CmpF64Step::new(cmp, xs_ptr, xs_len == 1, ys_ptr, ys_len == 1, os_ptr, total);
        drive_async(&mut k, ctx).await?;
    }

    let _ = &mut out;
    Ok(out)
}

/// `x ~ y` — match. Deep byte-level equality of two RefObjs. Returns a
/// single Bool atom (0b or 1b). Different kind ⇒ 0; different length ⇒ 0;
/// otherwise byte-for-byte compare of the payloads.
pub fn match_objs(x: &RefObj, y: &RefObj) -> RefObj {
    let xk = x.kind_raw();
    let yk = y.kind_raw();
    let eq = if xk != yk {
        false
    } else if x.is_atom() {
        // Both atoms of the same kind — compare elem_size bytes of payload.
        let elem = Kind::from_raw(xk).elem_size();
        unsafe {
            let a = (x.as_ptr() as *const u8).add(8);
            let b = (y.as_ptr() as *const u8).add(8);
            core::slice::from_raw_parts(a, elem) == core::slice::from_raw_parts(b, elem)
        }
    } else {
        // Both vectors of the same kind.
        let xl = x.len();
        if xl != y.len() {
            false
        } else {
            let elem = Kind::from_raw(xk).elem_size();
            let bytes = xl as usize * elem;
            unsafe {
                let a = (x.as_ptr() as *const u8).add(16);
                let b = (y.as_ptr() as *const u8).add(16);
                core::slice::from_raw_parts(a, bytes) == core::slice::from_raw_parts(b, bytes)
            }
        }
    };
    let r = if eq { 1u8 } else { 0u8 };
    unsafe { alloc_atom(Kind::Bool, r) }
}

// ---- helpers + state machines ----

#[inline]
unsafe fn slice_or_atom_i64(x: &RefObj) -> (*const i64, i64) {
    if x.is_atom() {
        ((x.as_ptr() as *const u8).add(8) as *const i64, 1)
    } else {
        (
            (x.as_ptr() as *const u8).add(16) as *const i64,
            x.len(),
        )
    }
}

#[inline]
unsafe fn slice_or_atom_f64(x: &RefObj) -> (*const f64, i64) {
    if x.is_atom() {
        ((x.as_ptr() as *const u8).add(8) as *const f64, 1)
    } else {
        (
            (x.as_ptr() as *const u8).add(16) as *const f64,
            x.len(),
        )
    }
}

struct CmpI64Step {
    cmp: Cmp,
    xs: *const i64,
    x_scalar: bool,
    ys: *const i64,
    y_scalar: bool,
    out: *mut u8,
    total: usize,
    off: usize,
}

impl CmpI64Step {
    fn new(
        cmp: Cmp,
        xs: *const i64,
        x_scalar: bool,
        ys: *const i64,
        y_scalar: bool,
        out: *mut u8,
        total: usize,
    ) -> Self {
        Self { cmp, xs, x_scalar, ys, y_scalar, out, total, off: 0 }
    }
}

impl ChunkStep for CmpI64Step {
    fn step(&mut self) -> Option<usize> {
        if self.off >= self.total {
            return None;
        }
        let end = (self.off + CMP_CHUNK).min(self.total);
        unsafe {
            for i in self.off..end {
                let a = if self.x_scalar { *self.xs } else { *self.xs.add(i) };
                let b = if self.y_scalar { *self.ys } else { *self.ys.add(i) };
                *self.out.add(i) = if self.cmp.apply_i64(a, b) { 1 } else { 0 };
            }
            if !self.x_scalar {
                let consumed = core::slice::from_raw_parts(self.xs.add(self.off), end - self.off);
                madvise_dontneed_slice(consumed);
            }
            if !self.y_scalar {
                let consumed = core::slice::from_raw_parts(self.ys.add(self.off), end - self.off);
                madvise_dontneed_slice(consumed);
            }
        }
        let n = end - self.off;
        self.off = end;
        Some(n)
    }
}

struct CmpF64Step {
    cmp: Cmp,
    xs: *const f64,
    x_scalar: bool,
    ys: *const f64,
    y_scalar: bool,
    out: *mut u8,
    total: usize,
    off: usize,
}

impl CmpF64Step {
    fn new(
        cmp: Cmp,
        xs: *const f64,
        x_scalar: bool,
        ys: *const f64,
        y_scalar: bool,
        out: *mut u8,
        total: usize,
    ) -> Self {
        Self { cmp, xs, x_scalar, ys, y_scalar, out, total, off: 0 }
    }
}

impl ChunkStep for CmpF64Step {
    fn step(&mut self) -> Option<usize> {
        if self.off >= self.total {
            return None;
        }
        let end = (self.off + CMP_CHUNK).min(self.total);
        unsafe {
            for i in self.off..end {
                let a = if self.x_scalar { *self.xs } else { *self.xs.add(i) };
                let b = if self.y_scalar { *self.ys } else { *self.ys.add(i) };
                *self.out.add(i) = if self.cmp.apply_f64(a, b) { 1 } else { 0 };
            }
            if !self.x_scalar {
                let consumed = core::slice::from_raw_parts(self.xs.add(self.off), end - self.off);
                madvise_dontneed_slice(consumed);
            }
            if !self.y_scalar {
                let consumed = core::slice::from_raw_parts(self.ys.add(self.off), end - self.off);
                madvise_dontneed_slice(consumed);
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

    fn make_i64_vec(data: &[i64]) -> RefObj {
        let ctx = Ctx::quiet();
        let mut v = unsafe { alloc_vec_i64(&ctx, data.len() as i64) };
        unsafe { v.as_mut_slice::<i64>().copy_from_slice(data); }
        v
    }

    #[test]
    fn eq_atoms() {
        let ctx = Ctx::quiet();
        let a = unsafe { alloc_atom(Kind::I64, 5i64) };
        let b = unsafe { alloc_atom(Kind::I64, 5i64) };
        let r = block_on(async { unsafe { compare_i64(Cmp::Eq, a, b, &ctx).await.unwrap() } });
        assert_eq!(r.kind(), Kind::Bool);
        assert_eq!(unsafe { r.atom::<u8>() }, 1);
    }

    #[test]
    fn lt_vec_vec() {
        let ctx = Ctx::quiet();
        let x = make_i64_vec(&[1, 5, 3]);
        let y = make_i64_vec(&[2, 5, 1]);
        let r = block_on(async { unsafe { compare_i64(Cmp::Lt, x, y, &ctx).await.unwrap() } });
        assert_eq!(r.kind(), Kind::Bool);
        assert_eq!(r.len(), 3);
        let s = unsafe { r.as_slice::<u8>() };
        assert_eq!(s, &[1, 0, 0]);
    }

    #[test]
    fn gt_vec_atom_broadcast() {
        let ctx = Ctx::quiet();
        let x = make_i64_vec(&[1, 5, 3, 10]);
        let y = unsafe { alloc_atom(Kind::I64, 3i64) };
        let r = block_on(async { unsafe { compare_i64(Cmp::Gt, x, y, &ctx).await.unwrap() } });
        let s = unsafe { r.as_slice::<u8>() };
        assert_eq!(s, &[0, 1, 0, 1]);
    }

    #[test]
    fn match_objs_equal_atoms() {
        let a = unsafe { alloc_atom(Kind::I64, 42i64) };
        let b = unsafe { alloc_atom(Kind::I64, 42i64) };
        let r = match_objs(&a, &b);
        assert_eq!(unsafe { r.atom::<u8>() }, 1);
    }

    #[test]
    fn match_objs_different_kind_is_zero() {
        let a = unsafe { alloc_atom(Kind::I64, 1i64) };
        let b = unsafe { alloc_atom(Kind::F64, 1.0f64) };
        let r = match_objs(&a, &b);
        assert_eq!(unsafe { r.atom::<u8>() }, 0);
    }

    #[test]
    fn match_objs_vectors() {
        let a = make_i64_vec(&[1, 2, 3]);
        let b = make_i64_vec(&[1, 2, 3]);
        let r = match_objs(&a, &b);
        assert_eq!(unsafe { r.atom::<u8>() }, 1);
    }

    #[test]
    fn match_objs_different_length() {
        let a = make_i64_vec(&[1, 2, 3]);
        let b = make_i64_vec(&[1, 2]);
        let r = match_objs(&a, &b);
        assert_eq!(unsafe { r.atom::<u8>() }, 0);
    }
}
